# meristats, a testing and results framework for the Meridian chess engine
# Copyright (C) 2026 Syed Taha
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import hashlib
import io
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import chess.pgn
import pyarrow as pa
import pyarrow.parquet as pq

from meristats import datasets, preflight, runner, stats
from meristats.build import BuiltEngine
from meristats.config import Config
from meristats.preflight import Check, require
from meristats.store import Store

PROTOCOL = "match"
VERSION = 1
LADDER = "ladder"
LADDER_VERSION = 1
ROUNDS = 1500
VERDICT = re.compile(r"SPRT \(\[[^]]*\]\) completed - (H[01]) was accepted")


class MatchError(Exception):
    """
    fastchess failed, or its games do not have the shape that the protocol needs.
    """


@dataclass(frozen=True)
class Limit:
    """
    How the search of a move is limited for both sides: `st` seconds per move, `depth` plies, or
    `nodes`.
    """

    kind: str
    value: float

    def __post_init__(self):
        if self.kind not in ("st", "depth", "nodes") or self.value <= 0:
            raise ValueError(f"not a limit: {self.kind} {self.value}")

    def argument(self) -> str:
        return (
            f"{self.kind}={self.value:g}" if self.kind == "st" else f"{self.kind}={int(self.value)}"
        )


@dataclass(frozen=True)
class Sprt:
    """
    The sequential test: Elo bounds and the two error probabilities.
    """

    elo0: float = 0.0
    elo1: float = 10.0
    alpha: float = 0.05
    beta: float = 0.05


@dataclass(frozen=True)
class Player:
    """
    One side of a match: its program, the UCI options that it gets, and its engine row.
    """

    name: str
    binary: Path
    options: dict
    engine_id: int


@dataclass(frozen=True)
class Game:
    number: int
    round: int
    opening: str
    white: str
    black: str
    result: str
    termination: str
    plies: int


def build_command(
    fastchess: Path,
    a: Player,
    b: Player,
    limit: Limit,
    book: Path,
    *,
    rounds: int,
    concurrency: int,
    seed: int,
    pgn: Path,
    log: Path,
    sprt: Sprt | None,
) -> list[str]:
    """
    The fastchess command of a match. Each opening is played twice with the colors swapped
    (`-games 2 -repeat`), and the openings come in random order from a fixed seed.
    """
    command = [str(fastchess)]

    for label, player in (("A", a), ("B", b)):
        command += ["-engine", f"cmd={player.binary}", f"name={label}"]
        command += [f"option.{name}={value}" for name, value in player.options.items()]

    command += [
        "-each",
        limit.argument(),
        "-openings",
        f"file={book}",
        "format=pgn",
        "order=random",
        "-srand",
        str(seed),
        "-games",
        "2",
        "-repeat",
        "-rounds",
        str(rounds),
        "-concurrency",
        str(concurrency),
        "-recover",
        "-pgnout",
        f"file={pgn}",
        "nodes=true",
        "timeleft=true",
        "-log",
        f"file={log}",
        "level=info",
    ]

    if sprt is not None:
        command += [
            "-sprt",
            f"elo0={sprt.elo0:g}",
            f"elo1={sprt.elo1:g}",
            f"alpha={sprt.alpha:g}",
            f"beta={sprt.beta:g}",
            "model=logistic",
        ]

    return command


def parse_games(text: str) -> list[Game]:
    """
    Reads the games of a PGN file written by fastchess.
    """
    games = []
    handle = io.StringIO(text)

    while True:
        game = chess.pgn.read_game(handle)

        if game is None:
            return games

        headers = game.headers
        games.append(
            Game(
                len(games) + 1,
                int(headers.get("Round", "0")),
                headers.get("Opening", "?"),
                headers.get("White", "?"),
                headers.get("Black", "?"),
                headers.get("Result", "*"),
                headers.get("Termination", "?"),
                int(headers.get("PlyCount", "0")),
            )
        )


def points_of(game: Game, name: str) -> int | None:
    """
    The points of `name` in a game, in half points (0, 1 or 2), or None for an unfinished game
    or a game that the player did not play.
    """
    if name not in (game.white, game.black) or game.result not in ("1-0", "0-1", "1/2-1/2"):
        return None

    if game.result == "1/2-1/2":
        return 1

    return 2 if (game.result == "1-0") == (game.white == name) else 0


def pair_results(games: list[Game], name: str = "A") -> list[tuple[int, int]]:
    """
    The results of the pairs of games, each as the points of `name` in the two games of a round.
    A round without two finished games does not count.
    """
    rounds: dict[int, list[int]] = {}

    for game in games:
        points = points_of(game, name)

        if points is not None:
            rounds.setdefault(game.round, []).append(points)

    return [(r[0], r[1]) for _, r in sorted(rounds.items()) if len(r) == 2]


def fastchess_verdict(output: str) -> str | None:
    """
    `H1` or `H0` when fastchess ended the match with a decision of its sequential test.
    """
    found = VERDICT.search(output)
    return found.group(1) if found else None


def summarise(games: list[Game], sprt: Sprt | None, output: str) -> dict:
    """
    The result of a match for engine A: wins, draws and losses, the score with its interval, the
    Elo difference, the pentanomial counts, and when a test ran, the log likelihood ratio and the
    verdicts of meristats and of fastchess.
    """
    points = [p for p in (points_of(g, "A") for g in games) if p is not None]
    wins, draws, losses = points.count(2), points.count(1), points.count(0)
    summary: dict = {
        "games": len(games),
        "finished": len(points),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "terminations": {
            t: sum(1 for g in games if g.termination == t)
            for t in sorted({g.termination for g in games})
        },
    }

    if points:
        result = stats.match_summary(wins, draws, losses)
        summary |= {
            "score": result.score,
            "score_low": result.score_low,
            "score_high": result.score_high,
            "elo": result.elo,
            "elo_low": result.elo_low,
            "elo_high": result.elo_high,
        }

    pairs = stats.pentanomial(pair_results(games))
    summary["pentanomial"] = pairs
    summary["pairs"] = sum(pairs)

    if sprt is not None:
        llr = stats.sprt_llr(pairs, sprt.elo0, sprt.elo1)
        ours = stats.sprt_decision(llr, sprt.alpha, sprt.beta)
        theirs = fastchess_verdict(output)
        low, high = stats.sprt_bounds(sprt.alpha, sprt.beta)
        summary["sprt"] = {
            "elo0": sprt.elo0,
            "elo1": sprt.elo1,
            "alpha": sprt.alpha,
            "beta": sprt.beta,
            "llr": llr,
            "bounds": [low, high],
            "verdict": ours,
            "fastchess_verdict": theirs or "continue",
            "agree": ours == (theirs or "continue"),
        }

    return summary


def games_table(games: list[Game]) -> bytes:
    table = pa.table(
        {
            "game": pa.array([g.number for g in games], pa.uint32()),
            "round": pa.array([g.round for g in games], pa.uint32()),
            "opening": pa.array([g.opening for g in games], pa.string()),
            "white": pa.array([g.white for g in games], pa.string()),
            "black": pa.array([g.black for g in games], pa.string()),
            "result": pa.array([g.result for g in games], pa.string()),
            "termination": pa.array([g.termination for g in games], pa.string()),
            "plies": pa.array([g.plies for g in games], pa.uint16()),
        }
    )
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="NONE")
    return buffer.getvalue()


def compliance(fastchess: Path, binary: Path) -> Check:
    """
    Runs `fastchess --compliance` on a program and checks that it passes.
    """
    try:
        done = subprocess.run(
            [str(fastchess), "--compliance", str(binary)],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return Check("compliance", False, f"{binary.name}: {error}")

    passed = "passed all compliance checks" in done.stdout
    return Check(
        "compliance", passed, f"{binary.name}: " + ("passed" if passed else done.stdout[-200:])
    )


def play(
    config: Config,
    a: Player,
    b: Player,
    limit: Limit,
    book: Path,
    work: Path,
    *,
    rounds: int,
    sprt: Sprt | None,
    seed: int,
    concurrency: int | None,
) -> tuple[str, str]:
    """
    Runs fastchess in `work` and returns the PGN text and the output of the program.
    """
    if config.fastchess is None or not Path(config.fastchess).exists():
        raise MatchError("the settings give no fastchess program")

    if work.exists():
        shutil.rmtree(work)

    work.mkdir(parents=True)
    pgn, log = work / "games.pgn", work / "fastchess.log"
    command = build_command(
        Path(config.fastchess),
        a,
        b,
        limit,
        book.resolve(),
        rounds=rounds,
        concurrency=concurrency or config.workers,
        seed=seed,
        pgn=pgn,
        log=log,
        sprt=sprt,
    )
    done = subprocess.run(command, capture_output=True, text=True, cwd=work)
    output = done.stdout + done.stderr
    (work / "output.txt").write_text(output, encoding="utf-8")

    if done.returncode != 0 or not pgn.exists():
        raise MatchError(f"fastchess failed ({done.returncode}): {output[-400:]}")

    return pgn.read_text(encoding="utf-8"), output


def run(
    config: Config,
    store: Store,
    engine_a: BuiltEngine,
    engine_b: BuiltEngine | None,
    book: Path,
    limit: Limit,
    *,
    a_config: str | None = None,
    b_config: str | None = None,
    a_eval_file: Path | None = None,
    b_eval_file: Path | None = None,
    b_player: Player | None = None,
    rounds: int = ROUNDS,
    sprt: Sprt | None = Sprt(),
    seed: int = 1,
    concurrency: int | None = None,
    hash_mb: int = 16,
    allow_dirty: bool = False,
    framework=...,
    check_compliance: bool = True,
    progress=lambda text: print(text, file=sys.stderr, flush=True),
) -> tuple[str, dict]:
    """
    Plays engine A against engine B and stores the games and the result.

    B is a built engine or, with `b_player`, an external program. Both sides get one thread and a
    fixed hash. A time limit gets `MoveOverhead 0` for a Meridian engine, so that `st` is the
    time that the search takes. Each side may get a `Config`. A run with a sequential test
    stops at its decision, and meristats computes the test again from the games, so both
    verdicts are stored.
    """
    context = runner.begin(config, store, allow_dirty=allow_dirty, framework=framework)
    book_id, book_hash = datasets.register_file(
        store, "opening-book", "8moves_v3", book, {"file": book.name}, keep_copy=False
    )

    def meridian_options(config_value, eval_file):
        options = {"Threads": 1, "Hash": hash_mb}

        if limit.kind == "st":
            options["MoveOverhead"] = 0

        if eval_file is not None:
            options["EvalFile"] = str(Path(eval_file).resolve())

        if config_value is not None:
            options["Config"] = config_value

        return options

    a = Player("A", engine_a.binary, meridian_options(a_config, a_eval_file), engine_a.engine_id)
    b = b_player or Player(
        "B", engine_b.binary, meridian_options(b_config, b_eval_file), engine_b.engine_id
    )
    checks = runner.engine_checks(engine_a)

    if engine_b is not None and b_player is None:
        checks += runner.engine_checks(engine_b)

    checks.append(preflight.dataset_hash(book, book_hash))

    if check_compliance and config.fastchess:
        checks += [
            compliance(Path(config.fastchess), a.binary),
            compliance(Path(config.fastchess), b.binary),
        ]

    context.checks += require(checks)
    context.sample("before")
    work = config.cache_dir / "matches" / context.started_utc.replace(":", "")
    pgn_text, output = play(
        config,
        a,
        b,
        limit,
        book,
        work,
        rounds=rounds,
        sprt=sprt,
        seed=seed,
        concurrency=concurrency,
    )
    context.sample("end")
    games = parse_games(pgn_text)

    if not games:
        raise MatchError("fastchess wrote no games")

    summary = summarise(games, sprt, output)
    progress(
        f"{summary['games']} games: +{summary['wins']} ={summary['draws']} -{summary['losses']}"
    )
    pgn_blob = store.put_blob(pgn_text.encode(), "pgn", "games.pgn")
    log_blob = store.put_blob(output.encode(), "tool-log", "fastchess-output.txt")
    items = store.put_blob(games_table(games), "items", f"{PROTOCOL}-games.parquet")
    limits = {
        "limit": {limit.kind: limit.value},
        "rounds": rounds,
        "seed": seed,
        "options_a": a.options,
        "options_b": b.options,
        "b_engine_id": b.engine_id,
        "eval_files": {
            side: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for side, path in (("a", a_eval_file), ("b", b_eval_file))
            if path is not None
        },
        "concurrency": concurrency or config.workers,
        "sprt": None if sprt is None else asdict(sprt),
    }
    run_id = runner.finish(
        context,
        protocol=PROTOCOL,
        version=VERSION,
        engine=engine_a,
        config=f"{a_config or 'default'} vs {b_config or 'default'}",
        limits=limits,
        dataset_id=book_id,
        summary=summary,
        items_blob=items.sha256,
        seed=seed,
        blobs=((pgn_blob.sha256, "pgn"), (log_blob.sha256, "tool-log"), (items.sha256, "items")),
        notes=None if summary.get("sprt", {}).get("agree", True) else "sprt_disagree",
    )
    return run_id, summary
