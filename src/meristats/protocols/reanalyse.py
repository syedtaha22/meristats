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


"""
The reanalyse protocol: the games of an account are searched again, position by position, deeper
than they were played, so that the engine can learn from them.

Each position is searched to a fixed depth. The result gives three things:

- the quiet positions, labelled with the deeper score and the result of the game, as data for
  training, in the format of the `positions` protocol
- the mistakes: the moves of the account that gave away at least `mistake_cp` centipawns, as the
  difference between the score of the position before the move and the score of the one after it
- the experience: the best move, score and depth of every position where the account was to
  move, for the experience file of the engine
"""

import hashlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import chess

from meristats import experience, lichess, preflight, runner, zobrist
from meristats.build import BuiltEngine
from meristats.config import Config
from meristats.preflight import require
from meristats.protocols.positions import white_score
from meristats.store import Store
from meristats.uci import UciEngine

PROTOCOL = "reanalyse"
VERSION = 1
MISTAKE_CP = 100
CAP_CP = 3000


class ReanalyseError(RuntimeError):
    pass


def capped(kind: str, value: int) -> int:
    """
    A score for the comparison of two positions: a mate counts as `CAP_CP`.
    """
    if kind == "mate":
        return CAP_CP if value > 0 else -CAP_CP

    return max(-CAP_CP, min(CAP_CP, value))


def loss_of(before: int, after: int) -> int:
    """
    What a move gave away. `before` is the score of the position that the move was played in, for
    the side that played it, and `after` is the score of the position that it led to, for the side
    that moved next, so that the played move is worth `-after`.
    """
    return before + after


def analyse_game(uci: UciEngine, game: dict, user: str, depth: int) -> dict:
    """
    Searches every position of a game and returns the rows of the game: one for each position
    with its score, best move and the move that was played.
    """
    mine = lichess.side_of(game, user)
    rows = []

    for board, played in lichess.boards_of(game):
        is_mine = board.turn == (chess.WHITE if mine else chess.BLACK)

        if not any(board.legal_moves):
            kind, value = ("mate", -1) if board.is_check() else ("cp", 0)
            rows.append(
                {"terminal": True, "kind": kind, "value": value, "mine": is_mine, "loss": None}
            )
            continue

        fen = board.fen(en_passant="fen")
        uci.new_game()
        uci.position(fen, [])
        answer = uci.go(f"depth {depth}", timeout=600)
        scored = [info for info in answer.infos if info.score_kind is not None]

        if not scored:
            raise ReanalyseError(f"no score for {fen} in game {game['id']}")

        last = scored[-1]
        best = chess.Move.from_uci(answer.bestmove)

        if best not in board.legal_moves:
            raise ReanalyseError(f"illegal move {answer.bestmove} for {fen}")

        plain, white = white_score((last.score_kind, last.score), board.turn)
        rows.append(
            {
                "terminal": False,
                "loss": None,
                "fen": fen,
                "mine": is_mine,
                "kind": last.score_kind,
                "value": last.score,
                "white": white,
                "depth": last.depth or depth,
                "best": best.uci(),
                "best_data": zobrist.move_data(board, best),
                "key": zobrist.engine_key(board),
                "played": played.uci() if played else None,
                "quiet": bool(
                    plain
                    and not board.is_check()
                    and not board.is_capture(best)
                    and best.promotion is None
                ),
            }
        )

    for index, row in enumerate(rows[:-1]):
        if row["terminal"] or not row["mine"] or row["played"] is None:
            row["loss"] = None
            continue

        following = rows[index + 1]
        before = capped(row["kind"], row["value"])
        after = capped(following["kind"], following["value"])
        row["loss"] = 0 if row["played"] == row["best"] else max(0, loss_of(before, after))

    return {"id": game["id"], "result": lichess.result_of(game), "rows": rows}


_worker: dict = {}


def _start(binary: str, options: dict, config: str | None, settings: dict) -> None:
    uci = UciEngine(Path(binary)).start()
    runner.apply_options(uci, options, config)
    _worker["uci"] = uci
    _worker["settings"] = settings


def _analyse(game: dict) -> dict | None:
    settings = _worker["settings"]

    if settings["deadline"] is not None and time.time() > settings["deadline"]:
        return None

    return analyse_game(_worker["uci"], game, settings["user"], settings["depth"])


def lines_of(analysis: dict, number: int) -> str:
    """
    The training lines of one game, `FEN<TAB>result<TAB>game<TAB>score`, for its quiet positions.
    """
    result = {1.0: "1", 0.5: "0.5", 0.0: "0"}[analysis["result"]]
    return "".join(
        f"{row['fen']}\t{result}\t{number}\t{row['white']}\n"
        for row in analysis["rows"]
        if not row["terminal"] and row["quiet"]
    )


def run(
    config: Config,
    store: Store,
    engine: BuiltEngine,
    games: list[dict],
    user: str,
    out_dir: Path,
    *,
    depth: int = 12,
    mistake_cp: int = MISTAKE_CP,
    engine_config: str | None = None,
    eval_file: Path | None = None,
    workers: int | None = None,
    deadline: float | None = None,
    allow_dirty: bool = False,
    framework=...,
    progress=lambda text: print(text, file=sys.stderr, flush=True),
) -> tuple[str, dict]:
    """
    Analyses `games` and writes `positions.txt`, `mistakes.txt`, `mistakes.json` and
    `experience.bin` into `out_dir`. A game that has not been started by `deadline`, a time from
    `time.time()`, is left out, and the summary lists the ids of the games that were analysed.
    """
    context = runner.begin(config, store, allow_dirty=allow_dirty, framework=framework)
    context.checks += runner.engine_checks(engine)
    options: dict = {"Hash": 16, "Threads": 1}
    limits = {"depth": depth, "mistake_cp": mistake_cp, "user": user, "options": options}

    if eval_file is not None:
        options["EvalFile"] = str(Path(eval_file).resolve())
        limits["eval_file_sha256"] = hashlib.sha256(Path(eval_file).read_bytes()).hexdigest()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    context.sample("before")
    settings = {"user": user, "depth": depth, "deadline": deadline}
    workers = workers or config.workers
    done: list[dict] = []

    if games:
        with Pool(
            min(workers, len(games)),
            initializer=_start,
            initargs=(str(engine.binary), options, engine_config, settings),
        ) as pool:
            for finished in pool.imap(_analyse, games):
                if finished is not None:
                    done.append(finished)

                    if len(done) % 10 == 0:
                        progress(f"{len(done)} of {len(games)} games analysed")

    context.sample("end")
    positions = mistakes = ""
    mistake_rows = []
    entries: dict[int, tuple[int, int, int]] = {}
    losses = []

    for number, analysis in enumerate(done, 1):
        positions += lines_of(analysis, number)
        result = {1.0: "1", 0.5: "0.5", 0.0: "0"}[analysis["result"]]

        for ply, row in enumerate(analysis["rows"]):
            if row["terminal"] or not row["mine"]:
                continue

            entries[row["key"]] = (
                row["best_data"],
                experience.engine_score(row["kind"], row["value"]),
                row["depth"],
            )

            if row["loss"] is not None:
                losses.append(row["loss"])

            if row["loss"] is not None and row["loss"] >= mistake_cp and row["quiet"]:
                mistakes += f"{row['fen']}\t{result}\t{number}\t{row['white']}\n"

            if row["loss"] is not None and row["loss"] >= mistake_cp:
                mistake_rows.append(
                    {
                        "game": analysis["id"],
                        "ply": ply,
                        "fen": row["fen"],
                        "played": row["played"],
                        "best": row["best"],
                        "loss_cp": row["loss"],
                    }
                )

    (out_dir / "positions.txt").write_text(positions)
    (out_dir / "mistakes.txt").write_text(mistakes)
    (out_dir / "mistakes.json").write_text(json.dumps(mistake_rows, indent=1))
    experience.write(out_dir / "experience.bin", entries)
    summary = {
        "games": len(games),
        "analysed": len(done),
        "analysed_ids": [a["id"] for a in done],
        "positions": positions.count("\n"),
        "mistakes": len(mistake_rows),
        "experience": len(entries),
        "mean_loss_cp": sum(losses) / len(losses) if losses else 0.0,
        "output": {
            "positions_sha256": hashlib.sha256(positions.encode()).hexdigest(),
            "directory": str(out_dir),
        },
    }
    require([preflight.Check("games", len(done) > 0 or not games, f"{len(done)} analysed")])
    run_id = runner.finish(
        context,
        protocol=PROTOCOL,
        version=VERSION,
        engine=engine,
        config=engine_config,
        limits=limits,
        dataset_id=None,
        summary=summary,
        items_blob=None,
        seed=None,
    )
    return run_id, summary
