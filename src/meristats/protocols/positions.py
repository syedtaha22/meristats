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
The positions protocol: the engine plays itself over UCI and the quiet positions of its games are
written with their results and scores, as data for training a network.

Every game starts from random legal moves. After the opening, every `skip` plies a quiet position
is kept: the side to move is not in check, the best move is no capture and no promotion, and the
score is not a mate. A game ends at a mate, a draw by the rules, or when one side has been far
ahead for several moves in a row. Each line is `FEN<TAB>result<TAB>game<TAB>score`, where the
result is 1, 0.5 or 0 for White and the score is in centipawns from the side of White.
"""

import hashlib
import random
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import chess

from meristats import preflight, runner
from meristats.build import BuiltEngine
from meristats.config import Config
from meristats.preflight import require
from meristats.store import Store
from meristats.uci import UciEngine

PROTOCOL = "positions"
VERSION = 1
MATE_BOUND = 29000
MAX_PLIES = 400
AHEAD_CP = 1000
AHEAD_MOVES = 4


class PositionsError(RuntimeError):
    pass


def white_score(score: tuple[str, int], side: bool) -> tuple[bool, int]:
    """
    The score of a search from the side of White, and whether it is a plain centipawn score. A mate
    counts as a score far beyond the margin of a won game.
    """
    kind, value = score
    plain = kind == "cp" and abs(value) < MATE_BOUND
    magnitude = value if kind == "cp" else (32000 if value > 0 else -32000)
    return plain, magnitude if side == chess.WHITE else -magnitude


def play_game(
    uci: UciEngine,
    game: int,
    *,
    seed: int,
    depth: int,
    random_plies: int = 8,
    skip: int = 3,
) -> tuple[float, list[tuple[str, int]]]:
    """
    Plays one game and returns the result for White with the positions that it kept, as pairs of a
    FEN and the score from the side of White. The game is a function of its number and the seed.
    """
    rng = random.Random(seed * 1000003 + game)
    board = chess.Board()
    moves: list[str] = []
    seen: Counter = Counter()

    for _ in range(random_plies):
        legal = list(board.legal_moves)

        if not legal:
            return 0.5, []

        seen[board._transposition_key()] += 1
        move = rng.choice(legal)
        moves.append(move.uci())
        board.push(move)

    uci.new_game()
    kept: list[tuple[str, int]] = []
    ahead_white = ahead_black = 0

    for ply in range(MAX_PLIES):
        if not any(board.legal_moves):
            if not board.is_check():
                return 0.5, kept

            return (0.0 if board.turn == chess.WHITE else 1.0), kept

        if board.halfmove_clock >= 100 or board.is_insufficient_material():
            return 0.5, kept

        if seen[board._transposition_key()] >= 2:
            return 0.5, kept

        uci.position(None, moves)
        answer = uci.go(f"depth {depth}")
        scored = [info for info in answer.infos if info.score_kind is not None]

        if not scored:
            raise PositionsError(f"no score from the search of game {game}, ply {ply}")

        plain, score = white_score((scored[-1].score_kind, scored[-1].score), board.turn)
        ahead_white = ahead_white + 1 if score > AHEAD_CP else 0
        ahead_black = ahead_black + 1 if score < -AHEAD_CP else 0

        if ahead_white >= AHEAD_MOVES:
            return 1.0, kept

        if ahead_black >= AHEAD_MOVES:
            return 0.0, kept

        try:
            move = chess.Move.from_uci(answer.bestmove)
        except ValueError as error:
            raise PositionsError(f"bad move {answer.bestmove!r} in game {game}") from error

        if move not in board.legal_moves:
            raise PositionsError(f"illegal move {answer.bestmove} in game {game}")

        quiet = (
            not board.is_check() and not board.is_capture(move) and move.promotion is None and plain
        )

        if quiet and ply % skip == 0:
            kept.append((board.fen(en_passant="fen"), score))

        seen[board._transposition_key()] += 1
        moves.append(answer.bestmove)
        board.push(move)

    return 0.5, kept


_worker: dict = {}


def _start(binary: str, options: dict, config: str | None, settings: dict) -> None:
    uci = UciEngine(Path(binary)).start()
    runner.apply_options(uci, options, config)
    _worker["uci"] = uci
    _worker["settings"] = settings


def _play(game: int) -> tuple[int, float, list[tuple[str, int]]]:
    result, kept = play_game(_worker["uci"], game, **_worker["settings"])
    return game, result, kept


def format_lines(game: int, result: float, kept: list[tuple[str, int]]) -> str:
    """
    The lines of one game: `FEN<TAB>result<TAB>game<TAB>score`.
    """
    text = {1.0: "1", 0.5: "0.5", 0.0: "0"}[result]
    return "".join(f"{fen}\t{text}\t{game}\t{score}\n" for fen, score in kept)


def run(
    config: Config,
    store: Store,
    engine: BuiltEngine,
    out: Path,
    *,
    games: int = 20000,
    depth: int = 7,
    engine_config: str | None = None,
    eval_file: Path | None = None,
    random_plies: int = 8,
    skip: int = 3,
    seed: int = 1,
    workers: int | None = None,
    allow_dirty: bool = False,
    framework=...,
    progress=lambda text: print(text, file=sys.stderr, flush=True),
) -> tuple[str, dict]:
    """
    Plays `games` games of the engine against itself, one thread each, and writes the positions to
    `out`, in the order of the games. The run stores the settings, the counts and the hash of the
    file. The file itself stays where it is, because it is the size of a data set.
    """
    context = runner.begin(config, store, allow_dirty=allow_dirty, framework=framework)
    context.checks += runner.engine_checks(engine)
    options: dict = {"Hash": 16, "Threads": 1}
    limits = {
        "games": games,
        "depth": depth,
        "random_plies": random_plies,
        "skip": skip,
        "options": options,
    }

    if eval_file is not None:
        options["EvalFile"] = str(Path(eval_file).resolve())
        limits["eval_file_sha256"] = hashlib.sha256(Path(eval_file).read_bytes()).hexdigest()

    context.sample("before")
    settings = {"seed": seed, "depth": depth, "random_plies": random_plies, "skip": skip}
    workers = workers or config.workers
    results = {1.0: 0, 0.5: 0, 0.0: 0}
    positions = 0
    digest = hashlib.sha256()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with (
        open(out, "wb") as handle,
        Pool(
            min(workers, games),
            initializer=_start,
            initargs=(str(engine.binary), options, engine_config, settings),
        ) as pool,
    ):
        for done, (game, result, kept) in enumerate(pool.imap(_play, range(games)), 1):
            data = format_lines(game, result, kept).encode()
            handle.write(data)
            digest.update(data)
            results[result] += 1
            positions += len(kept)

            if done % 500 == 0:
                progress(f"{done} games, {positions} positions")

    context.sample("end")
    summary = {
        "games": games,
        "positions": positions,
        "white_wins": results[1.0],
        "draws": results[0.5],
        "black_wins": results[0.0],
        "output": {"path": str(out), "sha256": digest.hexdigest(), "bytes": out.stat().st_size},
    }
    require([preflight.Check("output", positions > 0, f"{positions} positions")])
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
        seed=seed,
    )
    return run_id, summary
