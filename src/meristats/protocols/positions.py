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

A game starts from an opening of a book or from random legal moves, and the engine sometimes plays
another near-best move in the first plies after it. Each line is
`FEN<TAB>result<TAB>game<TAB>score`, where the result is 1, 0.5 or 0 for White and the score is in
centipawns from the side of White.
"""

import functools
import hashlib
import multiprocessing
import random
import signal
import sys
import time
from collections import Counter
from pathlib import Path

import chess
import chess.pgn

from meristats import preflight, runner
from meristats.build import BuiltEngine
from meristats.config import Config
from meristats.preflight import require
from meristats.store import Store
from meristats.uci import UciEngine

PROTOCOL = "positions"
VERSION = 2
MATE_BOUND = 29000
MAX_PLIES = 400
AHEAD_CP = 1000
AHEAD_MOVES = 4
BOOK_PLIES = 16
BOOK_RANDOM_PLIES = 2
OPENING_DEPTH = 6
OPENING_CP = 300
OPENING_TRIES = 8
EXPLORE_TRIES = 3
EXPLORE_DEPTH_LESS = 3


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


@functools.lru_cache(maxsize=2)
def load_book(path: Path, plies: int = BOOK_PLIES) -> tuple[tuple[str, ...], ...]:
    """
    The openings of a PGN file, each as the UCI moves of its first `plies` plies. Openings that
    are the same within those plies are listed once.
    """
    openings: dict[tuple[str, ...], None] = {}

    with open(path, encoding="utf-8", errors="replace") as handle:
        while (game := chess.pgn.read_game(handle)) is not None:
            moves = tuple(move.uci() for _, move in zip(range(plies), game.mainline_moves()))

            if moves:
                openings[moves] = None

    return tuple(openings)


def draw_opening(
    uci: UciEngine,
    rng: random.Random,
    *,
    book: tuple[tuple[str, ...], ...],
    book_share: float,
    min_random_plies: int,
    random_plies: int,
) -> tuple[list[str], Counter] | None:
    """
    Draws the moves of an opening and the positions that it passed, or None when the opening ends
    the game. It is drawn again when a shallow search finds the position lopsided, and the last
    draw stands when every try is lopsided.
    """
    for attempt in range(OPENING_TRIES):
        board = chess.Board()
        moves: list[str] = []
        seen: Counter = Counter()

        if book and rng.random() < book_share:
            for text in rng.choice(book):
                seen[board._transposition_key()] += 1
                moves.append(text)
                board.push_uci(text)

            extra = rng.randint(0, BOOK_RANDOM_PLIES)
        else:
            extra = rng.randint(min(min_random_plies, random_plies), random_plies)

        for _ in range(extra):
            legal = list(board.legal_moves)

            if not legal:
                return None

            seen[board._transposition_key()] += 1
            move = rng.choice(legal)
            moves.append(move.uci())
            board.push(move)

        if not any(board.legal_moves) or attempt == OPENING_TRIES - 1:
            return moves, seen

        uci.new_game()
        uci.position(None, moves)
        scored = [i for i in uci.go(f"depth {OPENING_DEPTH}").infos if i.score_kind is not None]

        if scored:
            plain, score = white_score((scored[-1].score_kind, scored[-1].score), board.turn)

            if plain and abs(score) <= OPENING_CP:
                return moves, seen


def other_move(
    uci: UciEngine,
    rng: random.Random,
    board: chess.Board,
    moves: list[str],
    best: chess.Move,
    best_score: int,
    *,
    depth: int,
    margin: int,
) -> chess.Move:
    """
    A move other than `best` that a shallow search scores within `margin` centipawns of
    `best_score` (from the side to move), or `best` when none of a few random tries qualifies.
    """
    others = [move for move in board.legal_moves if move != best]
    rng.shuffle(others)

    for move in others[:EXPLORE_TRIES]:
        uci.position(None, moves)
        answer = uci.go(f"depth {max(depth - EXPLORE_DEPTH_LESS, 1)} searchmoves {move.uci()}")
        scored = [info for info in answer.infos if info.score_kind == "cp"]

        if not scored or abs(scored[-1].score) >= MATE_BOUND:
            continue

        if scored[-1].score >= best_score - margin:
            return move

    return best


def play_game(
    uci: UciEngine,
    game: int,
    *,
    seed: int,
    depth: int,
    random_plies: int = 8,
    min_random_plies: int = 4,
    book: tuple[tuple[str, ...], ...] = (),
    book_share: float = 0.5,
    explore: float = 0.1,
    explore_plies: int = 30,
    explore_margin: int = 60,
    skip: int = 3,
) -> tuple[float, list[tuple[str, int]]]:
    """
    Plays one game and returns the result for White with the positions that it kept, as pairs of a
    FEN and the score from the side of White. The game is a function of its number and the seed.
    In each of the first `explore_plies` plies after the opening, the engine plays another move
    with the probability `explore`, if one scores within `explore_margin` centipawns of its best.
    """
    rng = random.Random(seed * 1000003 + game)
    opening = draw_opening(
        uci,
        rng,
        book=book,
        book_share=book_share,
        min_random_plies=min_random_plies,
        random_plies=random_plies,
    )

    if opening is None:
        return 0.5, []

    moves, seen = opening
    board = chess.Board()

    for text in moves:
        board.push_uci(text)

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

        if quiet and rng.random() * skip < 1:
            kept.append((board.fen(en_passant="fen"), score))

        if plain and ply < explore_plies and rng.random() < explore:
            side = score if board.turn == chess.WHITE else -score
            move = other_move(
                uci, rng, board, moves, move, side, depth=depth, margin=explore_margin
            )

        seen[board._transposition_key()] += 1
        moves.append(move.uci())
        board.push(move)

    return 0.5, kept


_worker: dict = {}


def _start(binary: str, options: dict, config: str | None, settings: dict) -> None:
    # Ctrl-C belongs to the main process, which ends the run. A worker and its engine ignore it.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
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
    min_random_plies: int = 4,
    book: Path | None = None,
    book_share: float = 0.5,
    explore: float = 0.1,
    skip: int = 3,
    seed: int = 1,
    workers: int | None = None,
    max_seconds: float | None = None,
    allow_dirty: bool = False,
    framework=...,
    progress=lambda text: print(text, file=sys.stderr, flush=True),
) -> tuple[str, dict]:
    """
    Plays `games` games of the engine against itself, one thread each, and writes the positions to
    `out`, in the order of the games. With `max_seconds` the run stops when that time has passed,
    and the games that are not finished are left out, and so does Ctrl-C. The run stores the
    settings, the counts and the hash of the file. The file itself stays where it is, because it is
    the size of a data set.
    """
    context = runner.begin(config, store, allow_dirty=allow_dirty, framework=framework)
    context.checks += runner.engine_checks(engine)
    options: dict = {"Hash": 16, "Threads": 1}
    limits = {
        "games": games,
        "depth": depth,
        "random_plies": random_plies,
        "min_random_plies": min_random_plies,
        "book_share": book_share,
        "explore": explore,
        "skip": skip,
        "max_seconds": max_seconds,
        "options": options,
    }

    if eval_file is not None:
        options["EvalFile"] = str(Path(eval_file).resolve())
        limits["eval_file_sha256"] = hashlib.sha256(Path(eval_file).read_bytes()).hexdigest()

    openings: tuple[tuple[str, ...], ...] = ()

    if book is not None:
        openings = load_book(Path(book))
        limits["book_sha256"] = hashlib.sha256(Path(book).read_bytes()).hexdigest()
        limits["book_openings"] = len(openings)

    context.sample("before")
    settings = {
        "seed": seed,
        "depth": depth,
        "random_plies": random_plies,
        "min_random_plies": min_random_plies,
        "book": openings,
        "book_share": book_share,
        "explore": explore,
        "skip": skip,
    }

    workers = workers or config.workers
    results = {1.0: 0, 0.5: 0, 0.0: 0}
    positions = duplicates = 0
    interrupted = False
    completed = 0
    written: set[bytes] = set()
    started = time.monotonic()
    digest = hashlib.sha256()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with (
        open(out, "wb") as handle,
        multiprocessing.get_context("forkserver").Pool(
            min(workers, games),
            initializer=_start,
            initargs=(str(engine.binary), options, engine_config, settings),
        ) as pool,
    ):
        try:
            for done, (game, result, kept) in enumerate(pool.imap(_play, range(games)), 1):
                fresh = []

                for fen, score in kept:
                    key = hashlib.blake2b(fen.rsplit(" ", 2)[0].encode(), digest_size=8).digest()

                    if key in written:
                        duplicates += 1
                    else:
                        written.add(key)
                        fresh.append((fen, score))

                kept = fresh
                data = format_lines(game, result, kept).encode()
                handle.write(data)
                digest.update(data)
                results[result] += 1
                positions += len(kept)
                completed = done

                if max_seconds is not None and time.monotonic() - started >= max_seconds:
                    pool.terminate()
                    break

                if done % 500 == 0:
                    progress(f"{done} games, {positions} positions")
        except KeyboardInterrupt:
            pool.terminate()
            interrupted = True

    context.sample("end")
    summary = {
        "games": completed,
        "positions": positions,
        "duplicates": duplicates,
        "interrupted": interrupted,
        "white_wins": results[1.0],
        "draws": results[0.5],
        "black_wins": results[0.0],
        "output": {"path": str(out), "sha256": digest.hexdigest(), "bytes": out.stat().st_size},
    }
    require([preflight.Check("output", positions > 0 or interrupted, f"{positions} positions")])
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
