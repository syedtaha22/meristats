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
import multiprocessing
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import chess
import pyarrow as pa
import pyarrow.parquet as pq

from meristats import datasets, preflight, runner, stats
from meristats.build import BuiltEngine
from meristats.config import Config
from meristats.datasets import Puzzle
from meristats.preflight import Check, require
from meristats.store import Store, utc_now
from meristats.uci import EngineError, UciEngine

PROTOCOL = "puzzles"
VERSION = 2
DETERMINISM_ITEMS = 1000
DETERMINISM_SEED = 12345
CHUNK = 200
THEME_MINIMUM = 200


@dataclass(frozen=True)
class Limit:
    """
    How a search of one move is limited: by `nodes`, by `depth` or by `movetime` in milliseconds.
    """

    kind: str
    value: int

    def __post_init__(self):
        if self.kind not in ("nodes", "depth", "movetime") or self.value < 1:
            raise ValueError(f"not a limit: {self.kind} {self.value}")

    def command(self) -> str:
        return f"{self.kind} {self.value}"

    @property
    def timed(self) -> bool:
        return self.kind == "movetime"


@dataclass(frozen=True)
class Item:
    """
    The result of one puzzle. Bit i of `ok_mask` is set when the i-th move of the solver was right.
    `first_fail` is 0 when every move was right, and the number of the first wrong move otherwise.
    """

    item: int
    solver_moves: int
    ok_mask: int
    first_fail: int
    nodes: int
    wall_ms: float
    depth_kept: tuple[int, ...]
    error: str = ""

    @property
    def solved(self) -> bool:
        return self.solver_moves > 0 and self.ok_mask == (1 << self.solver_moves) - 1

    def outcome(self) -> tuple:
        """
        What must repeat exactly in a search that is limited by nodes or depth.
        """
        return (self.ok_mask, self.first_fail, self.nodes, self.depth_kept)


def depth_kept(infos, expected: str) -> int:
    """
    The depth from which every iteration of the search, up to the last, had `expected` as the
    first move of its line, or 0 when the last iteration did not.
    """
    kept = 0

    for info in reversed(infos):
        if info.pv and info.pv[0] == expected:
            kept = info.depth or 0
        else:
            break

    return kept


def gives_checkmate(board: chess.Board, move: str) -> bool:
    """
    Whether `move` is legal on `board` and checkmates.
    """
    try:
        after = board.copy()
        after.push_uci(move)
    except ValueError:
        return False

    return after.is_checkmate()


def solve(uci: UciEngine, puzzle: Puzzle, limit: Limit, item: int) -> Item:
    """
    Plays the solution of a puzzle. Each move of the solver is searched from empty tables (a
    `ucinewgame`), the move of the solution and the reply of the opponent are played, and the line
    goes on until the engine plays a wrong move. A move is right when it is the move of the
    solution or when it gives checkmate.
    """
    try:
        board = chess.Board(puzzle.fen)
    except ValueError as error:
        return Item(item, puzzle.solver_moves, 0, 1, 0, 0.0, (), f"bad FEN: {error}")

    ok_mask, nodes, wall, kept, first_fail = 0, 0, 0.0, [], 0
    played: list[str] = []

    for step in range(puzzle.solver_moves):
        expected = puzzle.moves[2 * step]
        uci.new_game()
        uci.position(puzzle.fen, played)
        result = uci.go(limit.command())
        wall += result.elapsed * 1000.0
        nodes += (result.infos[-1].nodes or 0) if result.infos else 0
        kept.append(depth_kept(result.infos, expected))

        if result.bestmove == expected or gives_checkmate(board, result.bestmove):
            ok_mask |= 1 << step
        else:
            first_fail = step + 1
            break

        for move in puzzle.moves[2 * step : 2 * step + 2]:
            try:
                board.push_uci(move)
            except ValueError:
                return Item(
                    item,
                    puzzle.solver_moves,
                    ok_mask,
                    step + 1,
                    nodes,
                    wall,
                    tuple(kept),
                    f"illegal line move {move}",
                )

            played.append(move)

    return Item(item, puzzle.solver_moves, ok_mask, first_fail, nodes, wall, tuple(kept))


# ------------------------------------------------------------------ workers

_STATE: dict = {}


def _start_engine(binary: str, options: dict, config: str | None) -> tuple[UciEngine, list[str]]:
    uci = UciEngine(Path(binary)).start()
    replies = runner.apply_options(uci, options, config)
    return uci, replies


def _init_worker(binary, options, config, puzzles, limit):
    _STATE.update(
        binary=binary,
        options=options,
        config=config,
        puzzles=puzzles,
        limit=limit,
        uci=None,
        errors=0,
    )


def _work(span: tuple[int, int]) -> list[Item]:
    start, end = span
    results = []

    for index in range(start, end):
        puzzle = _STATE["puzzles"][index]

        try:
            if _STATE["uci"] is None:
                _STATE["uci"], _ = _start_engine(
                    _STATE["binary"], _STATE["options"], _STATE["config"]
                )

            results.append(solve(_STATE["uci"], puzzle, _STATE["limit"], index))
        except EngineError as error:
            if _STATE["uci"] is not None:
                _STATE["uci"].quit()

            _STATE["uci"] = None
            results.append(
                Item(
                    index, puzzle.solver_moves, 0, 1, 0, 0.0, (), f"{type(error).__name__}: {error}"
                )
            )

    return results


# ---------------------------------------------------------------- the run


def engine_options(limit: Limit, hash_mb: int, move_overhead: bool = True) -> dict:
    """
    The options that every engine of this protocol gets: one thread, a fixed table, and for a
    timed limit no move overhead, so that `movetime` is the time that the search takes.
    """
    options: dict[str, str | int] = {"Threads": 1, "Hash": hash_mb}

    if limit.timed and move_overhead:
        options["MoveOverhead"] = 0

    return options


def determinism_check(
    binary: Path, options: dict, config: str | None, puzzles: list[Puzzle], limit: Limit
) -> Check:
    """
    Plays a fixed sample of the puzzles twice, each time with its own engine process, and checks
    that the outcomes (the moves, the nodes and the depths) are equal.
    """
    count = min(DETERMINISM_ITEMS, len(puzzles))
    picks = sorted(random.Random(DETERMINISM_SEED).sample(range(len(puzzles)), count))
    outcomes = []

    for _ in range(2):
        try:
            uci, _ = _start_engine(str(binary), options, config)
        except EngineError as error:
            return Check("determinism sample", False, f"engine error: {error}")

        try:
            outcomes.append([solve(uci, puzzles[i], limit, i).outcome() for i in picks])
        except EngineError as error:
            return Check("determinism sample", False, f"engine error: {error}")
        finally:
            uci.quit()

    differ = sum(1 for a, b in zip(*outcomes) if a != b)
    return Check("determinism sample", differ == 0, f"{count} items, {differ} differ")


def items_table(results: list[Item]) -> bytes:
    """
    The per-item results as a Parquet file, without compression, because the store compresses it.
    """
    table = pa.table(
        {
            "item": pa.array([r.item for r in results], pa.uint32()),
            "solver_moves": pa.array([r.solver_moves for r in results], pa.uint8()),
            "ok_mask": pa.array([r.ok_mask for r in results], pa.uint16()),
            "first_fail": pa.array([r.first_fail for r in results], pa.uint8()),
            "nodes": pa.array([r.nodes for r in results], pa.uint64()),
            "wall_ms": pa.array([r.wall_ms for r in results], pa.float32()),
            "depth_kept": pa.array([list(r.depth_kept) for r in results], pa.list_(pa.uint8())),
            "error": pa.array([r.error for r in results], pa.string()),
        }
    )
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="NONE")
    return buffer.getvalue()


def read_items(data: bytes) -> list[Item]:
    """
    Reads the per-item results of a stored run.
    """
    table = pq.read_table(io.BytesIO(data)).to_pydict()
    return [
        Item(
            table["item"][i],
            table["solver_moves"][i],
            table["ok_mask"][i],
            table["first_fail"][i],
            table["nodes"][i],
            table["wall_ms"][i],
            tuple(table["depth_kept"][i]),
            table["error"][i],
        )
        for i in range(len(table["item"]))
    ]


def natural_key(value) -> list:
    """
    A sort key that puts numbers in numeric order: `2` before `10`, and `200-399` before
    `1000-1199`.
    """
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(value))]


def summarise(puzzles: list[Puzzle], results: list[Item]) -> dict:
    """
    The weighted results of a run, with 95% intervals: the whole sample, and by band of rating,
    number of moves of the solver, kind of puzzle and theme.
    """
    weights = [puzzles[r.item].weight for r in results]
    solved = [r.solved for r in results]
    first = [bool(r.ok_mask & 1) for r in results]
    ratings = [puzzles[r.item].rating for r in results]

    def group(keys, label):
        buckets: dict = {}

        for r, key in zip(results, keys):
            for one in key if isinstance(key, list) else [key]:
                buckets.setdefault(one, []).append(r)

        table = {}

        for key, members in sorted(buckets.items(), key=lambda pair: natural_key(pair[0])):
            if label == "theme" and len(members) < THEME_MINIMUM:
                continue

            table[str(key)] = share(members)

        return table

    def share(members):
        member_weights = [puzzles[r.item].weight for r in members]
        result = stats.weighted_share([r.solved for r in members], member_weights)
        result["first_move"] = stats.weighted_share(
            [bool(r.ok_mask & 1) for r in members], member_weights
        )["p"]
        result["mean_nodes"] = sum(w * r.nodes for w, r in zip(member_weights, members)) / sum(
            member_weights
        )
        return result

    kept = [
        (w, d)
        for w, r in zip(weights, results)
        for i, d in enumerate(r.depth_kept)
        if r.ok_mask >> i & 1 and d > 0
    ]
    whole = share(results)
    whole["mean_depth_kept"] = (
        sum(w * d for w, d in kept) / sum(w for w, _ in kept) if kept else None
    )
    whole["puzzle_rating"] = stats.puzzle_rating(ratings, weights, solved)
    whole["puzzle_rating_first_move"] = stats.puzzle_rating(ratings, weights, first)

    errors = [r for r in results if r.error]
    return {
        "items": len(results),
        "whole": whole,
        "by_rating_band": group(
            [f"{r // 200 * 200}-{r // 200 * 200 + 199}" for r in ratings], "band"
        ),
        "by_solver_moves": group([r.solver_moves for r in results], "moves"),
        "by_kind": group(
            [
                f"mate in {puzzles[r.item].mate}" if puzzles[r.item].mate else "no mate"
                for r in results
            ],
            "kind",
        ),
        "by_theme": group([list(puzzles[r.item].themes) for r in results], "theme"),
        "errors": {"count": len(errors), "first": [f"{r.item}: {r.error}" for r in errors[:10]]},
    }


def run(
    config: Config,
    store: Store,
    engine: BuiltEngine,
    dataset_path: Path,
    limit: Limit,
    *,
    engine_config: str | None = None,
    eval_file: Path | None = None,
    items: int = 0,
    workers: int | None = None,
    hash_mb: int = 16,
    allow_dirty: bool = False,
    framework=...,
    progress=lambda text: print(text, file=sys.stderr, flush=True),
) -> tuple[str, dict]:
    """
    Plays the solutions of the puzzles of a sample file with one engine and one configuration, and
    stores the run. Returns the run id and the summary. An `eval_file` is a network that the engine
    loads in place of its own, and the run records its hash.

    The first `items` puzzles are played, or all of them when `items` is 0. The engine gets one
    thread and a fixed hash, and a timed limit gets `MoveOverhead 0`. A search limited by nodes or
    depth is checked for determinism on a fixed sample before the run.
    """
    workers = workers or config.workers
    context = runner.begin(config, store, allow_dirty=allow_dirty, framework=framework)
    puzzles = datasets.load_puzzles(dataset_path, items)

    if not puzzles:
        raise datasets.DatasetError(f"no puzzles in {dataset_path}")

    dataset_id, digest = datasets.register_file(
        store,
        datasets.SAMPLE_NAME,
        "1",
        dataset_path,
        {"file": dataset_path.name, "rows_played": len(puzzles)},
    )
    options = engine_options(limit, hash_mb)

    if eval_file is not None:
        options["EvalFile"] = str(Path(eval_file).resolve())

    probe, replies = _start_engine(str(engine.binary), options, engine_config)
    probe.quit()
    checks = runner.engine_checks(engine)
    checks += [
        preflight.engine_replies(replies),
        preflight.one_thread(options),
        preflight.move_overhead(options, limit.timed)
        if "MoveOverhead" in options
        else Check("move overhead", True, "not needed"),
    ]
    checks.append(preflight.dataset_hash(dataset_path, digest))

    if not limit.timed:
        checks.append(determinism_check(engine.binary, options, engine_config, puzzles, limit))

    context.checks += require(checks)
    context.sample("before")
    spans = [(start, min(start + CHUNK, len(puzzles))) for start in range(0, len(puzzles), CHUNK)]
    ctx = multiprocessing.get_context("fork")
    results: list[Item] = []

    with ctx.Pool(
        workers, _init_worker, (str(engine.binary), options, engine_config, puzzles, limit)
    ) as pool:
        done, step = 0, max(1, len(puzzles) // 20)

        for batch in pool.imap(_work, spans):
            results += batch
            done += len(batch)

            if done // step != (done - len(batch)) // step:
                progress(f"{done} of {len(puzzles)} puzzles")

    results.sort(key=lambda r: r.item)
    context.sample("end")
    blob = store.put_blob(items_table(results), "items", f"{PROTOCOL}-items.parquet")
    summary = summarise(puzzles, results)
    limits = {
        "limit": {limit.kind: limit.value},
        "options": options,
        "items": len(puzzles),
        "workers": workers,
    }

    if eval_file is not None:
        limits["eval_file_sha256"] = hashlib.sha256(Path(eval_file).read_bytes()).hexdigest()
    run_id = store.add_run(
        protocol=PROTOCOL,
        protocol_version=VERSION,
        grade=context.grade,
        status="complete",
        started_utc=context.started_utc,
        finished_utc=utc_now(),
        engine_id=engine.engine_id,
        config=engine_config,
        limits=limits,
        dataset_id=dataset_id,
        seed=None,
        machine_id=context.machine_id,
        conditions=context.conditions(),
        code_commit=context.code_commit,
        code_dirty=context.code_dirty,
        summary=summary,
        items_blob=blob.sha256,
        blobs=((blob.sha256, "items"),),
    )
    return run_id, summary
