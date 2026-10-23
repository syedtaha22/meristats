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

import io
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from meristats import datasets, preflight, runner
from meristats.build import BuiltEngine
from meristats.config import Config
from meristats.preflight import require
from meristats.store import Store
from meristats.uci import UciEngine

PROTOCOL = "search-nodes"
VERSION = 1


@dataclass(frozen=True)
class Row:
    """
    One search of one position by one configuration to one depth. The time is the median over the
    rounds, and the other fields must be equal in every round. `score` is in centipawns, or the
    number of moves to mate when `is_mate` is true, negative when the engine is mated.
    """

    config: str
    depth: int
    position: str
    region: str
    best_move: str
    score: int
    is_mate: bool
    nodes: int
    wall_ms: float


class RoundsDisagree(Exception):
    """
    The same search gave different moves, scores or node counts in two rounds.
    """


def search(uci: UciEngine, fen: str, depth: int):
    """
    Searches a position from empty tables to a depth, and returns the best move, the score, the
    nodes of the last iteration and the time.
    """
    uci.new_game()
    uci.position(fen)
    result = uci.go(f"depth {depth}", timeout=600.0)
    last = result.infos[-1] if result.infos else None
    mate = last is not None and last.score_kind == "mate"
    return (
        result.bestmove,
        last.score if last and last.score is not None else 0,
        mate,
        last.nodes if last and last.nodes is not None else 0,
        result.elapsed * 1000.0,
    )


def table(rows: list[Row]) -> bytes:
    """
    The rows as a Parquet file, without compression, because the store compresses it.
    """
    data = pa.table(
        {
            "config": pa.array([r.config for r in rows], pa.string()),
            "depth": pa.array([r.depth for r in rows], pa.uint8()),
            "position": pa.array([r.position for r in rows], pa.string()),
            "region": pa.array([r.region for r in rows], pa.string()),
            "best_move": pa.array([r.best_move for r in rows], pa.string()),
            "score": pa.array([r.score for r in rows], pa.int32()),
            "is_mate": pa.array([r.is_mate for r in rows], pa.bool_()),
            "nodes": pa.array([r.nodes for r in rows], pa.uint64()),
            "wall_ms": pa.array([r.wall_ms for r in rows], pa.float32()),
        }
    )
    buffer = io.BytesIO()
    pq.write_table(data, buffer, compression="NONE")
    return buffer.getvalue()


def summarise(rows: list[Row]) -> dict:
    """
    For each configuration and depth: the mean nodes and mean time per move, and the speed in
    nodes per second over all positions.
    """
    summary: dict = {}

    for config in dict.fromkeys(r.config for r in rows):
        for depth in sorted({r.depth for r in rows if r.config == config}):
            chosen = [r for r in rows if r.config == config and r.depth == depth]
            total_ms = sum(r.wall_ms for r in chosen)
            summary.setdefault(config, {})[str(depth)] = {
                "positions": len(chosen),
                "mean_nodes": sum(r.nodes for r in chosen) / len(chosen),
                "mean_ms": total_ms / len(chosen),
                "nodes_per_second": sum(r.nodes for r in chosen) / total_ms * 1000
                if total_ms
                else None,
            }

    return summary


def run(
    config: Config,
    store: Store,
    engine: BuiltEngine,
    suite_path: Path,
    configs: list[str | None],
    depths: list[int],
    *,
    rounds: int = 3,
    cpu: int | None = None,
    hash_mb: int = 16,
    allow_dirty: bool = False,
    framework=...,
    progress=lambda text: print(text, file=sys.stderr, flush=True),
) -> tuple[str, dict]:
    """
    Searches every position of a suite to every depth with every configuration, in `rounds`
    rounds that interleave the configurations, and stores the run.

    The configurations are values of the engine option `Config`, or None for the default of the
    engine, which an engine without that option needs. Each configuration has its own engine
    process, started once. The moves, scores and node counts must be equal in every round, and
    the time is the median over the rounds.
    """
    if rounds < 1 or not depths or not configs:
        raise ValueError("rounds, depths and configs must not be empty")

    context = runner.begin(config, store, allow_dirty=allow_dirty, framework=framework)
    suite = datasets.read_suite(suite_path)
    dataset_id, digest = datasets.register_file(
        store, "suite", "1", suite_path, {"file": suite_path.name, "positions": len(suite)}
    )
    options = {"Threads": 1, "Hash": hash_mb}
    engines: dict = {}
    replies: list[str] = []

    try:
        for name in configs:
            uci = UciEngine(engine.binary, cpu=cpu).start()
            replies += runner.apply_options(uci, options, name)
            engines[name] = uci

        context.checks += require(
            runner.engine_checks(engine)
            + [
                preflight.engine_replies(replies),
                preflight.one_thread(options),
                preflight.dataset_hash(suite_path, digest),
            ]
        )
        context.sample("before")
        samples: dict = {}

        for number in range(rounds):
            for depth in depths:
                for name in configs:
                    for position in suite:
                        move, score, mate, nodes, ms = search(engines[name], position.fen, depth)
                        key = (name, depth, position.name)
                        outcome = (move, score, mate, nodes)

                        if key in samples and samples[key]["outcome"] != outcome:
                            raise RoundsDisagree(
                                f"{key} gave {samples[key]['outcome']} and then {outcome}"
                            )

                        samples.setdefault(key, {"outcome": outcome, "ms": []})["ms"].append(ms)

            progress(f"round {number + 1} of {rounds}")
    finally:
        for uci in engines.values():
            uci.quit()

    context.sample("end")
    rows = []

    for name in configs:
        for depth in depths:
            for position in suite:
                sample = samples[(name, depth, position.name)]
                move, score, mate, nodes = sample["outcome"]
                rows.append(
                    Row(
                        name or "default",
                        depth,
                        position.name,
                        position.region,
                        move,
                        score,
                        mate,
                        nodes,
                        statistics.median(sample["ms"]),
                    )
                )

    blob = store.put_blob(table(rows), "items", f"{PROTOCOL}-rows.parquet")
    summary = summarise(rows)
    run_id = runner.finish(
        context,
        protocol=PROTOCOL,
        version=VERSION,
        engine=engine,
        config=",".join(name or "default" for name in configs),
        limits={"depths": depths, "rounds": rounds, "options": options, "cpu": cpu},
        dataset_id=dataset_id,
        summary=summary,
        items_blob=blob.sha256,
        blobs=((blob.sha256, "items"),),
    )
    return run_id, summary
