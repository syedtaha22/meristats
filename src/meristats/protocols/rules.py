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
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from meristats import datasets, preflight, runner
from meristats.build import BuiltEngine
from meristats.config import Config
from meristats.preflight import require
from meristats.store import Store
from meristats.uci import UciEngine

PROTOCOL = "rules"
VERSION = 1
MAX_NODES = 200_000_000
REPETITIONS = 7


def run(
    config: Config,
    store: Store,
    engine: BuiltEngine,
    perft_path: Path,
    *,
    max_nodes: int = MAX_NODES,
    repetitions: int = REPETITIONS,
    cpu: int | None = None,
    allow_dirty: bool = False,
    framework=...,
    progress=lambda text: print(text, file=sys.stderr, flush=True),
) -> tuple[str, dict]:
    """
    Counts the nodes of every reference position at every depth whose published count is at most
    `max_nodes`, and compares them with the published counts. The deepest such depth of each
    position is then repeated `repetitions` times for the speed.

    A wrong count does not stop the run. It is recorded, and `all_correct` in the summary is false.
    """
    context = runner.begin(config, store, allow_dirty=allow_dirty, framework=framework)
    positions = datasets.read_perft(perft_path)
    dataset_id, digest = datasets.register_file(
        store, "perft", "1", perft_path, {"file": perft_path.name, "positions": len(positions)}
    )
    rows = []
    speeds = {}
    uci = UciEngine(engine.binary, cpu=cpu).start()

    try:
        context.checks += require(
            runner.engine_checks(engine) + [preflight.dataset_hash(perft_path, digest)]
        )
        context.sample("before")

        for position in positions:
            uci.position(position.fen)
            deepest = 0

            for depth, expected in enumerate(position.counts, 1):
                if expected > max_nodes:
                    break

                nodes, seconds = uci.perft(depth)
                rows.append(
                    {
                        "position": position.name,
                        "depth": depth,
                        "expected": expected,
                        "nodes": nodes,
                        "correct": nodes == expected,
                        "seconds": [seconds],
                    }
                )
                deepest = depth

            if deepest:
                rows[-1]["seconds"] += [uci.perft(deepest)[1] for _ in range(repetitions - 1)]
                speeds[position.name] = {
                    "depth": deepest,
                    "nodes": rows[-1]["expected"],
                    "median_seconds": statistics.median(rows[-1]["seconds"]),
                    "nodes_per_second": rows[-1]["expected"]
                    / statistics.median(rows[-1]["seconds"]),
                }

            progress(f"{position.name}: depths up to {deepest}")
    finally:
        uci.quit()

    context.sample("end")
    data = pa.table(
        {
            "position": pa.array([r["position"] for r in rows], pa.string()),
            "depth": pa.array([r["depth"] for r in rows], pa.uint8()),
            "expected_nodes": pa.array([r["expected"] for r in rows], pa.uint64()),
            "nodes": pa.array([r["nodes"] for r in rows], pa.uint64()),
            "correct": pa.array([r["correct"] for r in rows], pa.bool_()),
            "seconds": pa.array([r["seconds"] for r in rows], pa.list_(pa.float32())),
        }
    )
    buffer = io.BytesIO()
    pq.write_table(data, buffer, compression="NONE")
    blob = store.put_blob(buffer.getvalue(), "items", f"{PROTOCOL}-rows.parquet")
    wrong = [
        f"{r['position']} depth {r['depth']}: {r['nodes']} not {r['expected']}"
        for r in rows
        if not r["correct"]
    ]
    summary = {"all_correct": not wrong, "counts": len(rows), "wrong": wrong, "speed": speeds}
    run_id = runner.finish(
        context,
        protocol=PROTOCOL,
        version=VERSION,
        engine=engine,
        config=None,
        limits={"max_nodes": max_nodes, "repetitions": repetitions, "cpu": cpu},
        dataset_id=dataset_id,
        summary=summary,
        items_blob=blob.sha256,
        blobs=((blob.sha256, "items"),),
    )
    return run_id, summary
