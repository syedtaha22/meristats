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
The train protocol: a network of the engine is trained from a data file of labelled positions, and
the run records the settings, the hash of the data, the losses of every epoch, the times, and the
network itself. The engine of the run has to accept the network as its EvalFile.
"""

import hashlib
import sys
import time
from pathlib import Path

from meristats import nnue, runner, training
from meristats.build import BuiltEngine
from meristats.config import Config
from meristats.preflight import Check, require
from meristats.store import Store
from meristats.uci import UciEngine

PROTOCOL = "train"
VERSION = 1


def file_digest(path: Path) -> str:
    """
    The SHA-256 of a file, read in pieces, because a data file is the size of a data set.
    """
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)

    return digest.hexdigest()


def engine_accepts(engine: BuiltEngine, network: Path) -> Check:
    """
    The engine loads the network as its EvalFile and then searches with it.
    """
    try:
        with UciEngine(engine.binary).start() as uci:
            replies = uci.set_option("EvalFile", str(network))
            loaded = any("evaluation: network" in reply for reply in replies)
            uci.position(None, [])
            move = uci.go("depth 3").bestmove if loaded else ""
    except Exception as error:  # noqa: BLE001 - the check reports any failure of the engine
        return Check("engine loads the network", False, f"{type(error).__name__}: {error}")

    return Check(
        "engine loads the network",
        loaded and bool(move),
        "loaded and searched" if loaded and move else f"replies: {replies}",
    )


def run(
    config: Config,
    store: Store,
    engine: BuiltEngine,
    data: Path,
    *,
    out: Path | None = None,
    source: Path | None = None,
    epochs: int = 30,
    batch: int = 16384,
    lr: float = 0.001,
    lam: float = 0.7,
    max_score: int = 2500,
    limit: int = 0,
    seed: int = 1,
    init: Path | None = None,
    device: str | None = None,
    workers: int | None = None,
    allow_dirty: bool = False,
    framework=...,
    progress=lambda text: print(text, file=sys.stderr, flush=True),
) -> tuple[str, dict]:
    """
    Trains a network from `data` and stores the run. The network is written to `out` (by default
    in the cache folder of the run) and, with `source`, also as the C++ source of the engine's
    `src/nnue_net.inc`. Returns the run id and the summary.
    """
    context = runner.begin(config, store, allow_dirty=allow_dirty, framework=framework)
    context.checks += runner.engine_checks(engine)
    digest = file_digest(data)
    dataset_id = store.add_dataset(
        "training-positions",
        "1",
        digest,
        {"file": Path(data).name, "bytes": Path(data).stat().st_size},
    )
    start = nnue.read_network(init) if init is not None else None
    context.sample("before")
    stamp = time.monotonic()
    positions = training.read_dataset(
        data, lam=lam, max_score=max_score, limit=limit, workers=workers or config.workers
    )
    loading = time.monotonic() - stamp
    progress(f"{len(positions)} positions read in {loading:.1f}s")
    log: list[str] = []

    def say(text: str) -> None:
        log.append(text)
        progress(text)

    result = training.train(
        positions,
        epochs=epochs,
        batch=batch,
        lr=lr,
        seed=seed,
        init=start,
        device=device,
        log=say,
        seconds_loading=loading,
    )
    import torch

    context.sample("end")
    folder = (
        Path(out).parent
        if out
        else config.cache_dir / "train" / context.started_utc.replace(":", "")
    )
    folder.mkdir(parents=True, exist_ok=True)
    path = Path(out) if out else folder / "net.nnue"
    nnue.write_network(result.network, path)

    if source is not None:
        Path(source).write_text(nnue.source_text(result.network))

    network_bytes = path.read_bytes()
    context.checks += require([engine_accepts(engine, path)])
    network_blob = store.put_blob(network_bytes, "network", path.name)
    log_blob = store.put_blob("\n".join(log).encode(), "tool-log", "train.log")
    held = [e["held"] for e in result.epochs]
    summary = {
        "positions": {
            "read": positions.read,
            "kept": len(positions),
            "skipped": positions.skipped,
            "held_out": int(positions.held.sum()),
        },
        "best_held": result.best_held,
        "best_epoch": held.index(min(held)) + 1,
        "epochs": result.epochs,
        "seconds": {
            "loading": result.seconds_loading,
            "training": result.seconds_training,
            "per_epoch": result.seconds_training / max(1, len(result.epochs)),
        },
        "quantized": {
            "mean_difference_cp": result.quantized_difference,
            "mean_magnitude_cp": result.quantized_magnitude,
            "positions": result.checked,
        },
        "network": {
            "sha256": hashlib.sha256(network_bytes).hexdigest(),
            "bytes": len(network_bytes),
            "path": str(path),
        },
        "device": result.device,
        "torch": torch.__version__,
    }
    limits = {
        "epochs": epochs,
        "batch": batch,
        "lr": lr,
        "lambda": lam,
        "max_score": max_score,
        "limit": limit,
        "hidden": result.network.hidden,
        "data_sha256": digest,
        "init_sha256": None if init is None else file_digest(init),
    }
    run_id = runner.finish(
        context,
        protocol=PROTOCOL,
        version=VERSION,
        engine=engine,
        config=None,
        limits=limits,
        dataset_id=dataset_id,
        summary=summary,
        items_blob=None,
        blobs=((network_blob.sha256, "network"), (log_blob.sha256, "tool-log")),
        seed=seed,
    )
    return run_id, summary
