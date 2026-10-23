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
The worker of the train protocol. It reads a data file, trains a network, and writes the network
and a JSON file with the numbers of the training.

It runs in the Python that has PyTorch, which the setting `trainer_python` names, and it needs
only numpy and this package, so that the trainer and meristats can use different versions of
Python.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from meristats import nnue, training


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="meristats.train_worker")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--lam", type=float, required=True)
    parser.add_argument("--max-score", type=int, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--init", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    start = nnue.read_network(args.init) if args.init else None
    stamp = time.monotonic()
    positions = training.read_dataset(
        args.data,
        lam=args.lam,
        max_score=args.max_score,
        limit=args.limit,
        workers=args.workers,
    )
    loading = time.monotonic() - stamp
    log: list[str] = []

    def say(text: str) -> None:
        log.append(text)
        print(text, file=sys.stderr, flush=True)

    say(f"{len(positions)} positions read in {loading:.1f}s")
    result = training.train(
        positions,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        seed=args.seed,
        init=start,
        device=args.device,
        log=say,
        seconds_loading=loading,
    )
    import torch

    nnue.write_network(result.network, args.out)
    args.result.write_text(
        json.dumps(
            {
                "positions": {
                    "read": positions.read,
                    "kept": len(positions),
                    "skipped": positions.skipped,
                    "held_out": int(positions.held.sum()),
                },
                "best_held": result.best_held,
                "epochs": result.epochs,
                "seconds_loading": result.seconds_loading,
                "seconds_training": result.seconds_training,
                "quantized_difference": result.quantized_difference,
                "quantized_magnitude": result.quantized_magnitude,
                "checked": result.checked,
                "device": result.device,
                "hidden": result.network.hidden,
                "torch": torch.__version__,
                "log": log,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
