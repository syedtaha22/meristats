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

"""
One cycle of learning for the engine, in three steps:

1. data: the engine plays itself, in chunks of a few minutes, until the positions that no test has
   seen reach `test_after`
2. train: a network is trained from the deployed one on those positions, a sample of the earlier
   ones, the anchor data and, when there is a file of them, positions of Leela Chess Zero
3. eval: the network plays the deployed one at a fixed number of nodes per move, in a sequential
   test, for as long as the test needs

The cycle ends with the status `passed` when the test accepts the network, and `rejected` when it
does not. A cycle does not change the engine repository: whoever deploys the network reads
`candidate` and `deployed` from the summary. The data of a cycle that was not tested, because the
program stopped during the first step, counts for the next one, since every chunk is stored when it
is finished.

The deployed engine is the head of the branch of the networks. `test_after` grows by `growth` after
a cycle that is rejected, since the data was not enough for a better network, and stays after a
cycle that passed. The numbers that outlive a cycle are in `train-state.json` in the data folder.
"""

import json
import os
import subprocess
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from meristats import nnue, replay
from meristats.build import build_engine
from meristats.config import Config
from meristats.protocols import match as match_protocol
from meristats.protocols import positions as positions_protocol
from meristats.protocols import train as train_protocol
from meristats.store import Store

STATE = "train-state.json"
STAMP = "%Y%m%dT%H%M%SZ"


class CycleError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    """
    The numbers of a cycle. `deadline` is a time from `time.time()` after which no more data is
    generated, or None for no end.
    """

    deadline: float | None = None
    test_after: int = 600_000
    growth: float = 1.5
    chunk_minutes: float = 10
    selfplay_depth: int = 9
    selfplay_skip: int = 3
    epochs: int = 8
    patience: int = 1
    lr: float = 0.0002
    anchor_lines: int = 1_000_000
    leela_lines: int = 1_000_000
    history_lines: int = 1_500_000
    match_nodes: int = 50000
    match_minutes: float | None = None
    rounds: int = 5000
    elo0: float = 0.0
    elo1: float = 10.0
    seed: int = 1
    book: Path | None = None
    allow_dirty: bool = False


@dataclass
class State:
    """
    What outlives a cycle: the newest chunk that a test has seen, the number of new positions that
    the next test waits for, and the counters that keep every chunk and cycle on its own seed.
    """

    last_test: str = ""
    test_after: int = 600_000
    cycles: int = 0
    chunks: int = 0

    @classmethod
    def load(cls, data_dir: Path, test_after: int) -> "State":
        path = Path(data_dir) / STATE

        if not path.exists():
            return cls(test_after=test_after)

        return cls(**json.loads(path.read_text()))

    def save(self, data_dir: Path) -> None:
        path = Path(data_dir) / STATE
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        partial.write_text(json.dumps(asdict(self), indent=1))
        os.replace(partial, path)


def stages() -> dict[str, Callable]:
    """
    The functions that do the work of the steps, so that a test can replace them.
    """
    return {
        "build": build_engine,
        "selfplay": positions_protocol.run,
        "train": train_protocol.run,
        "match": match_protocol.run,
    }


def committed_network(repo: Path, sha: str, destination: Path) -> Path:
    """
    The network that is built into the engine at a commit, as a network file.
    """
    done = subprocess.run(
        ["git", "-C", str(repo), "show", f"{sha}:src/nnue_net.inc"], capture_output=True, text=True
    )

    if done.returncode != 0:
        raise CycleError(f"{sha} has no src/nnue_net.inc")

    source = destination.with_suffix(".inc")
    source.write_text(done.stdout)
    nnue.write_network(nnue.read_source(source), destination)
    return destination


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime(STAMP)


def count_positions(buffer: replay.Buffer, after: str) -> int:
    """
    The positions in the chunks that are newer than `after`.
    """
    return sum(
        1
        for path in buffer.files(("selfplay",))
        if path.name.removesuffix(".txt.zst") > after
        for _ in replay.read_lines(path)
    )


def selfplay_chunk(
    do: dict[str, Callable],
    config: Config,
    store: Store,
    engine,
    settings: "Settings",
    state: "State",
    buffer: replay.Buffer,
    work: Path,
    seconds: float,
    book: Path,
) -> dict:
    """
    Plays self-play games for `seconds`, adds the positions to the replay buffer as one chunk, and
    returns the counts. The state keeps the number of chunks, so that every chunk has its own seed.
    A chunk that Ctrl-C ended keeps its finished games, and the interrupt is raised after that.
    """
    out = work / "chunk.txt"
    _, played = do["selfplay"](
        config,
        store,
        engine,
        out,
        games=1_000_000,
        depth=settings.selfplay_depth,
        skip=settings.selfplay_skip,
        book=settings.book or book,
        seed=settings.seed + state.chunks,
        max_seconds=seconds,
        allow_dirty=settings.allow_dirty,
    )
    buffer.add("selfplay", f"{utc_stamp()}-{state.chunks:05d}", out.read_text())
    out.unlink()

    state.chunks += 1
    state.save(config.data_dir)

    if played.get("interrupted"):
        raise KeyboardInterrupt

    return played


def generate(
    config: Config,
    store: Store,
    settings: "Settings",
    state: "State",
    *,
    book: Path,
    functions: dict[str, Callable] | None = None,
    progress: Callable[[str], None] = lambda text: print(text, flush=True),
    clock: Callable[[], float] = time.time,
) -> dict:
    """
    Plays self-play games with the engine of the networks branch, chunk after chunk, and adds the
    positions to the replay buffer. It runs until the deadline of the settings, or until Ctrl-C
    when there is none, and trains and tests nothing. Returns the counts.
    """
    do = functions or stages()
    work = Path(config.data_dir) / "generate"
    work.mkdir(parents=True, exist_ok=True)
    buffer = replay.Buffer(Path(config.data_dir) / "replay")

    engine = do["build"](config, store, config.networks_branch)
    progress(f"engine: {engine.version_text} ({engine.commit_sha[:9]})")

    started = clock()
    summary = {"games": 0, "positions": 0, "chunks": 0, "seconds": 0}

    try:
        while True:
            left = None if settings.deadline is None else settings.deadline - clock()

            if left is not None and left < 30:
                break

            seconds = settings.chunk_minutes * 60 if left is None else min(
                settings.chunk_minutes * 60, left)
            played = selfplay_chunk(
                do, config, store, engine, settings, state, buffer, work, seconds, book
            )
            summary["games"] += played["games"]
            summary["positions"] += played["positions"]
            summary["chunks"] += 1

            progress(
                f"data: {summary['positions']} positions from {summary['games']} games "
                f"({summary['chunks']} chunks, {round(clock() - started)} s)"
            )
    except KeyboardInterrupt:
        progress("stopped")

    summary["seconds"] = round(clock() - started)
    return summary


def run(
    config: Config,
    store: Store,
    settings: Settings,
    state: State,
    *,
    book: Path,
    functions: dict[str, Callable] | None = None,
    progress: Callable[[str], None] = lambda text: print(text, flush=True),
    clock: Callable[[], float] = time.time,
) -> dict:
    """
    Runs one cycle and returns its summary, which is also written to `summary.json` in the folder
    of the cycle. A step that fails ends the cycle with the status `failed` and its error, and the
    state keeps the data for the next try. The state is saved after every chunk and at the end.
    """
    do = functions or stages()
    cycle = utc_stamp()
    work = Path(config.data_dir) / "cycles" / cycle
    work.mkdir(parents=True, exist_ok=True)
    summary: dict = {"cycle": cycle, "status": "running", "test_after": state.test_after,
                     "stages": {}}
    started = clock()
    buffer = replay.Buffer(Path(config.data_dir) / "replay")

    def finish() -> dict:
        summary["seconds"] = round(clock() - started)
        (work / "summary.json").write_text(json.dumps(summary, indent=1))
        state.save(config.data_dir)
        return summary

    try:
        engine = do["build"](config, store, config.networks_branch)
        summary["engine"] = {"commit": engine.commit_sha, "version": engine.version_text}
        deployed = committed_network(
            Path(config.engine_repo), engine.commit_sha, work / "deployed.nnue"
        )

        have = count_positions(buffer, state.last_test)
        games = positions = chunks = 0
        began = clock()

        while have < state.test_after:
            left = None if settings.deadline is None else settings.deadline - clock()

            if left is not None and left < 60:
                break

            seconds = settings.chunk_minutes * 60 if left is None else min(
                settings.chunk_minutes * 60, left)
            played = selfplay_chunk(
                do, config, store, engine, settings, state, buffer, work, seconds, book
            )
            games += played["games"]
            positions += played["positions"]
            chunks += 1
            have += played["positions"]
            progress(f"data: {have} of {state.test_after} new positions ({chunks} chunks)")

        summary["stages"]["selfplay"] = {
            "games": games, "positions": positions, "chunks": chunks,
            "seconds": round(clock() - began),
        }

        if have < 1:
            summary["status"] = "failed"
            summary["error"] = "there are no new positions to train on"
            return finish()

        newest = max(
            path.name.removesuffix(".txt.zst") for path in buffer.files(("selfplay",))
        )
        counts = replay.assemble(
            work / "train.txt",
            buffer,
            state.last_test,
            anchor=config.anchor_positions,
            leela=config.leela_positions,
            seed=settings.seed + state.cycles,
            history_lines=settings.history_lines,
            anchor_lines=settings.anchor_lines,
            leela_lines=settings.leela_lines,
        )
        summary["stages"]["data"] = counts
        _, trained = do["train"](
            config,
            store,
            engine,
            work / "train.txt",
            out=work / "candidate.nnue",
            epochs=settings.epochs,
            patience=settings.patience,
            lr=settings.lr,
            init=deployed,
            seed=settings.seed + state.cycles,
            allow_dirty=settings.allow_dirty,
        )
        summary["stages"]["train"] = {
            "best_held": trained["best_held"],
            "start_held": trained.get("start_held"),
            "best_epoch": trained["best_epoch"],
            "epochs": len(trained["epochs"]),
            "seconds": trained["seconds"],
            "network": trained["network"],
        }
        progress(
            f"train: held-out loss {trained['best_held']:.5f} "
            f"(deployed {trained.get('start_held') or 0:.5f}) in {trained['best_epoch']} epochs"
        )
        timeout = None if settings.match_minutes is None else settings.match_minutes * 60
        run_id, played = do["match"](
            config,
            store,
            engine,
            engine,
            settings.book or book,
            match_protocol.Limit("nodes", settings.match_nodes),
            a_eval_file=work / "candidate.nnue",
            rounds=settings.rounds,
            sprt=match_protocol.Sprt(settings.elo0, settings.elo1),
            seed=settings.seed + state.cycles,
            timeout=timeout,
            allow_dirty=settings.allow_dirty,
        )
        test = played.get("sprt", {})
        passed = test.get("verdict") == "H1" and test.get("agree", False)
        summary["stages"]["match"] = {
            "run": run_id,
            "games": played["games"],
            "wins": played["wins"],
            "draws": played["draws"],
            "losses": played["losses"],
            "nodes": settings.match_nodes,
            "score": played.get("score"),
            "elo": played.get("elo"),
            "elo_low": played.get("elo_low"),
            "elo_high": played.get("elo_high"),
            "verdict": test.get("verdict"),
            "timed_out": played.get("timed_out", False),
        }
        progress(f"eval: {played['games']} games, verdict {test.get('verdict')}")
        summary["status"] = "passed" if passed else "rejected"

        if passed:
            summary["candidate"] = str(work / "candidate.nnue")
            summary["deployed"] = str(work / "deployed.nnue")

        state.last_test = newest
        state.cycles += 1

        if not passed:
            state.test_after = int(state.test_after * settings.growth)
    except Exception as error:  # noqa: BLE001 - the cycle reports every failure
        summary["status"] = "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        summary["trace"] = traceback.format_exc()[-1500:]

    return finish()


def format_summary(summary: dict) -> str:
    """
    The summary of a cycle as text.
    """
    lines = [f"Cycle {summary['cycle']}: {summary['status']}"]

    if "error" in summary:
        lines.append(f"Error: {summary['error']}")

    stages = summary.get("stages", {})

    if "selfplay" in stages:
        s = stages["selfplay"]
        lines.append(f"Data: {s['positions']} positions from {s['games']} games")

    if "train" in stages:
        s = stages["train"]
        lines.append(f"Training: held-out loss {s['best_held']:.5f} at epoch {s['best_epoch']}")

    if "match" in stages:
        s = stages["match"]
        elo = (
            ""
            if s["elo"] is None
            else f", Elo {s['elo']:+.1f} ({s['elo_low']:+.1f} to {s['elo_high']:+.1f})"
        )
        lines.append(f"Match: {s['games']} games{elo}, verdict {s['verdict']}")

    return "\n".join(lines)
