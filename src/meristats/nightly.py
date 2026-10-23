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
The nightly cycle: the engine plays itself, the day's games of the account are analysed again,
a network is trained from all of it and from the days before, and the night ends with the status
`passed` when the network beats the deployed one in a sequential test. The night does not change
the engine repository: whoever deploys the network reads `candidate` and `deployed` from the
summary.

The stages, in order:

1. self-play of the deployed engine, for a number of minutes
2. the games of the account since the last night, searched again, deeper than they were played
3. the experience file of the engine, updated with the best moves of those searches
4. the replay buffer, and from it the training file
5. the training, starting from the deployed network
6. a match of the new network against the deployed one, at a fixed number of nodes per move

The deployed engine is the head of the branch of the networks. A night that is late for its
deadline ends with the status `late` and the stage that it did not reach.
"""

import json
import subprocess
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from meristats import experience, lichess, nnue, replay
from meristats.build import build_engine
from meristats.config import Config
from meristats.protocols import match as match_protocol
from meristats.protocols import positions as positions_protocol
from meristats.protocols import reanalyse as reanalyse_protocol
from meristats.protocols import train as train_protocol
from meristats.store import Store


class NightlyError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    """
    The numbers of a night. The times are in minutes, and `deadline` is a time from `time.time()`.
    """

    deadline: float
    night: str = ""
    selfplay_minutes: float = 60
    selfplay_depth: int = 9
    selfplay_skip: int = 3
    analysis_minutes: float = 30
    analysis_depth: int = 12
    epochs: int = 4
    lr: float = 0.0002
    anchor_lines: int = 1_000_000
    history_lines: int = 1_500_000
    mistake_repeat: int = 3
    match_nodes: int = 50000
    match_minutes: float = 120
    rounds: int = 5000
    elo0: float = 0.0
    elo1: float = 10.0
    seed: int = 1
    book: Path | None = None
    allow_dirty: bool = False


def stages() -> dict[str, Callable]:
    """
    The functions that do the work of the stages, so that a test can replace them.
    """
    return {
        "build": build_engine,
        "selfplay": positions_protocol.run,
        "sync": lichess.sync,
        "reanalyse": reanalyse_protocol.run,
        "train": train_protocol.run,
        "match": match_protocol.run,
    }


def _committed_network(repo: Path, sha: str, destination: Path) -> Path:
    """
    The network that is built into the engine at a commit, as a network file.
    """
    done = subprocess.run(
        ["git", "-C", str(repo), "show", f"{sha}:src/nnue_net.inc"], capture_output=True, text=True
    )

    if done.returncode != 0:
        raise NightlyError(f"{sha} has no src/nnue_net.inc")

    source = destination.with_suffix(".inc")
    source.write_text(done.stdout)
    nnue.write_network(nnue.read_source(source), destination)
    return destination


def run(
    config: Config,
    store: Store,
    settings: Settings,
    *,
    book: Path,
    functions: dict[str, Callable] | None = None,
    progress: Callable[[str], None] = lambda text: print(text, flush=True),
) -> dict:
    """
    Runs one night and returns its summary, which is also written to `summary.json` in the folder
    of the night. A stage that fails ends the night with the status `failed` and its error.
    """
    do = functions or stages()
    night = settings.night or datetime.now().strftime("%Y-%m-%d")
    work = Path(config.data_dir) / "nightly" / night
    work.mkdir(parents=True, exist_ok=True)
    summary: dict = {"night": night, "status": "running", "stages": {}}
    started = time.time()

    def remaining(reserve: float = 0.0) -> float:
        return settings.deadline - time.time() - reserve

    def late(stage: str) -> bool:
        if remaining() <= 60:
            summary["status"] = "late"
            summary["stopped_before"] = stage
            return True

        return False

    def finish() -> dict:
        summary["seconds"] = round(time.time() - started)
        (work / "summary.json").write_text(json.dumps(summary, indent=1))
        return summary

    try:
        engine = do["build"](config, store, config.networks_branch)
        summary["engine"] = {"commit": engine.commit_sha, "version": engine.version_text}
        deployed = _committed_network(
            Path(config.engine_repo), engine.commit_sha, work / "deployed.nnue"
        )
        seed = settings.seed

        if late("self-play"):
            return finish()

        seconds = min(settings.selfplay_minutes * 60, remaining(reserve=600))
        run_id, played = do["selfplay"](
            config,
            store,
            engine,
            work / "selfplay.txt",
            games=1_000_000,
            depth=settings.selfplay_depth,
            skip=settings.selfplay_skip,
            seed=seed,
            max_seconds=seconds,
            allow_dirty=settings.allow_dirty,
        )
        summary["stages"]["selfplay"] = {"run": run_id, **played}
        progress(f"self-play: {played['games']} games, {played['positions']} positions")
        buffer = replay.Buffer(Path(config.data_dir) / "replay")
        buffer.add("selfplay", night, (work / "selfplay.txt").read_text())
        mistakes_text = ""

        if config.lichess_user and not late("analysis"):
            problem = None

            try:
                fresh = do["sync"](config.data_dir, config.lichess_user)
            except lichess.LichessError as error:
                fresh, problem = [], str(error)

            known = lichess.analysed(config.data_dir, config.lichess_user)
            todo = [
                g
                for g in lichess.stored(lichess.games_file(config.data_dir, config.lichess_user))
                if g["id"] not in known
            ]
            summary["stages"]["sync"] = {"new": len(fresh), "waiting": len(todo)}

            if problem:
                summary["stages"]["sync"]["error"] = problem

            if todo:
                limit = min(settings.analysis_minutes * 60, remaining(reserve=600))
                run_id, analysed = do["reanalyse"](
                    config,
                    store,
                    engine,
                    todo,
                    config.lichess_user,
                    work / "reanalysis",
                    depth=settings.analysis_depth,
                    deadline=time.time() + limit,
                    allow_dirty=settings.allow_dirty,
                )
                lichess.mark_analysed(
                    config.data_dir, config.lichess_user, analysed["analysed_ids"]
                )
                summary["stages"]["reanalyse"] = {
                    "run": run_id,
                    **{k: v for k, v in analysed.items() if k != "analysed_ids"},
                }
                progress(
                    f"analysis: {analysed['analysed']} games, {analysed['mistakes']} mistakes, "
                    f"{analysed['positions']} positions"
                )
                folder = work / "reanalysis"
                buffer.add("lichess", night, (folder / "positions.txt").read_text())
                mistakes_text = (folder / "mistakes.txt").read_text()
                path = Path(config.data_dir) / "experience" / "experience.bin"
                path.parent.mkdir(parents=True, exist_ok=True)
                old = experience.read(path) if path.exists() else {}
                new = experience.merge(old, experience.read(folder / "experience.bin"))
                experience.write(path, new)
                summary["stages"]["experience"] = {
                    "entries": len(new),
                    "added": len(new) - len(old),
                }

        buffer.add("mistakes", night, mistakes_text)

        if late("training"):
            return finish()

        counts = replay.assemble(
            work / "train.txt",
            buffer,
            night,
            anchor=config.anchor_positions,
            seed=seed,
            history_lines=settings.history_lines,
            anchor_lines=settings.anchor_lines,
            mistake_repeat=settings.mistake_repeat,
        )
        summary["stages"]["data"] = counts
        run_id, trained = do["train"](
            config,
            store,
            engine,
            work / "train.txt",
            out=work / "candidate.nnue",
            epochs=settings.epochs,
            lr=settings.lr,
            init=deployed,
            seed=seed,
            allow_dirty=settings.allow_dirty,
        )
        summary["stages"]["train"] = {
            "run": run_id,
            "best_held": trained["best_held"],
            "best_epoch": trained["best_epoch"],
            "seconds": trained["seconds"],
            "network": trained["network"],
        }
        progress(f"training: held-out loss {trained['best_held']:.5f}")

        if late("match"):
            return finish()

        timeout = min(settings.match_minutes * 60, remaining(reserve=300))
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
            seed=seed,
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
        progress(f"match: {played['games']} games, verdict {test.get('verdict')}")

        summary["status"] = "passed" if passed else "rejected"

        if passed:
            summary["candidate"] = str(work / "candidate.nnue")
            summary["deployed"] = str(work / "deployed.nnue")
    except Exception as error:  # noqa: BLE001 - the night reports every failure
        summary["status"] = "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        summary["trace"] = traceback.format_exc()[-1500:]

    return finish()


def format_summary(summary: dict) -> str:
    """
    The summary of a night as text, for the second email.
    """
    lines = [f"Night {summary['night']}: {summary['status']}"]

    if "stopped_before" in summary:
        lines.append(f"Stopped before: {summary['stopped_before']}")

    if "error" in summary:
        lines.append(f"Error: {summary['error']}")

    stages = summary.get("stages", {})

    if "selfplay" in stages:
        s = stages["selfplay"]
        lines.append(f"Self-play: {s['games']} games, {s['positions']} positions")

    if "reanalyse" in stages:
        s = stages["reanalyse"]
        lines.append(
            f"Analysis: {s['analysed']} games, {s['mistakes']} mistakes, mean loss "
            f"{s['mean_loss_cp']:.0f} cp"
        )

    if "experience" in stages:
        lines.append(f"Experience: {stages['experience']['entries']} positions")

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
