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
The networks protocol: candidate networks of different shapes are trained and played against the
engine that is deployed, and the results are laid side by side.

A spec file (TOML) names the engine and the data, the length of the matches, and a list of
candidates. For each candidate the engine is built for its width (`CHESS_NNUE_HIDDEN`), the network
is trained, its speed is measured (nodes per second with the network), and a match against the
deployed engine is played. A match at a fixed number of nodes does not see that a wide network is
slower, so the speed is reported with the Elo, and `st` gives a match at a fixed time instead.

    engine = "HEAD"
    data = "/path/to/train.txt"
    rounds = 1000          # pairs of games, so 2000 games
    nodes = 50000          # or: st = 0.05
    seed = 1

    [[candidate]]
    name = "w256"
    hidden = 256
    epochs = 6
    lr = 0.0005
    init = "none"          # or "deployed", for a candidate of the width of the deployed network

The results go to `results.json` in the folder of the spec, and a candidate that is in the file is
not run again.
"""

import json
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from meristats import datasets, nnue
from meristats.build import BuiltEngine, build_engine, width_defines
from meristats.config import Config
from meristats.cycle import committed_network
from meristats.protocols import match as match_protocol
from meristats.protocols import train as train_protocol
from meristats.store import Store
from meristats.uci import UciEngine

SIMD_WIDTH = 16  # the engine adds the values of a layer 16 at a time
SPEED_NODES = 200_000


class NetworksError(ValueError):
    pass


@dataclass(frozen=True)
class Candidate:
    name: str
    hidden: int = nnue.HIDDEN
    epochs: int = 6
    lr: float = 0.0005
    batch: int = 16384
    lam: float = 0.7
    init: str = "none"
    data: Path | None = None
    limit: int = 0


@dataclass(frozen=True)
class Spec:
    engine: str
    data: Path
    limit: match_protocol.Limit
    rounds: int
    seed: int
    candidates: list[Candidate]
    folder: Path
    book: Path | None = None


def parameters(hidden: int) -> int:
    """
    The numbers in a network of this width.
    """
    return nnue.INPUTS * hidden + hidden + nnue.BUCKETS * 2 * hidden + nnue.BUCKETS


def load_spec(path: Path) -> Spec:
    """
    Reads and checks a spec file.
    """
    path = Path(path)

    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise NetworksError(f"cannot read {path}: {error}") from error

    for key in ("engine", "data", "rounds"):
        if key not in raw:
            raise NetworksError(f"{path} lacks {key}")

    if ("nodes" in raw) == ("st" in raw):
        raise NetworksError(f"{path} needs one of nodes and st")

    limit = (
        match_protocol.Limit("nodes", raw["nodes"])
        if "nodes" in raw
        else match_protocol.Limit("st", raw["st"])
    )
    candidates, names = [], set()

    for entry in raw.get("candidate", []):
        known = Candidate.__dataclass_fields__

        if unknown := set(entry) - set(known):
            raise NetworksError(f"a candidate has options that do not exist: {sorted(unknown)}")

        candidate = Candidate(**{**entry, "data": Path(entry["data"]) if "data" in entry else None})

        if candidate.name in names:
            raise NetworksError(f"two candidates are called {candidate.name}")

        if candidate.hidden % SIMD_WIDTH:
            raise NetworksError(f"{candidate.name}: hidden has to be a multiple of {SIMD_WIDTH}")

        if candidate.init not in ("none", "deployed"):
            raise NetworksError(f"{candidate.name}: init is none or deployed")

        if candidate.init == "deployed" and candidate.hidden != nnue.HIDDEN:
            raise NetworksError(
                f"{candidate.name}: the deployed network has {nnue.HIDDEN} values for each side"
            )

        names.add(candidate.name)
        candidates.append(candidate)

    if not candidates:
        raise NetworksError(f"{path} has no candidate")

    return Spec(
        engine=raw["engine"],
        data=Path(raw["data"]).expanduser(),
        limit=limit,
        rounds=int(raw["rounds"]),
        seed=int(raw.get("seed", 1)),
        candidates=candidates,
        folder=path.parent,
        book=Path(raw["book"]).expanduser() if "book" in raw else None,
    )


def measure_speed(binary: Path, network: Path | None, suite: Path, nodes: int = SPEED_NODES) -> int:
    """
    The nodes that the engine searches in a second, with the network, over the positions of the
    suite, each searched for a number of nodes.
    """
    total = 0
    seconds = 0.0

    with UciEngine(binary).start() as uci:
        uci.set_option("Hash", 16)
        uci.set_option("Threads", 1)

        if network is not None:
            uci.set_option("EvalFile", str(network))

        for entry in datasets.read_suite(suite):
            uci.new_game()
            uci.position(entry.fen, [])
            answer = uci.go(f"nodes {nodes}", timeout=300)
            counted = [info.nodes for info in answer.infos if info.nodes]
            total += counted[-1] if counted else 0
            seconds += answer.elapsed

    return round(total / seconds) if seconds else 0


def stages() -> dict[str, Callable]:
    """
    The functions that do the work, so that a test can replace them.
    """
    return {
        "build": build_engine,
        "train": train_protocol.run,
        "match": match_protocol.run,
        "speed": measure_speed,
        "deployed": committed_network,
    }


def run(
    config: Config,
    store: Store,
    spec: Spec,
    *,
    suite: Path,
    book: Path,
    allow_dirty: bool = False,
    functions: dict[str, Callable] | None = None,
    progress: Callable[[str], None] = lambda text: print(text, flush=True),
) -> list[dict]:
    """
    Runs the candidates that are not in the results of the folder of the spec, and returns all
    the results, in the order of the spec.
    """
    do = functions or stages()
    results_file = spec.folder / "results.json"
    done = (
        {row["name"]: row for row in json.loads(results_file.read_text())}
        if (results_file.exists())
        else {}
    )
    base = do["build"](config, store, spec.engine)
    base_speed = do["speed"](base.binary, None, suite)
    deployed = None

    for candidate in spec.candidates:
        if candidate.name in done:
            progress(f"{candidate.name}: in the results already")
            continue

        engine: BuiltEngine = do["build"](
            config, store, spec.engine, defines=width_defines(candidate.hidden)
        )
        folder = spec.folder / candidate.name
        folder.mkdir(parents=True, exist_ok=True)
        init = None

        if candidate.init == "deployed":
            deployed = deployed or do["deployed"](
                Path(config.engine_repo), base.commit_sha, spec.folder / "deployed.nnue"
            )
            init = deployed

        train_id, trained = do["train"](
            config,
            store,
            engine,
            candidate.data or spec.data,
            out=folder / "net.nnue",
            epochs=candidate.epochs,
            lr=candidate.lr,
            batch=candidate.batch,
            lam=candidate.lam,
            limit=candidate.limit,
            seed=spec.seed,
            init=init,
            hidden=candidate.hidden,
            allow_dirty=allow_dirty,
        )
        network = Path(trained["network"]["path"])
        speed = do["speed"](engine.binary, network, suite)
        match_id, played = do["match"](
            config,
            store,
            engine,
            base,
            spec.book or book,
            spec.limit,
            a_eval_file=network,
            rounds=spec.rounds,
            sprt=None,
            seed=spec.seed,
            allow_dirty=allow_dirty,
        )
        done[candidate.name] = {
            "name": candidate.name,
            "hidden": candidate.hidden,
            "parameters": parameters(candidate.hidden),
            "epochs": candidate.epochs,
            "lr": candidate.lr,
            "held_loss": trained["best_held"],
            "games": played["games"],
            "elo": played.get("elo"),
            "elo_low": played.get("elo_low"),
            "elo_high": played.get("elo_high"),
            "nps": speed,
            "speed": speed / base_speed if base_speed else None,
            "train_run": train_id,
            "match_run": match_id,
        }
        results_file.write_text(json.dumps(list(done.values()), indent=1))
        progress(format_table([done[candidate.name]]).splitlines()[-1])

    return [done[c.name] for c in spec.candidates if c.name in done]


def format_table(rows: list[dict]) -> str:
    """
    The results as a table, with the speed as a share of the speed of the deployed network.
    """

    def elo(row: dict) -> str:
        if row["elo"] is None:
            return "-"

        return f"{row['elo']:+.1f} ({row['elo_low']:+.1f} to {row['elo_high']:+.1f})"

    lines = [f"{'name':<14}{'hidden':>7}{'loss':>9}{'speed':>7}  Elo"]

    for row in rows:
        speed = "-" if row["speed"] is None else f"{row['speed']:.2f}x"
        lines.append(
            f"{row['name']:<14}{row['hidden']:>7}{row['held_loss']:>9.5f}{speed:>7}  {elo(row)}"
        )

    return "\n".join(lines)
