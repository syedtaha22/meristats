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

import random
import shutil
import sys
from pathlib import Path

from meristats import datasets, preflight, runner
from meristats.build import BuiltEngine
from meristats.config import Config
from meristats.preflight import require
from meristats.protocols import match
from meristats.store import Store

PROTOCOL = "spsa"
VERSION = 1
SEED = 7

# technique -> parameters: name, start, perturbation, minimum, maximum
PARAMETERS = {
    "reverse_futility": [("rfp_margin", 107, 15, 40, 300)],
    "futility": [("futility_base", 74, 15, 20, 400), ("futility_step", 214, 20, 30, 400)],
    "razoring": [("razor_base", 800, 60, 300, 1500), ("razor_step", 300, 40, 50, 600)],
    "late_move_pruning": [("lmp_base", 3, 1, 1, 8)],
    "lmr_table": [("lmr_scale", 225, 20, 120, 400)],
    "null_margin": [("null_margin_step", 246, 30, 60, 500)],
    "aspiration": [("aspiration_delta", 18, 6, 8, 80)],
    "delta_pruning": [("delta_margin", 200, 30, 50, 500)],
    "iir": [("iir_depth", 6, 1, 3, 10)],
    "singular": [("singular_depth", 8, 1, 5, 12)],
}

# The techniques that the presets `tuned` and `network` have on, and the one that only `network`
# has on and that has parameters here. The start values above are the defaults of `SearchOptions`,
# and a base can name another value with `+name=value`.
DEFAULT_ON = {"reverse_futility", "futility", "null_margin", "aspiration"}
NETWORK_ON = {"lmr_table"}


def parameters_of(base: str) -> list[tuple[str, int, int, int, int]]:
    """
    The parameters to tune for the techniques in the base, with their start values, as
    (name, start, perturbation, minimum, maximum).
    """
    changes = base.split("+")[1:]
    flags = {c.split("=")[0] for c in changes if not c.endswith("=0")}
    off = {c.split("=")[0] for c in changes if c.endswith("=0")}

    if base.split("+")[0] in ("tuned", "network"):
        flags |= DEFAULT_ON - off

    if base.split("+")[0] == "network":
        flags |= NETWORK_ON - off

    explicit = {c.split("=")[0]: int(c.split("=")[1]) for c in changes if "=" in c}
    found = []

    for flag, parameters in PARAMETERS.items():
        if flag in flags:
            for name, start, step, low, high in parameters:
                found.append((name, explicit.get(name, start), step, low, high))

    return found


def preset_string(prefix: str, values: dict[str, float]) -> str:
    """
    The `Config` value of a base with the parameters set to `values`, rounded.
    """
    return prefix + "".join(f"+{name}={round(value)}" for name, value in values.items())


def base_prefix(base: str, names) -> str:
    """
    The base without the parameters that are tuned.
    """
    kept = [c for c in base.split("+") if "=" not in c or c.split("=")[0] not in names]
    return "+".join(kept)


def perturbation(step: int, k: int) -> float:
    """
    The size of the perturbation at iteration `k`, which shrinks as the tuning goes on.
    """
    return max(1.0, step / k**0.101)


def perturb(theta: dict, parameters, k: int, rng: random.Random):
    """
    The random signs of iteration `k` and the two parameter sets, moved up and down by them.
    """
    signs = {name: rng.choice((-1, 1)) for name in theta}
    plus, minus = {}, {}

    for name, _, step, low, high in parameters:
        c = perturbation(step, k)
        plus[name] = min(high, max(low, theta[name] + c * signs[name]))
        minus[name] = min(high, max(low, theta[name] - c * signs[name]))

    return signs, plus, minus


def update(theta: dict, parameters, signs: dict, score: float, k: int) -> dict:
    """
    The parameters after iteration `k`: moved toward the side that won, by a step that falls with
    `k`.
    """
    gain = (score - 0.5) * 6 / (1 + k / 30) ** 0.602
    moved = {}

    for name, _, step, low, high in parameters:
        c = perturbation(step, k)
        moved[name] = min(high, max(low, theta[name] + gain * c * signs[name]))

    return moved


def run(
    config: Config,
    store: Store,
    engine: BuiltEngine,
    base: str,
    book: Path,
    limit: match.Limit,
    *,
    iterations: int = 120,
    pairs: int = 24,
    concurrency: int | None = None,
    hash_mb: int = 16,
    allow_dirty: bool = False,
    framework=...,
    check_compliance: bool = True,
    progress=lambda text: print(text, file=sys.stderr, flush=True),
) -> tuple[str, dict]:
    """
    Tunes the numeric parameters of the techniques in `base` by simultaneous perturbation (SPSA).

    Each iteration plays a short match, `pairs` pairs of games, between the base with every
    parameter moved up and the base with every parameter moved down, by a random sign for each
    parameter, and moves the parameters toward the side that won. The matches have no sequential
    test, and the run stores every iteration, the final parameters, and the base with them set.
    """
    parameters = parameters_of(base)

    if not parameters:
        raise ValueError(f"nothing to tune in {base}")

    context = runner.begin(config, store, allow_dirty=allow_dirty, framework=framework)
    book_id, book_hash = datasets.register_file(
        store, "opening-book", "8moves_v3", book, {"file": book.name}, keep_copy=False
    )
    checks = runner.engine_checks(engine) + [preflight.dataset_hash(book, book_hash)]

    if check_compliance and config.fastchess:
        checks.append(match.compliance(Path(config.fastchess), engine.binary))

    context.checks += require(checks)
    context.sample("before")
    rng = random.Random(SEED)
    theta = {name: float(start) for name, start, _, _, _ in parameters}
    prefix = base_prefix(base, theta)
    trajectory = []
    options = {"Threads": 1, "Hash": hash_mb}

    if limit.kind == "st":
        options["MoveOverhead"] = 0

    for k in range(1, iterations + 1):
        signs, plus, minus = perturb(theta, parameters, k, rng)
        a = match.Player(
            "A",
            engine.binary,
            {**options, "Config": preset_string(prefix, plus)},
            engine.engine_id,
        )
        b = match.Player(
            "B",
            engine.binary,
            {**options, "Config": preset_string(prefix, minus)},
            engine.engine_id,
        )
        work = config.cache_dir / "matches" / f"{context.started_utc.replace(':', '')}-spsa"
        pgn_text, output = match.play(
            config,
            a,
            b,
            limit,
            book,
            work,
            rounds=pairs,
            sprt=None,
            seed=1000 + k,
            concurrency=concurrency,
        )
        shutil.rmtree(work, ignore_errors=True)
        summary = match.summarise(match.parse_games(pgn_text), None, output)
        score = summary.get("score", 0.5)
        theta = update(theta, parameters, signs, score, k)
        trajectory.append(
            {"k": k, "score": score, "theta": {n: round(v, 3) for n, v in theta.items()}}
        )

        if k % 10 == 0 or k == iterations:
            progress(f"iteration {k}: " + str({n: round(v, 1) for n, v in theta.items()}))

    context.sample("end")
    tuned = preset_string(prefix, theta)
    summary = {
        "base": base,
        "tuned": tuned,
        "parameters": [
            {"name": n, "start": s, "step": st, "min": lo, "max": hi}
            for n, s, st, lo, hi in parameters
        ],
        "final": {n: round(v, 3) for n, v in theta.items()},
        "iterations": iterations,
        "pairs": pairs,
        "trajectory": trajectory,
    }
    run_id = runner.finish(
        context,
        protocol=PROTOCOL,
        version=VERSION,
        engine=engine,
        config=base,
        limits={
            "limit": {limit.kind: limit.value},
            "iterations": iterations,
            "pairs": pairs,
            "seed": SEED,
            "options": options,
        },
        dataset_id=book_id,
        summary=summary,
        items_blob=None,
        seed=SEED,
    )
    return run_id, summary
