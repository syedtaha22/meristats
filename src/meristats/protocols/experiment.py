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

import sys
from dataclasses import dataclass
from pathlib import Path

from meristats import report, runner
from meristats.build import BuiltEngine
from meristats.config import Config
from meristats.protocols import match, puzzles, search_nodes
from meristats.store import Store

PROTOCOL = "experiment"
VERSION = 1
NODES_RATIO_LIMIT = 0.95
PUZZLE_ALPHA = 0.05


@dataclass(frozen=True)
class Settings:
    """
    What an experiment runs: the puzzles, the depths of the node count, and the match.
    """

    puzzle_items: int = 200_000
    puzzle_nodes: int = 20_000
    depths: tuple[int, ...] = (5, 7)
    st: float = 0.05
    rounds: int = match.ROUNDS
    sprt: match.Sprt = match.Sprt()
    seed: int = 1
    cpu: int | None = None


def decide(puzzle_difference: dict, match_summary: dict, nodes_ratio: float | None):
    """
    Decides whether a change is kept. It is kept when all of these hold:

    - the puzzles do not get significantly worse: the paired difference in the share solved is
      not negative with a p value below 0.05;
    - the sequential test of the match accepts H1, or it has no decision, the score is at least
      0.5, and the nodes at the deepest depth are at most 0.95 of the base.

    Returns whether to keep it, and the reasons in words.
    """
    reasons = []
    worse = puzzle_difference["difference"] < 0 and puzzle_difference["p"] < PUZZLE_ALPHA
    reasons.append(
        f"puzzles {100 * puzzle_difference['difference']:+.2f} points "
        f"(p {puzzle_difference['p']:.3g}): " + ("worse" if worse else "not worse")
    )
    sprt = match_summary.get("sprt", {})
    verdict = sprt.get("verdict", "continue")
    score = match_summary.get("score", 0.5)
    reasons.append(f"match score {100 * score:.1f}%, SPRT {verdict}")
    fewer = nodes_ratio is not None and nodes_ratio <= NODES_RATIO_LIMIT
    reasons.append(
        "nodes at the deepest depth: "
        + ("not measured" if nodes_ratio is None else f"{nodes_ratio:.3f} of the base")
    )
    won = verdict == "H1" or (verdict == "continue" and score >= 0.5 and fewer)
    return (not worse and won), reasons


def nodes_ratio(summary: dict, new: str, base: str) -> float | None:
    """
    The mean nodes of `new` over the mean nodes of `base` at the deepest depth of a `search-nodes`
    summary, or None when one of them has no row.
    """
    rows_new, rows_base = summary.get(new), summary.get(base)

    if not rows_new or not rows_base:
        return None

    depth = max(rows_new, key=int)

    if depth not in rows_base or not rows_base[depth]["mean_nodes"]:
        return None

    return rows_new[depth]["mean_nodes"] / rows_base[depth]["mean_nodes"]


def run(
    config: Config,
    store: Store,
    engine: BuiltEngine,
    base: str,
    change: str,
    suite: Path,
    book: Path,
    dataset: Path,
    settings: Settings = Settings(),
    *,
    base_puzzle_run: str | None = None,
    allow_dirty: bool = False,
    framework=...,
    progress=lambda text: print(text, file=sys.stderr, flush=True),
) -> tuple[str, dict]:
    """
    Tests one change to the search against a base, and stores the result.

    The change is added to the base with `+`, as in `network+razoring`. The new configuration is
    played on the puzzles (the base too, unless `base_puzzle_run` names a run of the base on the
    same puzzles), compared on the nodes at fixed depths, and played against the base in a match
    with a sequential test. `decide` says whether to keep it. Every part is a run of its own, and
    the experiment is a run that names them.
    """
    new = f"{base}+{change}"
    limit = puzzles.Limit("nodes", settings.puzzle_nodes)
    common = {"allow_dirty": allow_dirty, "framework": framework}

    if base_puzzle_run is None:
        progress(f"puzzles of {base}")
        base_puzzle_run, _ = puzzles.run(
            config,
            store,
            engine,
            dataset,
            limit,
            engine_config=base,
            items=settings.puzzle_items,
            progress=progress,
            **common,
        )

    progress(f"puzzles of {new}")
    new_puzzle_run, _ = puzzles.run(
        config,
        store,
        engine,
        dataset,
        limit,
        engine_config=new,
        items=settings.puzzle_items,
        progress=progress,
        **common,
    )
    compare_id, comparison = report.save_comparison(
        config,
        store,
        new_puzzle_run,
        base_puzzle_run,
        allow_dirty=allow_dirty,
        framework=framework,
    )
    progress(f"nodes at depths {list(settings.depths)}")
    nodes_id, nodes_summary = search_nodes.run(
        config,
        store,
        engine,
        suite,
        [base, new],
        list(settings.depths),
        rounds=1,
        cpu=settings.cpu,
        progress=progress,
        **common,
    )
    ratio = nodes_ratio(nodes_summary, new, base)
    progress(f"match {new} against {base}")
    match_id, match_summary = match.run(
        config,
        store,
        engine,
        engine,
        book,
        match.Limit("st", settings.st),
        a_config=new,
        b_config=base,
        rounds=settings.rounds,
        sprt=settings.sprt,
        seed=settings.seed,
        progress=progress,
        **common,
    )
    keep, reasons = decide(comparison["whole"], match_summary, ratio)
    context = runner.begin(config, store, **common)
    parts = [store.get_run(r) for r in (new_puzzle_run, compare_id, nodes_id, match_id)]

    if any(part["grade"] != "verified" for part in parts):
        context.grade = "unverified"

    summary = {
        "base": base,
        "new": new,
        "change": change,
        "keep": keep,
        "reasons": reasons,
        "nodes_ratio": ratio,
        "runs": {
            "base_puzzles": base_puzzle_run,
            "new_puzzles": new_puzzle_run,
            "comparison": compare_id,
            "nodes": nodes_id,
            "match": match_id,
        },
    }
    run_id = runner.finish(
        context,
        protocol=PROTOCOL,
        version=VERSION,
        engine=engine,
        config=new,
        limits={
            "base": base,
            "change": change,
            "puzzle_items": settings.puzzle_items,
            "puzzle_nodes": settings.puzzle_nodes,
            "depths": list(settings.depths),
            "st": settings.st,
            "rounds": settings.rounds,
        },
        dataset_id=None,
        summary=summary,
        items_blob=None,
        seed=settings.seed,
    )
    progress(f"{new}: {'kept' if keep else 'dropped'}: " + "; ".join(reasons))
    return run_id, summary


def run_changes(
    config: Config,
    store: Store,
    engine: BuiltEngine,
    base: str,
    changes: list[str],
    suite: Path,
    book: Path,
    dataset: Path,
    settings: Settings = Settings(),
    **options,
) -> tuple[str, list[tuple[str, dict]]]:
    """
    Tests a list of changes one after the other. A change that is kept is added to the base of the
    next one, unless it starts with `!`, which tests it without keeping it. Returns the final base
    and the result of each change.
    """
    results = []
    base_run = None

    for change in changes:
        test_only = change.startswith("!")
        name = change.lstrip("!")
        run_id, summary = run(
            config,
            store,
            engine,
            base,
            name,
            suite,
            book,
            dataset,
            settings,
            base_puzzle_run=base_run,
            **options,
        )
        results.append((run_id, summary))

        if summary["keep"] and not test_only:
            base = summary["new"]
            base_run = summary["runs"]["new_puzzles"]

    return base, results
