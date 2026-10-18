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

import json

from meristats import runner, stats
from meristats.protocols import puzzles
from meristats.store import Store

COMPARE_PROTOCOL = "compare"
COMPARE_VERSION = 1


class ReportError(Exception):
    """
    A run cannot be reported or compared.
    """


def find_run(store: Store, text: str):
    """
    The run whose id is `text` or starts with it. A prefix that matches several runs is refused.
    """
    rows = store.db.execute(
        "SELECT * FROM runs WHERE run_id LIKE ? ORDER BY run_id", (text + "%",)
    ).fetchall()

    if not rows:
        raise ReportError(f"no run {text}")

    exact = [row for row in rows if row["run_id"] == text]

    if exact:
        return exact[0]

    if len(rows) > 1:
        raise ReportError(f"{text} matches {len(rows)} runs; use more of the id")

    return rows[0]


def _puzzle_weights(store: Store, run, items: int) -> list[tuple[float, int, int]]:
    """
    The weight, rating and mate length of the first `items` puzzles of the dataset of a run.
    """
    dataset = store.db.execute(
        "SELECT sha256 FROM datasets WHERE dataset_id = ?", (run["dataset_id"],)
    ).fetchone()

    if dataset is None:
        raise ReportError(f"run {run['run_id']} has no dataset")

    rows = []

    for line in store.get_blob(dataset["sha256"]).decode().splitlines():
        if len(rows) >= items:
            break

        fields = line.split("\t")
        rows.append((float(fields[3]), int(fields[1]), int(fields[2])))

    return rows


def _puzzle_items(store: Store, run) -> list[puzzles.Item]:
    if run["protocol"] != "puzzles" or not run["items_blob"]:
        raise ReportError(f"run {run['run_id']} has no per-puzzle results of the puzzle protocol")

    return puzzles.read_items(store.get_blob(run["items_blob"]))


def compare_puzzles(store: Store, id_a: str, id_b: str) -> dict:
    """
    Compares two puzzle runs on the same puzzles. The difference is in the share solved, A minus
    B, with a 95% interval and the exact McNemar test on the puzzles that only one of them
    solved, for the whole sample and for each kind of puzzle.
    """
    a, b = find_run(store, id_a), find_run(store, id_b)
    items_a, items_b = _puzzle_items(store, a), _puzzle_items(store, b)

    if a["dataset_id"] != b["dataset_id"]:
        raise ReportError("the runs used different datasets")

    if [i.item for i in items_a] != [i.item for i in items_b]:
        raise ReportError("the runs played different puzzles")

    meta = _puzzle_weights(store, a, len(items_a))
    weights = [meta[i.item][0] for i in items_a]
    solved_a, solved_b = [i.solved for i in items_a], [i.solved for i in items_b]
    result = {
        "a": {"run": a["run_id"], "config": a["config"], **stats.weighted_share(solved_a, weights)},
        "b": {"run": b["run_id"], "config": b["config"], **stats.weighted_share(solved_b, weights)},
        "whole": stats.paired_difference(solved_a, solved_b, weights),
        "items": len(items_a),
        "by_kind": {},
    }
    kinds = {i.item: meta[i.item][2] for i in items_a}

    for kind in sorted(set(kinds.values())):
        chosen = [n for n, i in enumerate(items_a) if kinds[i.item] == kind]
        result["by_kind"]["no mate" if kind == 0 else f"mate in {kind}"] = {
            "items": len(chosen),
            **stats.paired_difference(
                [solved_a[n] for n in chosen],
                [solved_b[n] for n in chosen],
                [weights[n] for n in chosen],
            ),
        }

    return result


def format_comparison(result: dict) -> str:
    """
    The comparison as text: the two shares, the difference with its interval and p value, and the
    difference for each kind of puzzle.
    """

    def line(name: str, diff: dict, items: int | None = None) -> str:
        noise = "inside the noise" if diff["low"] <= 0 <= diff["high"] else "outside the noise"
        count = f"{items:>7}  " if items is not None else ""
        return (
            f"{name:<10} {count}{100 * diff['difference']:+7.2f} points "
            f"({100 * diff['low']:+.2f} to {100 * diff['high']:+.2f}), only A {diff['only_a']}, "
            f"only B {diff['only_b']}, p {diff['p']:.3g}, {noise}"
        )

    lines = [
        f"A  {result['a']['config'] or 'default':<28} solved {100 * result['a']['p']:.2f}%  "
        f"{result['a']['run']}",
        f"B  {result['b']['config'] or 'default':<28} solved {100 * result['b']['p']:.2f}%  "
        f"{result['b']['run']}",
        line("all", result["whole"], result["items"]),
    ]
    lines += [line(kind, diff, diff["items"]) for kind, diff in result["by_kind"].items()]
    return "\n".join(lines)


def format_run(store: Store, text: str) -> str:
    """
    The headline numbers and the tables of one run.
    """
    run = find_run(store, text)
    summary = json.loads(run["summary_json"])
    head = [
        f"{run['run_id']}  {run['protocol']} v{run['protocol_version']}  {run['grade']}  "
        f"{run['status']}",
        f"config {run['config'] or 'default'}  started {run['started_utc']}",
    ]

    if run["protocol"] == "puzzles" and "whole" in summary:
        whole = summary["whole"]
        head.append(
            f"solved {100 * whole['p']:.2f}% ({100 * whole['low']:.2f} to "
            f"{100 * whole['high']:.2f}) of {summary['items']} puzzles, first move "
            f"{100 * whole['first_move']:.2f}%, puzzle rating "
            f"{whole['puzzle_rating']:.0f}, mean nodes {whole['mean_nodes']:.0f}"
        )

        for title, key in (
            ("by rating band", "by_rating_band"),
            ("by moves of the solver", "by_solver_moves"),
            ("by kind", "by_kind"),
        ):
            head.append(f"-- {title}")

            for name, row in summary[key].items():
                head.append(
                    f"  {name:<14} {100 * row['p']:6.2f}%  n {row['n']:>7}  "
                    f"mean nodes {row['mean_nodes']:.0f}"
                )

        weakest = sorted(summary["by_theme"].items(), key=lambda kv: kv[1]["p"])[:10]
        head.append("-- weakest themes")
        head += [f"  {name:<22} {100 * row['p']:6.2f}%  n {row['n']:>7}" for name, row in weakest]
        return "\n".join(head)

    head.append(json.dumps(summary, indent=2, sort_keys=True)[:4000])
    return "\n".join(head)


def save_comparison(
    config, store: Store, id_a: str, id_b: str, *, allow_dirty: bool = False, framework=...
) -> tuple[str, dict]:
    """
    Computes the comparison of two puzzle runs and stores it as a run of the protocol `compare`,
    so that it can be pinned and published. The run holds the two run ids, the result and the
    engine of run A. Its grade is `verified` only when both runs are verified and meristats has no
    changes.
    """
    a, b = find_run(store, id_a), find_run(store, id_b)
    result = compare_puzzles(store, a["run_id"], b["run_id"])
    context = runner.begin(config, store, allow_dirty=allow_dirty, framework=framework)

    if not (a["grade"] == "verified" == b["grade"]):
        context.grade = "unverified"

    run_id = runner.finish(
        context,
        protocol=COMPARE_PROTOCOL,
        version=COMPARE_VERSION,
        engine=None,
        config=f"{a['config'] or 'default'} against {b['config'] or 'default'}",
        limits={"a": a["run_id"], "b": b["run_id"]},
        dataset_id=a["dataset_id"],
        summary=result,
        items_blob=None,
    )
    return run_id, result


def limit_text(run) -> str:
    """
    The limit of a run as text, such as `nodes 20000`, or an empty string when it has none.
    """
    limit = json.loads(run["limits_json"]).get("limit") or {}
    return ", ".join(f"{key} {value}" for key, value in limit.items())


def default_label(run) -> str:
    """
    The label that a run gets on the site when none is given: the config and the limit, and for a
    comparison the config of the second run.
    """
    config = run["config"] or "default"

    if run["protocol"] == "compare" and " against " in config:
        return config.split(" against ", 1)[1]

    limit = limit_text(run)
    return f"{config} ({limit})" if limit else config
