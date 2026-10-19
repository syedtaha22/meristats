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
import re
import shutil
from dataclasses import dataclass
from html import escape, unescape
from importlib.resources import files
from pathlib import Path
from string import Template

from meristats import __version__
from meristats.store import Store

TITLE = "meristats"
PAGE_SIZES = (10, 25, 50, 100)
SORT_ICON = (
    '<svg class="sort-icon" viewBox="0 0 10 14" aria-hidden="true">'
    '<path class="up" d="M5 1 9 6H1z"/><path class="down" d="M5 13 1 8h8z"/></svg>'
)
TAGLINE = "Tests and published results for the Meridian chess engine."
ENGINE_URL = "https://github.com/syedtaha22/meridian"
SOURCE_URL = "https://github.com/syedtaha22/meristats"
HOME_URL = "https://syedtaha22.github.io/"
ASSETS = ("style.css", "site.js", "logo.svg", "favicon.svg")

PAGES = {
    "steps": ("Puzzles", "The share of Lichess puzzles that the engine solves under each limit."),
    "techniques": (
        "Technique comparison",
        "What switching one technique on or off changes, on the same puzzles.",
    ),
    "speed": ("Speed and correctness", "Search speed, and move generation checked by perft."),
    "matches": ("Matches", "Engine against engine, with a sequential test."),
    "stockfish": ("Rating against Stockfish", "The engine against Stockfish at several levels."),
}

PROTOCOL_TEXT = {
    "puzzles": (
        "Puzzles",
        2,
        [
            "The sample is 1,000,000 puzzles of the Lichess puzzle database: every mate in 3, 4 "
            "and 5 moves, and the others in proportion by kind and band of 200 rating points, "
            "with seed 1. Each puzzle has a weight, the number of puzzles of the database that it "
            "stands for.",
            "Each move of the solver is searched from empty tables with one thread and a hash of "
            "16 MB, limited by nodes, by depth, or by time per move (with MoveOverhead 0).",
            "A move is right when it is the move of the solution or when it gives checkmate. A "
            "puzzle is solved when every move of the solver is right, and the line stops at the "
            "first wrong move.",
            "Shares are weighted, with Wilson intervals at 95% from the effective sample size. The "
            "puzzle rating is the rating at which the logistic model of solving a puzzle fits the "
            "weighted results best.",
        ],
    ),
    "search-nodes": (
        "Search to a fixed depth",
        1,
        [
            "Each position of a suite of 15 (5 openings, 5 middlegames, 5 endgames) is searched "
            "from empty tables to each depth, with one thread and a hash of 16 MB.",
            "The moves, scores and node counts must be equal in every round, and the time is the "
            "median over the rounds, with the configurations interleaved.",
        ],
    ),
    "rules": (
        "Move generation",
        1,
        [
            "Six reference positions are counted by perft at every depth whose published count is "
            "at most 200 million nodes, and compared with the published counts. The deepest such "
            "depth of each position is repeated 7 times for the speed."
        ],
    ),
    "match": (
        "Match",
        2,
        [
            "Two engines play pairs of games from the openings of the book 8moves_v3 (the two "
            "games of an opening swap the colors), with one thread and a hash of 16 MB, at a "
            "fixed time per move, depth or node count. A move that takes up to 20 ms longer than "
            "its time per move is not lost on time.",
            "The sequential test (SPRT) has Elo bounds 0 and 10 and error probabilities 0.05, on "
            "the pentanomial counts of the pairs. The verdicts of meristats and of fastchess are "
            "both stored.",
        ],
    ),
    "compare": (
        "Comparisons",
        1,
        [
            "Two runs on the same puzzles are compared. The difference is in the share solved, A "
            "minus B, with a 95% interval from the weighted differences, and the exact McNemar "
            "test counts the puzzles that only one of them solved. A difference whose interval "
            "holds 0 is inside the noise."
        ],
    ),
    "ladder": (
        "Rating against Stockfish",
        2,
        [
            "The engine plays Stockfish 19 at several levels (UCI_LimitStrength and UCI_Elo), and "
            "the rating is fitted from all levels together, with the levels as fixed ratings.",
            "The rating is on the scale of UCI_Elo, which Stockfish calibrates for longer time "
            "controls, so it holds for this speed on this machine and is not a rating-list rating.",
        ],
    ),
}

PROVENANCE_TEXT = (
    "Every result has a run. A run records the engine commit and the hash of the binary, the "
    "machine, the settings, the data that it used, the commit of this framework, and the "
    "conditions when it ran."
)

GRADE_TEXT = {
    "verified": "Made from a clean state.",
    "unverified": "Made from a state with changes.",
}


@dataclass
class Entry:
    label: str
    run: dict
    engine: dict | None
    machine: dict | None
    dataset: dict | None

    @property
    def summary(self) -> dict:
        return json.loads(self.run["summary_json"])

    @property
    def limits(self) -> dict:
        return json.loads(self.run["limits_json"])

    @property
    def conditions(self) -> dict:
        return json.loads(self.run["conditions_json"])


def _row(row) -> dict | None:
    return dict(row) if row is not None else None


def collect(store: Store) -> dict[str, list[Entry]]:
    """
    The pinned runs of each page, in the order of their labels.
    """
    pages: dict[str, list[Entry]] = {}

    for pin in store.pins():
        run = store.get_run(pin["run_id"])
        engine = store.db.execute(
            "SELECT * FROM engines WHERE engine_id = ?", (run["engine_id"],)
        ).fetchone()
        machine = store.db.execute(
            "SELECT * FROM machines WHERE machine_id = ?", (run["machine_id"],)
        ).fetchone()
        dataset = store.db.execute(
            "SELECT * FROM datasets WHERE dataset_id = ?", (run["dataset_id"],)
        ).fetchone()
        pages.setdefault(pin["page"], []).append(
            Entry(pin["label"], dict(run), _row(engine), _row(machine), _row(dataset))
        )

    return pages


# ----------------------------------------------------------------------- rendering


def _e(value) -> str:
    return escape("" if value is None else str(value))


def _pct(value, digits: int = 1) -> str:
    return "n/a" if value is None else f"{100 * value:.{digits}f}%"


def _plain(html: str) -> str:
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]*>", " ", html))).strip()


def _limit(limits: dict) -> str:
    limit = limits.get("limit") or {}
    return (
        ", ".join(
            f"{key} {value:g}" if isinstance(value, float) else f"{key} {value}"
            for key, value in limit.items()
        )
        or "not recorded"
    )


def _commit_link(repo_url: str, sha: str | None) -> str:
    if not sha or len(sha) != 40:
        return _e(sha or "unavailable")

    return f'<a href="{repo_url}/commit/{_e(sha)}">{_e(sha[:12])}</a>'


def _anchor(entry: Entry) -> str:
    return "run-" + "".join(c if c.isalnum() else "-" for c in entry.run["run_id"]).lower()


def _stack(main: str, small: str = "") -> str:
    """
    A table cell with a main value and a smaller line below it.
    """
    below = f'<span class="small">{small}</span>' if small else ""
    return f'<span class="main">{main}</span>{below}'


def _interval(low: str, high: str) -> str:
    return f"{low} to {high}"


def _grade_dot(grade: str) -> str:
    return f'<span class="dot dot-{_e(grade)}" title="{_e(grade)}"></span>'


def _grade_badge(grade: str) -> str:
    css = "badge" if grade == "verified" else "badge warn"
    return f'<span class="{css}">{_e(grade)}</span>'


def _run_cell(entry: Entry, sub: str = "") -> str:
    """
    The first cell of a row: the label, which opens the details of the run, and a line with the
    grade.
    """
    line = _e(entry.run["grade"]) + (f" - {sub}" if sub else "")
    link = f'<a class="run-link" href="#{_anchor(entry)}">{_e(entry.label)}</a>'
    return f'<span class="main">{_grade_dot(entry.run["grade"])}{link}</span>' + (
        f'<span class="small">{line}</span>'
    )


@dataclass
class Cell:
    html: str
    sort: str | float = ""


@dataclass
class Row:
    entry: Entry
    cells: list[Cell]
    group: str = ""


@dataclass
class Table:
    columns: list[tuple[str, bool]]
    rows: list[Row]
    group_title: str = ""


def _sort_select(columns: list[tuple[str, bool]]) -> str:
    """
    The selector of the order of a table: its own order, or one of its columns in either direction.
    """
    options = ['<option value="">Default order</option>']

    for index, (name, numeric) in enumerate(columns):
        low, high = ("low to high", "high to low") if numeric else ("A to Z", "Z to A")
        options.append(f'<option value="{index}:asc">{_e(name)}, {low}</option>')
        options.append(f'<option value="{index}:desc">{_e(name)}, {high}</option>')

    return (
        '<label class="sort"><span>Sort</span><select aria-label="Sort the table">'
        + "".join(options)
        + "</select></label>"
    )


def _render_table(table: Table) -> str:
    grades = list(dict.fromkeys(row.entry.run["grade"] for row in table.rows))
    groups = list(dict.fromkeys(row.group for row in table.rows if row.group))
    head = "".join(
        f'<th scope="col" data-sort="{"num" if numeric else "text"}" '
        f'class="{"num" if numeric else ""}"><button type="button">{_e(name)}'
        f"{SORT_ICON}</button></th>"
        for name, numeric in table.columns
    )
    body = []

    for row in table.rows:
        cells = "".join(
            f'<td class="{"num" if table.columns[index][1] else ""}" '
            f'data-sort="{_e(cell.sort)}">{cell.html}</td>'
            for index, cell in enumerate(row.cells)
        )
        text = " ".join(
            [
                row.entry.label,
                row.entry.run["config"] or "",
                row.entry.run["protocol"],
                row.entry.run["grade"],
                row.group,
                *(_plain(cell.html) for cell in row.cells),
            ]
        ).lower()
        body.append(
            f'<tr data-run="{_anchor(row.entry)}" data-grade="{_e(row.entry.run["grade"])}" '
            f'data-group="{_e(row.group)}" data-search="{_e(text)}">{cells}</tr>'
        )

    def chips(kind: str, title: str, values: list[str]) -> str:
        if len(values) < 2:
            return ""

        buttons = "".join(
            f'<button type="button" class="chip" data-value="{_e(value)}">{_e(value)}</button>'
            for value in values
        )
        return (
            f'<div class="chips" data-filter="{kind}" role="group" aria-label="{_e(title)}">'
            f'<button type="button" class="chip on" data-value="">All</button>{buttons}</div>'
        )

    toolbar = (
        '<div class="toolbar"><label class="search">'
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
        'stroke-linecap="round" aria-hidden="true"><circle cx="11" cy="11" r="6.5"/>'
        '<path d="M20 20l-4.2-4.2"/></svg>'
        '<input type="search" placeholder="Search" aria-label="Search the table"></label>'
        + _sort_select(table.columns)
        + chips("grade", "Grade", grades)
        + chips("group", table.group_title or "Group", groups)
        + f'<span class="count" aria-live="polite">{len(table.rows)} of {len(table.rows)}</span>'
        "</div>"
    )
    return (
        f'<div class="panel" data-table>{toolbar}<div class="table-wrap"><table class="results">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
        '<p class="empty" hidden>Nothing matches.</p></div>'
        '<div class="pager" hidden><label class="per-page">Rows<select aria-label="Rows per page">'
        + "".join(f'<option value="{size}">{size}</option>' for size in PAGE_SIZES)
        + '</select></label><span class="range" aria-live="polite"></span>'
        '<nav class="pages" aria-label="Pages"></nav></div></div>'
    )


# ------------------------------------------------------------------- one table for each protocol


def _puzzles_table(entries: list[Entry]) -> Table:
    rows = []

    for entry in entries:
        summary = entry.summary
        whole = summary.get("whole")

        if not whole:
            continue

        share, rating, nodes = whole, whole["puzzle_rating"], whole["mean_nodes"]

        limit = _limit(entry.limits)
        rows.append(
            Row(
                entry,
                [
                    Cell(_run_cell(entry, f"{summary['items']:,} puzzles"), entry.label.lower()),
                    Cell(
                        _stack(
                            _pct(share["p"]),
                            _interval(_pct(share["low"]), _pct(share["high"])),
                        ),
                        share["p"],
                    ),
                    Cell(f"{rating:.0f}", rating),
                    Cell(f"{nodes:,.0f}", nodes),
                    Cell(_e(limit), limit),
                ],
                limit,
            )
        )

    return Table(
        [
            ("Run", False),
            ("Solved", True),
            ("Rating", True),
            ("Mean nodes", True),
            ("Limit", False),
        ],
        rows,
        "Limit",
    )


def _nodes_table(entries: list[Entry]) -> Table:
    rows = []

    for entry in entries:
        summary = entry.summary
        table = summary.get("by_config_depth") or {
            k: v for k, v in summary.items() if isinstance(v, dict)
        }

        for config, depths in table.items():
            for depth, row in sorted(depths.items(), key=lambda item: int(item[0])):
                speed = row.get("nodes_per_second")
                rows.append(
                    Row(
                        entry,
                        [
                            Cell(_run_cell(entry), entry.label.lower()),
                            Cell(_e(config), config),
                            Cell(_e(depth), int(depth)),
                            Cell(f"{row['mean_nodes']:,.0f}", row["mean_nodes"]),
                            Cell(f"{row['mean_ms']:.2f}", row["mean_ms"]),
                            Cell("n/a" if not speed else f"{speed / 1000:,.0f}", speed or 0),
                        ],
                        f"depth {depth}",
                    )
                )

    return Table(
        [
            ("Run", False),
            ("Config", False),
            ("Depth", True),
            ("Mean nodes", True),
            ("Mean ms", True),
            ("kNPS", True),
        ],
        rows,
        "Depth",
    )


def _rules_table(entries: list[Entry]) -> Table:
    rows = []

    for entry in entries:
        summary = entry.summary
        speeds = summary.get("speed", {})
        best = max((s["nodes_per_second"] for s in speeds.values()), default=None)
        correct = bool(summary.get("all_correct"))
        rows.append(
            Row(
                entry,
                [
                    Cell(_run_cell(entry), entry.label.lower()),
                    Cell("yes" if correct else "NO", int(correct)),
                    Cell(_e(summary.get("counts", summary.get("positions", ""))), 0),
                    Cell("n/a" if best is None else f"{best / 1e6:.1f}", best or 0),
                ],
                "all counts correct" if correct else "a count is wrong",
            )
        )

    return Table(
        [("Run", False), ("Correct", False), ("Counts", True), ("Best Mnps", True)],
        rows,
        "Result",
    )


def _elo_text(summary: dict) -> str:
    elo = summary.get("elo")

    if elo is None or elo in (float("inf"), float("-inf")):
        return "n/a"

    return f"{elo:+.0f}"


def _match_table(entries: list[Entry]) -> Table:
    rows = []

    for entry in entries:
        summary = entry.summary
        sprt = summary.get("sprt")
        verdict = f"{_e(sprt['verdict'])}" if sprt else "n/a"
        score = summary.get("score")
        rows.append(
            Row(
                entry,
                [
                    Cell(_run_cell(entry, f"{summary.get('games', 0)} games"), entry.label.lower()),
                    Cell(_pct(score), score or 0),
                    Cell(_elo_text(summary), summary.get("elo") or 0),
                    Cell(
                        f"+{summary.get('wins', 0)} ={summary.get('draws', 0)} "
                        f"-{summary.get('losses', 0)}",
                        summary.get("wins", 0),
                    ),
                    Cell(verdict, verdict),
                ],
                f"SPRT {verdict}" if sprt else "",
            )
        )

    return Table(
        [("Run", False), ("Score", True), ("Elo", True), ("W D L", True), ("SPRT", False)],
        rows,
        "Test",
    )


def _ladder_table(entries: list[Entry]) -> Table:
    rows = []

    for entry in entries:
        summary = entry.summary
        rating = summary.get("rating")
        rows.append(
            Row(
                entry,
                [
                    Cell(_run_cell(entry, f"{summary.get('games', 0)} games"), entry.label.lower()),
                    Cell(
                        "n/a"
                        if not rating
                        else _stack(
                            f"{rating['value']:.0f}",
                            _interval(f"{rating['low']:.0f}", f"{rating['high']:.0f}"),
                        ),
                        rating["value"] if rating else 0,
                    ),
                ],
            )
        )

    return Table([("Run", False), ("Rating", True)], rows)


def _compare_table(entries: list[Entry]) -> Table:
    rows = []

    for entry in entries:
        whole = entry.summary["whole"]
        inside = whole["low"] <= 0 <= whole["high"]
        change = -whole["difference"]
        low, high = -whole["high"], -whole["low"]
        reading = "inside the noise" if inside else "outside the noise"
        direction = "" if inside else ("better" if change > 0 else "worse")
        shown = f"{100 * change:+.2f}"
        rows.append(
            Row(
                entry,
                [
                    Cell(_run_cell(entry), entry.label.lower()),
                    Cell(
                        _stack(
                            f'<span class="{"pos" if change > 0 else "neg"}">{shown}</span>'
                            if not inside
                            else shown,
                            _interval(f"{100 * low:+.2f}", f"{100 * high:+.2f}"),
                        ),
                        change,
                    ),
                    Cell(
                        f'<span class="pill {"quiet" if inside else "loud"}">{reading}</span>',
                        int(not inside),
                    ),
                    Cell(f"{whole['p']:.3g}", whole["p"]),
                ],
                f"{reading}" + (f", {direction}" if direction else ""),
            )
        )

    return Table(
        [("Run", False), ("Change in points", True), ("Reading", False), ("p", True)],
        rows,
        "Reading",
    )


def _other_table(entries: list[Entry]) -> Table:
    rows = [
        Row(
            entry,
            [
                Cell(_run_cell(entry), entry.label.lower()),
                Cell(_e(entry.run["protocol"]), entry.run["protocol"]),
                Cell(_e(entry.run["config"] or ""), entry.run["config"] or ""),
            ],
        )
        for entry in entries
    ]
    return Table([("Run", False), ("Protocol", False), ("Config", False)], rows)


TABLES = {
    "compare": _compare_table,
    "puzzles": _puzzles_table,
    "search-nodes": _nodes_table,
    "rules": _rules_table,
    "match": _match_table,
    "ladder": _ladder_table,
}


# ------------------------------------------------------------------------- run details


def _result_facts(entry: Entry) -> list[tuple[str, str]]:
    """
    The result of a run, as the rows of the Result group of its details.
    """
    summary, protocol = entry.summary, entry.run["protocol"]

    if protocol == "puzzles":
        whole = summary.get("whole")

        if not whole:
            return []

        return [
            ("Solved", f"{_pct(whole['p'])} ({_pct(whole['low'])} to {_pct(whole['high'])})"),
            ("First move", _pct(whole["first_move"])),
            ("Puzzle rating", f"{whole['puzzle_rating']:.0f}"),
            ("Mean nodes", f"{whole['mean_nodes']:,.0f}"),
            ("Puzzles", f"{summary['items']:,}"),
            ("Limit", _limit(entry.limits)),
        ]

    if protocol == "compare":
        whole = summary["whole"]
        return [
            ("A solved", f"{_pct(summary['a']['p'])} ({summary['a'].get('config', '')})"),
            ("B solved", f"{_pct(summary['b']['p'])} ({summary['b'].get('config', '')})"),
            (
                "Change, B minus A",
                f"{-100 * whole['difference']:+.2f} points "
                f"({-100 * whole['high']:+.2f} to {-100 * whole['low']:+.2f})",
            ),
            ("Only A solved", str(whole["only_a"])),
            ("Only B solved", str(whole["only_b"])),
            ("p (McNemar)", f"{whole['p']:.3g}"),
            ("Puzzles", f"{summary.get('items', 0):,}"),
        ]

    if protocol == "match":
        facts = [
            ("Games", str(summary.get("games", ""))),
            (
                "Wins, draws, losses",
                f"+{summary.get('wins', 0)} ={summary.get('draws', 0)} -{summary.get('losses', 0)}",
            ),
            ("Score", _pct(summary.get("score"))),
            ("Elo", _elo_text(summary)),
        ]
        sprt = summary.get("sprt")

        if sprt:
            facts += [
                ("SPRT, meristats", str(sprt["verdict"])),
                ("SPRT, fastchess", str(sprt["fastchess_verdict"])),
            ]

        return facts

    if protocol == "ladder":
        rating = summary.get("rating")
        facts = [
            (
                "Rating",
                "n/a"
                if not rating
                else f"{rating['value']:.0f} ({rating['low']:.0f} to {rating['high']:.0f})",
            ),
            ("Games", str(summary.get("games", ""))),
        ]
        facts += [
            (f"Level {level}", f"+{row['wins']} ={row['draws']} -{row['losses']}")
            for level, row in summary.get("levels", {}).items()
        ]
        return facts

    if protocol == "rules":
        facts = [
            ("All counts correct", "yes" if summary.get("all_correct") else "NO"),
            ("Counts", str(summary.get("counts", summary.get("positions", "")))),
        ]
        facts += [
            (f"Speed, {name}", f"{row['nodes_per_second'] / 1e6:.1f} Mnps at depth {row['depth']}")
            for name, row in summary.get("speed", {}).items()
        ]
        return facts

    if protocol == "search-nodes":
        table = summary.get("by_config_depth") or {
            k: v for k, v in summary.items() if isinstance(v, dict)
        }
        return [
            ("Configurations", ", ".join(table)),
            (
                "Depths",
                ", ".join(sorted({d for depths in table.values() for d in depths}, key=int)),
            ),
        ]

    return []


def _limits_text(limits: dict) -> str:
    return "<br>".join(f"{_e(key)}: {_e(value)}" for key, value in sorted(limits.items()))


def _facts(entry: Entry) -> list[tuple[str, list[tuple[str, str]]]]:
    """
    The details of a run in groups, each a name and a list of (field, html) rows.
    """
    run, engine, machine = entry.run, entry.engine, entry.machine
    samples = entry.conditions.get("samples") or []
    first = samples[0] if samples else {}
    groups = [
        ("Result", [(name, _e(value)) for name, value in _result_facts(entry)]),
        (
            "Run",
            [
                ("Run", _e(run["run_id"])),
                ("Protocol", _e(f"{run['protocol']} version {run['protocol_version']}")),
                ("Grade", f"{_grade_badge(run['grade'])} {_e(GRADE_TEXT.get(run['grade'], ''))}"),
                ("Started", _e(run["started_utc"])),
                ("Config", _e(run["config"])),
                ("Limits", _limits_text(entry.limits)),
                ("Notes", _e(run["notes"])),
            ],
        ),
        (
            "Engine",
            [
                (
                    "Commit",
                    None if not engine else _commit_link(ENGINE_URL, engine["commit_sha"]),
                ),
                ("Binary", None if not engine else _e(engine["binary_sha256"])),
                ("Name", None if not engine else _e(engine["version_text"])),
            ],
        ),
        (
            "Data",
            [
                (
                    "Dataset",
                    None
                    if not entry.dataset
                    else _e(f"{entry.dataset['name']} {entry.dataset['sha256'][:12]}"),
                )
            ],
        ),
        (
            "Machine",
            [
                (
                    "CPU",
                    None
                    if not machine
                    else _e(f"{machine['cpu_model']}, {machine['logical_cpus']} logical CPUs"),
                ),
                ("Kernel", None if not machine else _e(machine["kernel"])),
            ],
        ),
        (
            "Framework",
            [
                (
                    "Commit",
                    _commit_link(SOURCE_URL, run["code_commit"]) if run["code_commit"] else None,
                ),
                (
                    "State",
                    None
                    if run["code_dirty"] is None
                    else ("had changes" if run["code_dirty"] else "clean"),
                ),
            ],
        ),
        (
            "Conditions",
            [
                ("Power source", _e(first.get("power_source"))),
                ("Governor", _e(first.get("governor"))),
                ("Load at start", _e(first.get("load_1min"))),
            ],
        ),
    ]
    return [
        (name, [(field, value) for field, value in rows if value not in (None, "")])
        for name, rows in groups
    ]


def _headline(entry: Entry) -> str:
    facts = dict(_result_facts(entry))

    for key in ("Solved", "Rating", "Score", "Change, B minus A", "All counts correct"):
        if key in facts:
            return f"{key.split(',')[0].lower()} {facts[key].split(' (')[0]}"

    return entry.run["protocol"]


def _details(entry: Entry) -> str:
    groups = []

    for name, rows in _facts(entry):
        if not rows:
            continue

        body = "".join(
            f'<tr><th scope="row">{_e(field)}</th><td>{value}</td></tr>' for field, value in rows
        )
        groups.append(
            f'<tbody><tr class="group"><th colspan="2">{_e(name)}</th></tr>{body}</tbody>'
        )

    return (
        f'<details class="run" id="{_anchor(entry)}" data-run="{_anchor(entry)}">'
        f'<summary><span class="run-name">{_grade_dot(entry.run["grade"])}{_e(entry.label)}</span>'
        f'<span class="run-meta">{_e(_headline(entry))}</span>'
        '<svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<path d="M6 9l6 6 6-6"/></svg></summary>'
        f'<div class="run-body"><table class="facts">{"".join(groups)}</table>'
        '<a class="to-top" href="#top">Back to the table</a></div></details>'
    )


# ---------------------------------------------------------------------------- documents


def _render(template: str, **values) -> str:
    text = files("meristats").joinpath("templates", template).read_text(encoding="utf-8")
    return Template(text).substitute(**values)


def _document(page_title: str, description: str, body: str) -> str:
    return _render(
        "base.html",
        title=TITLE,
        page_title=_e(page_title),
        description=_e(description),
        body=body,
        version=__version__,
        source_url=SOURCE_URL,
        engine_url=ENGINE_URL,
        home_url=HOME_URL,
    )


BACK = (
    '<a class="back" href="./"><svg viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
    'aria-hidden="true"><path d="M15 6l-6 6 6 6"/></svg>meristats</a>'
)


def _page_body(title: str, description: str, entries: list[Entry]) -> str:
    sections = []

    for protocol in dict.fromkeys(e.run["protocol"] for e in entries):
        chosen = [e for e in entries if e.run["protocol"] == protocol]
        heading = PROTOCOL_TEXT.get(protocol, (protocol,))[0]
        table = TABLES.get(protocol, _other_table)(chosen)
        note = f'<p class="note">{_e(TABLE_NOTES[protocol])}</p>' if protocol in TABLE_NOTES else ""
        sections.append(f"<section><h2>{_e(heading)}</h2>{_render_table(table)}{note}</section>")

    details = "".join(_details(e) for e in entries)
    return (
        f'<div class="wrap" id="top">{BACK}<section class="page-head">'
        f'<p class="eyebrow">Results</p><h1>{_e(title)}</h1><p>{_e(description)}</p></section>'
        f"{''.join(sections)}"
        f'<section><h2>Run details</h2><p class="note">{_e(PROVENANCE_TEXT)} Select a run in a '
        f'table to open its details.</p><div class="runs" data-runs>{details}</div></section></div>'
    )


TABLE_NOTES = {
    "compare": (
        "Change is the second run minus the first, in points of puzzles solved, with the 95% "
        "interval below it. A run whose interval holds 0 is inside the noise."
    ),
}


def _protocols_body() -> str:
    parts = []

    for protocol, (title, version, paragraphs) in PROTOCOL_TEXT.items():
        text = "".join(f"<p>{_e(p)}</p>" for p in paragraphs)
        parts.append(
            f'<section class="protocol" id="{_e(protocol)}"><h2>{_e(title)} '
            f'<span class="badge">{_e(protocol)} v{version}</span></h2>{text}</section>'
        )

    grades = "".join(
        f'<tr><th scope="row">{_grade_badge(grade)}</th><td>{_e(text)}</td></tr>'
        for grade, text in GRADE_TEXT.items()
    )
    return (
        f'<div class="wrap narrow">{BACK}<section class="page-head"><p class="eyebrow">Method</p>'
        "<h1>Protocols</h1><p>Every number comes from a fixed, versioned protocol. A change to "
        "a protocol is a new version, and earlier runs keep theirs.</p></section>"
        + "".join(parts)
        + f'<section><h2>Grades</h2><div class="table-wrap"><table class="facts grades">'
        f"<tbody>{grades}</tbody></table></div></section></div>"
    )


def _results_json(pages: dict[str, list[Entry]]) -> str:
    data = {"generator": f"meristats {__version__}", "pages": {}}

    for page, entries in pages.items():
        data["pages"][page] = [
            {
                "label": e.label,
                "run": {k: v for k, v in e.run.items() if k not in ("items_blob",)},
                "summary": e.summary,
                "limits": e.limits,
                "engine": e.engine,
                "machine": e.machine,
                "dataset": e.dataset,
            }
            for e in entries
        ]

    return json.dumps(data, indent=2, sort_keys=True, default=str) + "\n"


def _search_index(pages: dict[str, list[Entry]], titles: dict[str, str]) -> str:
    """
    The runs of every page, as the data of the search box of the front page.
    """
    items = [
        {
            "label": entry.label,
            "page": titles[page],
            "href": f"{page}#{_anchor(entry)}",
            "grade": entry.run["grade"],
            "text": _headline(entry),
        }
        for page, entries in pages.items()
        for entry in entries
    ]
    return json.dumps(items).replace("<", "\\u003c")


def _clear(out_dir: Path) -> None:
    """
    Removes what an earlier publication wrote into `out_dir`: the files of the front page and its
    assets, and every folder that holds an `index.html`, with `assets` and `data`. Other files and
    the `.git` entry stay.
    """
    for entry in out_dir.iterdir():
        if entry.name == ".git":
            continue

        if entry.is_dir():
            if entry.name in ("assets", "data") or (entry / "index.html").exists():
                shutil.rmtree(entry)
        elif entry.suffix in (".html", ".json", ".css", ".js", ".svg"):
            entry.unlink()


def publish(out_dir: Path, store: Store | None = None) -> list[Path]:
    """
    Writes the site into `out_dir` and returns the paths of the files that it wrote.

    The site has a fixed set of pages: the front page, one `<page>.html` for each page that has
    pinned runs, the protocols, the files of the pages in `assets`, and one data file,
    `data/results.json`, with the pinned runs. The links name a page without `.html`, as GitHub
    Pages and `meristats serve` serve it. A run has no page of its own, and nothing is written
    for a run that is not pinned. With no store the site is the front page with a note that nothing
    is published. What an earlier publication wrote into `out_dir` is removed first.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = collect(store) if store is not None else {}
    written = []

    def write(name: str, text: str) -> None:
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append(path)

    _clear(out_dir)

    cards = []
    titles = {}
    order = list(PAGES)

    for number, page in enumerate(
        sorted(pages, key=lambda name: (order.index(name) if name in order else len(order), name)),
        1,
    ):
        title, description = PAGES.get(page, (page.replace("-", " ").capitalize(), ""))
        titles[page] = title
        write(
            f"{page}.html",
            _document(
                f"{title} - {TITLE}", description, _page_body(title, description, pages[page])
            ),
        )
        cards.append(
            f'          <a class="card" href="{_e(page)}"><div class="num">'
            f"{number:02d}</div><h3>{_e(title)}</h3><p>{_e(description)}</p>"
            f'<span class="card-count">{len(pages[page])} '
            f"{'run' if len(pages[page]) == 1 else 'runs'}</span></a>"
        )

    cards.append(
        f'          <a class="card" href="protocols"><div class="num">'
        f"{len(pages) + 1:02d}</div><h3>Protocols</h3><p>How each result is "
        "measured, with the exact parameters.</p></a>"
    )
    count = sum(len(entries) for entries in pages.values())
    status = f"{count} runs are published." if count else "No results are published yet."
    detail = (
        "Search them here, or open a page."
        if count
        else "Results appear here when the first run is published."
    )
    body = _render(
        "home.html",
        title=TITLE,
        tagline=TAGLINE,
        engine_url=ENGINE_URL,
        source_url=SOURCE_URL,
        status=_e(status),
        status_detail=_e(detail),
        page_cards="\n".join(cards),
        search="" if not count else _render("search.html", index=_search_index(pages, titles)),
    )
    write("index.html", _document(TITLE, TAGLINE, body))
    write(
        "protocols.html",
        _document(f"Protocols - {TITLE}", "How each result is measured.", _protocols_body()),
    )
    write("data/results.json", _results_json(pages))

    for name in ASSETS:
        target = out_dir / "assets" / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(files("meristats").joinpath("static", name).read_bytes())
        written.append(target)

    marker = out_dir / ".nojekyll"
    marker.write_text("", encoding="utf-8")
    written.append(marker)
    return written
