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
from html.parser import HTMLParser

import pytest

from meristats import publish as site
from meristats.cli import main
from meristats.publish import publish
from meristats.store import StoreError

PUZZLE_SUMMARY = {
    "items": 1000,
    "whole": {
        "p": 0.9337,
        "low": 0.9327,
        "high": 0.9347,
        "first_move": 0.9578,
        "puzzle_rating": 2466.4,
        "mean_nodes": 25456.0,
        "n": 1000,
        "n_eff": 900.0,
    },
    "errors": {"count": 0, "first": []},
}


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.local = set()

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name not in ("href", "src") or not value:
                continue

            if "://" not in value and not value.startswith("#"):
                self.local.add(value.split("#")[0].split("?")[0])


def add_run(
    store,
    protocol="puzzles",
    summary=None,
    grade="verified",
    config="network",
    limits=None,
    engine=True,
    started="2026-10-18T05:00:00Z",
):
    engine_id = (
        store.add_engine("meridian", "m", "a" * 40, "b" * 64, {}, 5, "Meridian 0.5.0")
        if engine
        else None
    )
    machine_id = store.add_machine(
        {"cpu_model": "Test CPU", "logical_cpus": 8, "kernel": "6.1", "os": "Linux"}
    )
    return store.add_run(
        protocol=protocol,
        protocol_version=2,
        grade=grade,
        status="complete",
        started_utc=started,
        finished_utc=started,
        engine_id=engine_id,
        config=config,
        limits=limits or {"limit": {"nodes": 20000}},
        dataset_id=None,
        seed=None,
        machine_id=machine_id,
        conditions={
            "samples": [
                {"at": "start", "power_source": "ac", "governor": "powersave", "load_1min": 1.2}
            ]
        },
        code_commit="c" * 40,
        code_dirty=False,
        summary=summary or PUZZLE_SUMMARY,
    )


def test_without_a_store_the_site_is_the_front_page_and_the_protocols(tmp_path):
    written = publish(tmp_path)

    assert {str(p.relative_to(tmp_path)) for p in written} == {
        "index.html",
        "protocols.html",
        "data/results.json",
        "assets/style.css",
        "assets/site.js",
        "assets/logo.svg",
        "assets/favicon.svg",
        ".nojekyll",
    }
    assert (tmp_path / ".nojekyll").read_text() == ""
    assert "No results are published yet." in (tmp_path / "index.html").read_text()
    assert json.loads((tmp_path / "data" / "results.json").read_text())["pages"] == {}


def test_every_local_reference_of_every_page_exists(tmp_path, store):
    run = add_run(store)
    store.pin("steps", "Step 10", run)
    publish(tmp_path, store)

    pages = list(tmp_path.rglob("*.html"))
    assert len(pages) == 3

    for page in pages:
        parser = References()
        parser.feed(page.read_text())

        for name in parser.local:
            target = (page.parent / name).resolve()
            target = target / "index.html" if target.is_dir() else target
            served = target if target.exists() else target.with_name(target.name + ".html")
            assert served.exists(), f"{page.name} names {name}"


def test_the_pinned_runs_make_pages_and_a_data_file_and_nothing_else(tmp_path, store):
    one = add_run(store)
    two = add_run(store, started="2026-10-18T06:00:00Z")
    add_run(store, started="2026-10-18T07:00:00Z")  # not pinned
    store.pin("steps", "Step 10", one)
    store.pin("techniques", "razoring on", two)
    out = tmp_path / "site"
    publish(out, store)
    names = {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()}

    assert names == {
        "index.html",
        "steps.html",
        "techniques.html",
        "protocols.html",
        "data/results.json",
        "assets/style.css",
        "assets/site.js",
        "assets/logo.svg",
        "assets/favicon.svg",
        ".nojekyll",
    }
    data = json.loads((out / "data" / "results.json").read_text())
    assert set(data["pages"]) == {"steps", "techniques"}
    assert [e["run"]["run_id"] for e in data["pages"]["steps"]] == [one]
    index = (out / "index.html").read_text()
    assert "2 runs are published." in index
    assert 'href="steps"' in index and 'href="protocols"' in index


def test_a_puzzle_run_shows_its_numbers_and_its_provenance(tmp_path, store):
    store.pin("steps", "Step 10", add_run(store))
    publish(tmp_path, store)
    page = (tmp_path / "steps.html").read_text()

    assert "93.4%" in page and "95.8%" in page and "2466" in page and "25,456" in page
    assert "nodes 20000" in page and "1,000" in page
    assert '<span class="badge">verified</span>' in page
    assert f'href="https://github.com/syedtaha22/meridian/commit/{"a" * 40}"' in page
    assert f'href="https://github.com/syedtaha22/meristats/commit/{"c" * 40}"' in page
    assert "Test CPU, 8 logical CPUs" in page and "powersave" in page
    assert "<h2>Puzzles</h2>" in page and "Run details" in page


def test_labels_and_text_are_escaped(tmp_path, store):
    run = add_run(store, config="<img src=x onerror=alert(1)>")
    store.pin("steps", "<script>alert(1)</script>", run)
    publish(tmp_path, store)
    page = (tmp_path / "steps.html").read_text()

    assert "<script>alert(1)</script>" not in page and "<img src=x" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "&lt;img src=x onerror=alert(1)&gt;" in page


def test_verified_and_unverified_runs_are_marked(tmp_path, store):
    store.pin("steps", "clean", add_run(store, grade="verified"))
    store.pin(
        "steps", "changed", add_run(store, grade="unverified", started="2026-10-18T06:00:00Z")
    )
    publish(tmp_path, store)
    page = (tmp_path / "steps.html").read_text()

    assert '<span class="badge">verified</span>' in page
    assert '<span class="badge warn">unverified</span>' in page
    assert "Made from a clean state." in page and "Made from a state with changes." in page


def test_search_rules_match_and_ladder_pages_render(tmp_path, store):
    nodes = {
        "network": {
            "6": {
                "positions": 15,
                "mean_nodes": 4030.0,
                "mean_ms": 4.9,
                "nodes_per_second": 823006.0,
            }
        }
    }
    rules = {
        "all_correct": True,
        "counts": 26,
        "wrong": [],
        "speed": {
            "initial": {"depth": 5, "nodes": 1, "median_seconds": 1, "nodes_per_second": 63e6}
        },
    }
    match = {
        "games": 80,
        "wins": 44,
        "draws": 27,
        "losses": 9,
        "score": 0.719,
        "elo": 163.0,
        "sprt": {"verdict": "H1", "fastchess_verdict": "H1", "llr": 3.0, "agree": True},
    }
    ladder = {
        "games": 300,
        "levels": {"2000": {"wins": 40, "draws": 5, "losses": 5}},
        "rating": {"value": 2930.4, "low": 2876.0, "high": 2986.0},
    }
    store.pin(
        "speed", "nodes", add_run(store, "search-nodes", nodes, started="2026-10-18T01:00:00Z")
    )
    store.pin("speed", "perft", add_run(store, "rules", rules, started="2026-10-18T02:00:00Z"))
    store.pin("matches", "threads", add_run(store, "match", match, started="2026-10-18T03:00:00Z"))
    store.pin(
        "stockfish", "ladder", add_run(store, "ladder", ladder, started="2026-10-18T04:00:00Z")
    )
    publish(tmp_path, store)
    speed = (tmp_path / "speed.html").read_text()
    matches = (tmp_path / "matches.html").read_text()
    stockfish = (tmp_path / "stockfish.html").read_text()

    assert "4,030" in speed and "823" in speed and "63.0" in speed and "yes" in speed
    assert "+44 =27 -9" in matches and "71.9%" in matches and "+163" in matches
    assert ">H1<" in matches and "SPRT, fastchess" in matches
    assert (
        "2930" in stockfish
        and "2876 to 2986" in stockfish
        and "Level 2000" in stockfish
        and "+40 =5 -5" in stockfish
    )


def test_an_unknown_page_and_an_unknown_protocol_still_render(tmp_path, store):
    run = add_run(store, "tune", {"files": ["a"]})
    store.pin("my-notes", "stage 1", run)
    publish(tmp_path, store)

    assert "My notes" in (tmp_path / "index.html").read_text()
    assert "stage 1" in (tmp_path / "my-notes.html").read_text()


def test_publishing_twice_gives_identical_files(tmp_path, store):
    store.pin("steps", "Step 10", add_run(store))
    publish(tmp_path / "a", store)
    publish(tmp_path / "b", store)

    for path in (tmp_path / "a").rglob("*"):
        if path.is_file():
            relative = path.relative_to(tmp_path / "a")
            assert path.read_bytes() == (tmp_path / "b" / relative).read_bytes()


def test_the_protocols_page_defines_every_protocol_with_its_version(tmp_path):
    publish(tmp_path)
    page = (tmp_path / "protocols.html").read_text()

    for protocol in site.PROTOCOL_TEXT:
        assert f'id="{protocol}"' in page

    assert "8moves_v3" in page and "Wilson" in page and "pentanomial" in page


def test_the_newest_pin_wins_and_a_removed_label_leaves_the_page(tmp_path, store):
    first, second = add_run(store), add_run(store, started="2026-10-18T06:00:00Z")
    store.pin("steps", "Step 10", first)
    store.pin("steps", "Step 10", second)
    store.pin("steps", "Step 09", first)
    store.pin("steps", "Step 09", None)

    assert [(p["label"], p["run_id"]) for p in store.pins()] == [("Step 10", second)]

    store.pin("steps", "Step 10", None)
    publish(tmp_path, store)
    assert not (tmp_path / "steps.html").exists()


def test_pinning_a_run_that_does_not_exist_is_refused(store):
    with pytest.raises(StoreError, match="no run"):
        store.pin("steps", "x", "nope")


def test_pins_cannot_be_changed_or_removed(store):
    import sqlite3

    store.pin("steps", "x", add_run(store))

    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        store.db.execute("UPDATE pins SET label = 'y'")

    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        store.db.execute("DELETE FROM pins")


def test_the_publish_command_uses_the_store_when_there_are_settings(tmp_path, config_file, capsys):
    out = tmp_path / "site"

    assert main(["--config", str(config_file), "publish", "--out", str(out)]) == 0
    assert (out / "index.html").exists() and str(out / "index.html") in capsys.readouterr().out


def test_the_pin_commands(tmp_path, config_file, capsys):
    from meristats.config import load
    from meristats.store import Store

    with Store(load(config_file).data_dir) as store:
        run = add_run(store)

    assert (
        main(["--config", str(config_file), "pin", run, "--page", "steps", "--label", "Step 10"])
        == 0
    )
    assert main(["--config", str(config_file), "pins"]) == 0
    assert "Step 10" in capsys.readouterr().out
    assert main(["--config", str(config_file), "runs"]) == 0
    assert run in capsys.readouterr().out
    assert (
        main(["--config", str(config_file), "unpin", "--page", "steps", "--label", "Step 10"]) == 0
    )
    assert (
        main(["--config", str(config_file), "pin", "nope", "--page", "steps", "--label", "x"]) == 1
    )


def test_the_pages_are_listed_in_a_fixed_order_with_unknown_pages_last(tmp_path, store):
    for index, page in enumerate(["zzz", "speed", "stockfish", "steps", "matches", "aaa"]):
        store.pin(page, "x", add_run(store, started=f"2026-10-18T0{index}:00:00Z"))

    out = tmp_path / "site"
    publish(out, store)
    index = (out / "index.html").read_text()
    positions = [
        index.index(f'href="{name}"')
        for name in ("steps", "speed", "matches", "stockfish", "aaa", "zzz")
    ]

    assert positions == sorted(positions)


def test_a_comparison_run_renders_the_difference_and_the_reading(tmp_path, store):
    inside = {
        "a": {"p": 0.934, "config": "network"},
        "b": {"p": 0.9345, "config": "network+x"},
        "whole": {
            "difference": -0.0005,
            "low": -0.002,
            "high": 0.001,
            "only_a": 12,
            "only_b": 15,
            "p": 0.7,
        },
        "items": 1000,
        "by_kind": {},
    }
    outside = {
        "a": {"p": 0.934, "config": "network"},
        "b": {"p": 0.9, "config": "network+y"},
        "whole": {
            "difference": 0.034,
            "low": 0.03,
            "high": 0.038,
            "only_a": 90,
            "only_b": 3,
            "p": 1e-20,
        },
        "items": 1000,
        "by_kind": {},
    }
    store.pin("techniques", "x on", add_run(store, "compare", inside, config="network against x"))
    store.pin(
        "techniques", "y on", add_run(store, "compare", outside, started="2026-10-18T06:00:00Z")
    )
    publish(tmp_path / "site", store)
    page = (tmp_path / "site" / "techniques.html").read_text()

    assert "+0.05" in page and "-0.10 to +0.20" in page and "inside the noise" in page
    assert "-3.40" in page and "outside the noise" in page
    assert "Only A solved" in page and "<td>12</td>" in page and "<td>15</td>" in page
    assert "<h2>Comparisons</h2>" in page


def test_a_page_has_a_way_back_and_loads_the_script(tmp_path, store):
    store.pin("steps", "Step 10", add_run(store))
    publish(tmp_path, store)

    for name in ("steps", "protocols"):
        page = (tmp_path / f"{name}.html").read_text()
        assert '<a class="back" href="./">' in page

    page = (tmp_path / "steps.html").read_text()
    assert '<script src="assets/site.js" defer></script>' in page
    assert '<link rel="stylesheet" href="assets/style.css">' in page
    assert '<script src="assets/site.js" defer></script>' in (tmp_path / "index.html").read_text()


def test_a_table_can_be_searched_filtered_and_sorted(tmp_path, store):
    store.pin("steps", "Step 10", add_run(store, grade="verified"))
    store.pin(
        "steps", "Step 11", add_run(store, grade="unverified", started="2026-10-18T06:00:00Z")
    )
    store.pin(
        "steps",
        "Step 12",
        add_run(store, limits={"limit": {"depth": 4}}, started="2026-10-18T07:00:00Z"),
    )
    publish(tmp_path, store)
    page = (tmp_path / "steps.html").read_text()

    assert page.count("data-table") == 1 and 'type="search"' in page
    assert 'data-filter="grade"' in page and 'data-filter="group"' in page
    assert 'data-value="unverified"' in page and 'data-value="depth 4"' in page
    assert page.count('data-sort="num"') == 3 and page.count('data-sort="text"') == 2
    assert "3 of 3" in page

    for row in page.split("<tr data-run=")[1:]:
        assert 'data-search="' in row.split(">")[0]


def test_a_link_in_a_table_names_the_details_of_its_run(tmp_path, store):
    store.pin("steps", "Step 10", add_run(store))
    publish(tmp_path, store)
    page = (tmp_path / "steps.html").read_text()
    link = re.search(r'<a class="run-link" href="#(run-[^"]+)"', page)

    assert link and f'<details class="run" id="{link.group(1)}"' in page
    assert f'<tr data-run="{link.group(1)}"' in page


def test_the_details_of_a_run_are_tables(tmp_path, store):
    store.pin("steps", "Step 10", add_run(store))
    publish(tmp_path, store)
    page = (tmp_path / "steps.html").read_text()
    details = page.split('<details class="run"')[1].split("</details>")[0]

    assert "<dl" not in page and '<table class="facts">' in details
    for group in ("Result", "Run", "Engine", "Machine", "Framework", "Conditions"):
        assert f'<th colspan="2">{group}</th>' in details
    assert '<th scope="row">Solved</th>' in details
    assert '<th scope="row">Governor</th><td>powersave</td>' in details


def test_the_front_page_searches_every_published_run(tmp_path, store):
    store.pin("steps", "Step 10", add_run(store))
    store.pin("matches", "<b>x</b>", add_run(store, started="2026-10-18T06:00:00Z"))
    publish(tmp_path, store)
    index = (tmp_path / "index.html").read_text()
    data = json.loads(
        re.search(r'<script type="application/json" id="site-index">(.*?)</script>', index).group(1)
    )

    assert {item["label"] for item in data} == {"Step 10", "<b>x</b>"}
    assert all(item["href"].split("#")[0] in ("steps", "matches") for item in data)
    assert "<b>x</b>" not in index
    assert "No results are published yet." not in index and 'id="site-search"' in index


def test_the_front_page_has_no_search_without_runs(tmp_path):
    publish(tmp_path)

    assert 'id="site-search"' not in (tmp_path / "index.html").read_text()


def test_the_count_of_a_page_is_in_the_singular_for_one_run(tmp_path, store):
    store.pin("steps", "Step 10", add_run(store))
    publish(tmp_path, store)

    assert "1 run</span>" in (tmp_path / "index.html").read_text()


def test_links_are_not_underlined_and_the_icon_has_room_on_every_side():
    from importlib.resources import files

    css = files("meristats").joinpath("static", "style.css").read_text()
    icon = files("meristats").joinpath("static", "favicon.svg").read_text()
    width, height = re.search(r'viewBox="[\d. ]+ [\d. ]+ ([\d.]+) ([\d.]+)"', icon).groups()

    assert "a { color: var(--accent); text-decoration: none; }" in css
    assert width == height


def test_runs_can_be_filtered_and_pins_get_a_default_label(tmp_path, config_file, capsys):
    from meristats.config import load
    from meristats.store import Store

    with Store(load(config_file).data_dir) as store:
        first = add_run(store, grade="verified", config="baseline")
        second = add_run(
            store, grade="unverified", config="network+see=0", started="2026-10-18T06:00:00Z"
        )

    def run(*arguments):
        code = main(["--config", str(config_file), *arguments])
        return code, capsys.readouterr().out

    assert run("runs", "--grade", "verified")[1].count("\n") == 1
    assert first in run("runs", "--grade", "verified")[1]
    assert (
        second in run("runs", "--search", "see=0")[1]
        and first not in run("runs", "--search", "see=0")[1]
    )
    assert run("runs", "--last", "1")[1].count("\n") == 1
    assert "nodes 20000" in run("runs")[1]

    code, out = run("pin", first, second, "--page", "steps")
    assert code == 0 and "baseline (nodes 20000)" in out and "network+see=0 (nodes 20000)" in out
    assert run("pin", first, second, "--page", "steps", "--label", "x")[0] == 1
    assert {p["label"] for p in store_pins(config_file)} == {
        "baseline (nodes 20000)",
        "network+see=0 (nodes 20000)",
    }
    assert run("pin", first[:12], "--page", "steps", "--label", "Base")[0] == 0


def store_pins(config_file):
    from meristats.config import load
    from meristats.store import Store

    with Store(load(config_file).data_dir) as store:
        return store.pins()


def test_a_table_is_paged_on_the_same_page(tmp_path, store):
    for index in range(3):
        store.pin("steps", f"Step {index}", add_run(store, started=f"2026-10-18T0{index}:00:00Z"))

    publish(tmp_path, store)
    page = (tmp_path / "steps.html").read_text()
    select = re.search(r'<select aria-label="Rows per page">(.*?)</select>', page).group(1)

    assert re.findall(r'<option value="(\d+)">', select) == ["10", "25", "50", "100"]
    assert '<div class="pager" hidden>' in page and '<nav class="pages"' in page
    assert page.count("<tr data-run=") == 3 and (tmp_path / "steps.html").exists()
    assert not list(tmp_path.glob("steps-*.html"))


def test_the_script_is_valid_javascript_and_pages_the_table(tmp_path):
    import shutil
    import subprocess
    from importlib.resources import files

    source = files("meristats").joinpath("static", "site.js").read_text()

    assert "Math.ceil(matched.length / size)" in source and "meristats-page-size" in source

    node = shutil.which("node")

    if node is None:
        pytest.skip("node is not installed")

    script = tmp_path / "site.js"
    script.write_text(source)
    done = subprocess.run([node, "--check", str(script)], capture_output=True, text=True)

    assert done.returncode == 0, done.stderr


def test_every_table_has_a_sort_selector_with_each_column_in_both_directions(tmp_path, store):
    store.pin("steps", "Step 10", add_run(store))
    publish(tmp_path, store)
    page = (tmp_path / "steps.html").read_text()
    select = re.search(r'<select aria-label="Sort the table">(.*?)</select>', page).group(1)
    values = re.findall(r'<option value="([^"]*)">', select)

    assert values == [""] + [f"{i}:{d}" for i in range(5) for d in ("asc", "desc")]
    assert "Solved, high to low" in select and "Run, A to Z" in select


def test_the_protocols_page_is_a_centred_column(tmp_path):
    from importlib.resources import files

    publish(tmp_path)
    css = files("meristats").joinpath("static", "style.css").read_text()

    assert '<div class="wrap narrow">' in (tmp_path / "protocols.html").read_text()
    assert ".wrap.narrow { max-width: 56rem; }" in css


def test_the_favicon_draws_the_logo_with_the_same_paths_and_strokes():
    from importlib.resources import files

    icon = files("meristats").joinpath("static", "favicon.svg").read_text()
    logo = files("meristats").joinpath("static", "logo.svg").read_text()

    for pattern in (r' d="([^"]+)"', r'stroke-width="([\d.]+)"'):
        assert re.findall(pattern, icon) == re.findall(pattern, logo)


def test_the_sort_icons_are_svg_and_the_pager_arrows_are_not_text(tmp_path, store):
    from importlib.resources import files

    store.pin("steps", "Step 10", add_run(store))
    publish(tmp_path, store)
    page = (tmp_path / "steps.html").read_text()
    script = files("meristats").joinpath("static", "site.js").read_text()
    css = files("meristats").joinpath("static", "style.css").read_text()

    assert page.count('<svg class="sort-icon"') == 5
    assert not any(ord(char) in (0x2191, 0x2193, 0x2195, 0x2039, 0x203A) for char in page + css)
    assert "\\u2039" not in script and "\\u203a" not in script


def test_publishing_again_removes_what_the_last_publication_wrote(tmp_path, store):
    run = add_run(store)
    store.pin("steps", "Step 10", run)
    store.pin("techniques", "x", run)
    publish(tmp_path, store)
    (tmp_path / "old-page").mkdir()
    (tmp_path / "old-page" / "index.html").write_text("a page in a folder of its own")
    (tmp_path / "style.css").write_text("an old stylesheet")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref")
    (tmp_path / "notes.txt").write_text("mine")
    store.pin("techniques", "x", None)
    publish(tmp_path, store)

    assert not (tmp_path / "old-page").exists() and not (tmp_path / "style.css").exists()
    assert not (tmp_path / "techniques.html").exists()
    assert (tmp_path / "steps.html").exists()
    assert (tmp_path / ".git" / "HEAD").read_text() == "ref"
    assert (tmp_path / "notes.txt").read_text() == "mine"


def test_the_site_is_served_without_the_extension_or_a_trailing_slash(tmp_path, store):
    import threading
    import urllib.error
    import urllib.request

    from meristats.serve import make_server

    store.pin("steps", "Step 10", add_run(store))
    publish(tmp_path, store)
    server = make_server(tmp_path, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def get(path):
        with urllib.request.urlopen(base + path, timeout=10) as answer:
            return answer.status, answer.read().decode()

    try:
        assert "Puzzles" in get("/steps")[1] and "Puzzles" in get("/steps.html")[1]
        assert "Protocols" in get("/protocols")[1] and "Search" in get("/")[1]
        assert get("/assets/style.css")[0] == 200

        with pytest.raises(urllib.error.HTTPError) as missing:
            get("/nothing")

        assert missing.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
