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

import pytest

from meristats import report
from meristats.protocols import puzzles
from meristats.report import ReportError, compare_puzzles, find_run, format_comparison, format_run


def write_sample(path, rows):
    """
    rows: (id, rating, mate, weight)
    """
    with open(path, "w") as file:
        for puzzle_id, rating, mate, weight in rows:
            file.write(f"{puzzle_id}\t{rating}\t{mate}\t{weight}\ttheme\tfen\te7e5\n")


@pytest.fixture
def two_runs(store, tmp_path):
    """
    Two puzzle runs on 6 puzzles of two kinds: A solves puzzles 0 to 4, B solves 1 to 5.
    """
    sample = tmp_path / "s.tsv"
    write_sample(
        sample,
        [
            ("p0", 1500, 0, 2.0),
            ("p1", 1500, 0, 1.0),
            ("p2", 1500, 3, 1.0),
            ("p3", 1500, 3, 1.0),
            ("p4", 1500, 0, 1.0),
            ("p5", 1500, 3, 4.0),
        ],
    )
    from meristats import datasets

    dataset_id, _ = datasets.register_file(store, "puzzles", "1", sample, {})
    engine = store.add_engine("meridian", "m", "a" * 40, "b" * 64, {}, 5, "M")
    runs = []

    for name, solved in (("A", {0, 1, 2, 3, 4}), ("B", {1, 2, 3, 4, 5})):
        items = [
            puzzles.Item(n, 1, 1 if n in solved else 0, 0 if n in solved else 1, 10, 1.0, (1,))
            for n in range(6)
        ]
        blob = store.put_blob(puzzles.items_table(items), "items")
        runs.append(
            store.add_run(
                protocol="puzzles",
                protocol_version=2,
                grade="verified",
                status="complete",
                started_utc=f"2026-10-18T0{len(runs)}:00:00Z",
                finished_utc="x",
                engine_id=engine,
                config=f"network+{name}",
                limits={"limit": {"nodes": 1}},
                dataset_id=dataset_id,
                seed=None,
                machine_id=None,
                conditions={},
                code_commit=None,
                code_dirty=None,
                summary={"items": 6},
                items_blob=blob.sha256,
                blobs=((blob.sha256, "items"),),
            )
        )

    return runs


def test_the_difference_counts_the_puzzles_that_only_one_solved(store, two_runs):
    result = compare_puzzles(store, *two_runs)
    whole = result["whole"]

    assert whole["only_a"] == 1 and whole["only_b"] == 1 and whole["p"] == 1.0
    # weights: p0 2, p5 4 -> A minus B = (2 - 4) / 10
    assert whole["difference"] == pytest.approx(-0.2) and result["items"] == 6
    assert result["a"]["p"] == pytest.approx(6 / 10) and result["b"]["p"] == pytest.approx(8 / 10)


def test_the_difference_by_kind_of_puzzle(store, two_runs):
    result = compare_puzzles(store, *two_runs)

    assert set(result["by_kind"]) == {"no mate", "mate in 3"}
    assert result["by_kind"]["no mate"]["items"] == 3
    assert result["by_kind"]["no mate"]["difference"] == pytest.approx(2 / 4)
    assert result["by_kind"]["mate in 3"]["only_b"] == 1
    assert result["by_kind"]["mate in 3"]["difference"] == pytest.approx(-4 / 6)


def test_a_comparison_of_a_run_with_itself_is_zero(store, two_runs):
    result = compare_puzzles(store, two_runs[0], two_runs[0])

    assert result["whole"]["difference"] == 0 and result["whole"]["p"] == 1.0
    assert "inside the noise" in format_comparison(result)


def test_runs_are_found_by_a_prefix_of_the_id(store, two_runs):
    assert find_run(store, two_runs[0])["run_id"] == two_runs[0]
    assert find_run(store, two_runs[1][:20])["run_id"] == two_runs[1]

    with pytest.raises(ReportError, match="matches 2 runs"):
        find_run(store, "20261018T0")

    with pytest.raises(ReportError, match="no run"):
        find_run(store, "nothing")


def test_runs_on_different_puzzles_or_data_are_refused(store, two_runs, tmp_path):
    items = [puzzles.Item(n, 1, 1, 0, 1, 1.0, (1,)) for n in range(3)]
    blob = store.put_blob(puzzles.items_table(items), "items")
    base = dict(
        protocol="puzzles",
        protocol_version=2,
        grade="verified",
        status="complete",
        started_utc="2026-10-19T00:00:00Z",
        finished_utc="x",
        engine_id=None,
        config="c",
        limits={},
        dataset_id=1,
        seed=None,
        machine_id=None,
        conditions={},
        code_commit=None,
        code_dirty=None,
        summary={},
        items_blob=blob.sha256,
    )
    short = store.add_run(**base)

    with pytest.raises(ReportError, match="different puzzles"):
        compare_puzzles(store, two_runs[0], short)

    other = store.add_dataset("other", "1", "9" * 64, {})
    elsewhere = store.add_run(
        **{**base, "dataset_id": other, "started_utc": "2026-10-19T01:00:00Z"}
    )

    with pytest.raises(ReportError, match="different datasets"):
        compare_puzzles(store, two_runs[0], elsewhere)


def test_a_run_without_per_puzzle_results_is_refused(store, two_runs):
    without_items = store.add_run(
        protocol="puzzles",
        protocol_version=0,
        grade="unverified",
        status="complete",
        started_utc="2026-10-19T00:00:00Z",
        finished_utc="x",
        engine_id=None,
        config=None,
        limits={},
        dataset_id=None,
        seed=None,
        machine_id=None,
        conditions={},
        code_commit=None,
        code_dirty=None,
        summary={},
    )

    with pytest.raises(ReportError, match="no per-puzzle results"):
        compare_puzzles(store, two_runs[0], without_items)


def test_format_comparison_lists_every_kind(store, two_runs):
    text = format_comparison(compare_puzzles(store, *two_runs))

    assert text.splitlines()[0].startswith("A  network+A") and "solved 60.00%" in text
    assert "mate in 3" in text and "no mate" in text and "points" in text


def test_format_run_prints_the_tables_of_a_puzzle_run(store, tmp_path):
    summary = {
        "items": 10,
        "whole": {
            "p": 0.9,
            "low": 0.8,
            "high": 0.95,
            "first_move": 0.95,
            "puzzle_rating": 2400.0,
            "mean_nodes": 1000.0,
        },
        "by_rating_band": {"1000-1199": {"p": 0.99, "n": 5, "mean_nodes": 10.0}},
        "by_solver_moves": {"2": {"p": 0.9, "n": 7, "mean_nodes": 20.0}},
        "by_kind": {"no mate": {"p": 0.9, "n": 7, "mean_nodes": 20.0}},
        "by_theme": {
            "fork": {"p": 0.8, "n": 300, "mean_nodes": 5.0},
            "pin": {"p": 0.7, "n": 300, "mean_nodes": 5.0},
        },
    }
    run = store.add_run(
        protocol="puzzles",
        protocol_version=2,
        grade="verified",
        status="complete",
        started_utc="2026-10-19T00:00:00Z",
        finished_utc="x",
        engine_id=None,
        config="network",
        limits={},
        dataset_id=None,
        seed=None,
        machine_id=None,
        conditions={},
        code_commit=None,
        code_dirty=None,
        summary=summary,
    )
    text = format_run(store, run)

    assert "solved 90.00% (80.00 to 95.00) of 10 puzzles" in text and "puzzle rating 2400" in text
    assert "1000-1199" in text and "weakest themes" in text
    assert text.index("pin") < text.index("fork")


def test_format_run_for_other_protocols_prints_the_summary(store):
    run = store.add_run(
        protocol="rules",
        protocol_version=1,
        grade="verified",
        status="complete",
        started_utc="2026-10-19T00:00:00Z",
        finished_utc="x",
        engine_id=None,
        config=None,
        limits={},
        dataset_id=None,
        seed=None,
        machine_id=None,
        conditions={},
        code_commit=None,
        code_dirty=None,
        summary={"all_correct": True},
    )

    assert '"all_correct": true' in format_run(store, run)


def test_compare_and_report_commands(store, two_runs, tmp_path, tiny_repo, capsys):
    from meristats.cli import main

    repo, _ = tiny_repo
    file = tmp_path / "m.toml"
    file.write_text(
        f'engine_repo = "{repo}"\ndata_dir = "{store.root}"\n'
        f'cache_dir = "{tmp_path / "c"}"\nworkers = 1\n'
    )

    assert main(["--config", str(file), "compare", two_runs[0], two_runs[1]]) == 0
    out = capsys.readouterr().out
    assert "network+A" in out and "network+B" in out and "mate in 3" in out

    assert main(["--config", str(file), "report", two_runs[0][:24]]) == 0
    assert "puzzles v2" in capsys.readouterr().out

    assert main(["--config", str(file), "compare", "nope", two_runs[1]]) == 1
    assert "no run nope" in capsys.readouterr().err


def test_weights_stop_at_the_number_of_items_played(store, tmp_path):
    from meristats import datasets

    sample = tmp_path / "s.tsv"
    write_sample(sample, [(f"p{i}", 1500, 0, 1.0) for i in range(10)])
    dataset_id, _ = datasets.register_file(store, "d", "1", sample, {})
    run = {"dataset_id": dataset_id, "run_id": "r"}

    assert len(report._puzzle_weights(store, run, 4)) == 4
    assert len(report._puzzle_weights(store, run, 50)) == 10


def test_a_run_without_a_dataset_cannot_give_weights(store):
    with pytest.raises(ReportError, match="has no dataset"):
        report._puzzle_weights(store, {"dataset_id": None, "run_id": "r"}, 5)


def test_a_saved_comparison_is_a_derived_run(config, store, two_runs, framework_repo):
    run_id, result = report.save_comparison(
        config, store, two_runs[0], two_runs[1], framework=framework_repo
    )
    row = store.get_run(run_id)

    assert row["protocol"] == "compare" and row["grade"] == "verified"
    assert row["config"] == "network+A against network+B"
    assert json.loads(row["limits_json"]) == {"a": two_runs[0], "b": two_runs[1]}
    assert json.loads(row["summary_json"])["whole"]["only_a"] == result["whole"]["only_a"] == 1
    assert row["dataset_id"] is not None and row["code_commit"] is not None


def test_a_comparison_of_an_unverified_run_is_unverified(config, store, two_runs, framework_repo):
    import json as _json

    unverified = store.add_run(
        protocol="puzzles",
        protocol_version=2,
        grade="unverified",
        status="complete",
        started_utc="2026-10-20T00:00:00Z",
        finished_utc="x",
        engine_id=None,
        config="c",
        limits={},
        dataset_id=store.get_run(two_runs[0])["dataset_id"],
        seed=None,
        machine_id=None,
        conditions={},
        code_commit=None,
        code_dirty=None,
        summary={},
        items_blob=store.get_run(two_runs[0])["items_blob"],
    )
    run_id, _ = report.save_comparison(
        config, store, two_runs[0], unverified, framework=framework_repo
    )

    assert store.get_run(run_id)["grade"] == "unverified"
    assert _json.loads(store.get_run(run_id)["limits_json"])["b"] == unverified


def test_a_dirty_framework_stops_a_saved_comparison(config, store, two_runs, framework_repo):
    from meristats.preflight import PreflightError

    (framework_repo / "file").write_text("changed")

    with pytest.raises(PreflightError, match="framework clean"):
        report.save_comparison(config, store, two_runs[0], two_runs[1], framework=framework_repo)

    run_id, _ = report.save_comparison(
        config, store, two_runs[0], two_runs[1], allow_dirty=True, framework=framework_repo
    )
    assert store.get_run(run_id)["grade"] == "unverified"


def test_compare_pins_each_comparison_with_the_config_of_its_run(
    store, two_runs, tmp_path, tiny_repo, capsys
):
    from meristats.cli import main

    repo, _ = tiny_repo
    file = tmp_path / "m.toml"
    file.write_text(
        f'engine_repo = "{repo}"\ndata_dir = "{store.root}"\n'
        f'cache_dir = "{tmp_path / "c"}"\nworkers = 1\n'
    )
    command = ["--config", str(file), "compare", two_runs[0], two_runs[1], two_runs[1]]

    assert main([*command, "--pin", "techniques", "--allow-dirty"]) == 0
    out = capsys.readouterr().out
    pins = store.pins()

    assert out.count("stored as") == 2 and out.count("is on techniques as network+B") == 2
    assert [(p["page"], p["label"]) for p in pins] == [("techniques", "network+B")]
    assert store.get_run(pins[0]["run_id"])["protocol"] == "compare"
