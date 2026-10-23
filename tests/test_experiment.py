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

import pytest

from meristats import report
from meristats.build import BuiltEngine
from meristats.protocols import experiment as protocol
from meristats.protocols import match, puzzles, search_nodes
from meristats.protocols.experiment import Settings

GOOD = {"difference": 0.001, "low": -0.001, "high": 0.003, "p": 0.4, "only_a": 5, "only_b": 3}
WORSE = {"difference": -0.01, "low": -0.012, "high": -0.008, "p": 1e-9, "only_a": 5, "only_b": 90}
NOISE = {"difference": -0.001, "low": -0.003, "high": 0.001, "p": 0.3, "only_a": 5, "only_b": 9}


def summary(verdict="H1", score=0.55):
    return {"score": score, "sprt": {"verdict": verdict}}


def test_a_change_that_wins_its_match_is_kept():
    keep, reasons = protocol.decide(GOOD, summary("H1", 0.56), 1.3)

    assert keep and "not worse" in reasons[0] and "SPRT H1" in reasons[1]
    assert "1.300 of the base" in reasons[2]


def test_a_change_that_loses_its_match_is_dropped():
    assert not protocol.decide(GOOD, summary("H0", 0.45), 0.5)[0]


def test_no_decision_needs_a_score_of_half_and_fewer_nodes():
    assert protocol.decide(GOOD, summary("continue", 0.51), 0.9)[0]
    assert not protocol.decide(GOOD, summary("continue", 0.51), 0.96)[0]
    assert not protocol.decide(GOOD, summary("continue", 0.49), 0.5)[0]
    assert not protocol.decide(GOOD, summary("continue", 0.6), None)[0]
    assert protocol.decide(GOOD, summary("continue", 0.5), 0.95)[0]


def test_significantly_worse_puzzles_drop_a_change_even_when_the_match_is_won():
    keep, reasons = protocol.decide(WORSE, summary("H1", 0.6), 0.5)

    assert not keep and "worse" in reasons[0] and "not worse" not in reasons[0]


def test_a_small_loss_on_puzzles_inside_the_noise_does_not_drop_a_change():
    assert protocol.decide(NOISE, summary("H1", 0.6), 1.0)[0]


def test_nodes_ratio_uses_the_deepest_depth():
    table = {
        "network": {"5": {"mean_nodes": 1000.0}, "7": {"mean_nodes": 8000.0}},
        "network+x": {"5": {"mean_nodes": 900.0}, "7": {"mean_nodes": 6000.0}},
    }

    assert protocol.nodes_ratio(table, "network+x", "network") == pytest.approx(0.75)
    assert protocol.nodes_ratio(table, "network+y", "network") is None
    assert protocol.nodes_ratio({"network": {}, "network+x": {}}, "network+x", "network") is None
    assert (
        protocol.nodes_ratio(
            {"a": {"7": {"mean_nodes": 1}}, "b": {"5": {"mean_nodes": 1}}}, "a", "b"
        )
        is None
    )
    assert (
        protocol.nodes_ratio(
            {"a": {"7": {"mean_nodes": 1}}, "b": {"7": {"mean_nodes": 0}}}, "a", "b"
        )
        is None
    )


class Recorder:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.results.pop(0)


@pytest.fixture
def stubs(monkeypatch, config, store, framework_repo):
    """
    Replaces the sub-protocols with recorders, and makes real runs for the experiment to name.
    """
    engine_id = store.add_engine("meridian", "m", "a" * 40, "b" * 64, {}, 5, "M")
    engine = BuiltEngine(engine_id, "a" * 40, config.engine_repo / "x", "b" * 64, 5, "M", {})
    counter = {"n": 0}

    def fake_run(label):
        counter["n"] += 1
        return store.add_run(
            protocol=label,
            protocol_version=1,
            grade="verified",
            status="complete",
            started_utc=f"2026-10-18T0{counter['n']}:00:00Z",
            finished_utc="x",
            engine_id=engine_id,
            config=label,
            limits={},
            dataset_id=None,
            seed=None,
            machine_id=None,
            conditions={},
            code_commit=None,
            code_dirty=None,
            summary={},
        )

    return monkeypatch, engine, fake_run, framework_repo


def install(stubs, *, comparison=GOOD, verdict="H1", score=0.56, nodes=0.8, puzzle_runs=None):
    monkeypatch, engine, fake_run, framework = stubs
    puzzle_calls = Recorder([(fake_run("puzzles"), {}) for _ in range(20)])
    monkeypatch.setattr(puzzles, "run", puzzle_calls)
    monkeypatch.setattr(
        report, "save_comparison", lambda *a, **k: (fake_run("compare"), {"whole": comparison})
    )
    nodes_summary = {
        "network": {"7": {"mean_nodes": 1000.0}},
        "network+razoring": {"7": {"mean_nodes": 1000.0 * nodes}},
    }
    nodes_calls = Recorder([(fake_run("search-nodes"), nodes_summary) for _ in range(20)])
    monkeypatch.setattr(search_nodes, "run", nodes_calls)
    match_calls = Recorder([(fake_run("match"), summary(verdict, score)) for _ in range(20)])
    monkeypatch.setattr(match, "run", match_calls)
    return puzzle_calls, nodes_calls, match_calls


def test_an_experiment_runs_the_parts_and_names_them(stubs, config, store, tmp_path):
    monkeypatch, engine, fake_run, framework = stubs
    puzzle_calls, nodes_calls, match_calls = install(stubs)
    run_id, result = protocol.run(
        config,
        store,
        engine,
        "network",
        "razoring",
        tmp_path / "suite",
        tmp_path / "book",
        tmp_path / "puzzles.tsv",
        Settings(puzzle_items=500, depths=(5, 7), rounds=50),
        framework=framework,
        progress=lambda t: None,
    )
    row = store.get_run(run_id)

    assert result["keep"] and result["new"] == "network+razoring" and result["nodes_ratio"] == 0.8
    assert row["protocol"] == "experiment" and row["config"] == "network+razoring"
    assert row["grade"] == "verified"
    assert len(puzzle_calls.calls) == 2
    assert puzzle_calls.calls[0][1]["engine_config"] == "network"
    assert puzzle_calls.calls[1][1]["engine_config"] == "network+razoring"
    assert puzzle_calls.calls[0][1]["items"] == 500
    assert nodes_calls.calls[0][0][4] == ["network", "network+razoring"]
    assert nodes_calls.calls[0][0][5] == [5, 7]
    ((args, kwargs),) = match_calls.calls
    assert kwargs["a_config"] == "network+razoring" and kwargs["b_config"] == "network"
    assert kwargs["rounds"] == 50 and args[5] == match.Limit("st", 0.05)
    assert set(result["runs"]) == {"base_puzzles", "new_puzzles", "comparison", "nodes", "match"}
    assert all(store.get_run(r) is not None for r in result["runs"].values())


def test_a_base_run_can_be_reused(stubs, config, store, tmp_path):
    monkeypatch, engine, fake_run, framework = stubs
    puzzle_calls, _, _ = install(stubs)
    existing = fake_run("puzzles")
    _, result = protocol.run(
        config,
        store,
        engine,
        "network",
        "razoring",
        tmp_path / "s",
        tmp_path / "b",
        tmp_path / "p",
        base_puzzle_run=existing,
        framework=framework,
        progress=lambda t: None,
    )

    assert len(puzzle_calls.calls) == 1 and result["runs"]["base_puzzles"] == existing


def test_an_unverified_part_makes_the_experiment_unverified(stubs, config, store, tmp_path):
    monkeypatch, engine, fake_run, framework = stubs
    install(stubs)
    (framework / "file").write_text("changed")
    run_id, _ = protocol.run(
        config,
        store,
        engine,
        "network",
        "razoring",
        tmp_path / "s",
        tmp_path / "b",
        tmp_path / "p",
        allow_dirty=True,
        framework=framework,
        progress=lambda t: None,
    )

    assert store.get_run(run_id)["grade"] == "unverified"


def test_a_list_of_changes_grows_the_base_with_the_kept_ones(stubs, config, store, tmp_path):
    monkeypatch, engine, fake_run, framework = stubs
    puzzle_calls, nodes_calls, match_calls = install(stubs)
    nodes = {
        "network": {"7": {"mean_nodes": 1000.0}},
        "network+a": {"7": {"mean_nodes": 800.0}},
        "network+a+b": {"7": {"mean_nodes": 800.0}},
        "network+a+b+c": {"7": {"mean_nodes": 800.0}},
    }
    monkeypatch.setattr(search_nodes, "run", Recorder([(fake_run("search-nodes"), nodes)] * 3))
    monkeypatch.setattr(
        match,
        "run",
        Recorder(
            [
                (fake_run("match"), summary("H1", 0.6)),  # a is kept
                (fake_run("match"), summary("H0", 0.4)),  # b is dropped
                (fake_run("match"), summary("H1", 0.6)),  # !c is kept but only tested
            ]
        ),
    )
    base, results = protocol.run_changes(
        config,
        store,
        engine,
        "network",
        ["a", "b", "!c"],
        tmp_path / "s",
        tmp_path / "b",
        tmp_path / "p",
        framework=framework,
        progress=lambda t: None,
    )

    assert base == "network+a"
    assert [r[1]["new"] for r in results] == ["network+a", "network+a+b", "network+a+c"]
    assert [r[1]["keep"] for r in results] == [True, False, True]
    # the base puzzles are played once, and the next changes reuse the run of the kept change
    configs = [c[1]["engine_config"] for c in puzzle_calls.calls]
    assert configs == ["network", "network+a", "network+a+b", "network+a+c"]


def test_settings_have_the_documented_defaults():
    settings = Settings()

    assert settings.puzzle_items == 200_000 and settings.puzzle_nodes == 20_000
    assert settings.depths == (5, 7) and settings.st == 0.05
    assert settings.sprt == match.Sprt(0.0, 10.0, 0.05, 0.05)
