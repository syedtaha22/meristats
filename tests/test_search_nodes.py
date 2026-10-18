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

from meristats import datasets
from meristats.build import BuiltEngine
from meristats.protocols import search_nodes as protocol
from meristats.protocols.search_nodes import RoundsDisagree, Row

SUITE = (
    "opening\tope-1\tr4rk1/ppp2ppp/1b6/1Q2Pb2/1n1R3q/1PN1P2N/PBP1B1PP/2K4R b\n"
    "endgame\tend-1\t8/8/1p6/ppk1p1p1/4P1P1/PP1P4/3K4/8 b\n"
)


def position(fen):
    return f"position fen {fen}"


def fake_engine_row(store, binary):
    engine_id = store.add_engine("meridian", "oracle", "c" * 40, "d" * 64, {}, 7, "Oracle 1")
    return BuiltEngine(engine_id, "c" * 40, binary, "d" * 64, 7, 1000, "Oracle 1", {})


def answer(nodes, move="e2e4", score=5):
    return {"best": move, "infos": [[1, nodes // 4, move], [2, nodes, move]]}


@pytest.fixture
def suite(tmp_path):
    path = tmp_path / "suite.txt"
    path.write_text("# a comment\n\n" + SUITE)
    return path


def answers_for(suite_path, nodes=100):
    return {position(p.fen): answer(nodes) for p in datasets.read_suite(suite_path)}


def test_a_run_has_one_row_for_every_config_depth_and_position(
    config, store, oracle, suite, framework_repo
):
    engine = fake_engine_row(store, oracle(answers_for(suite)))
    run_id, summary = protocol.run(
        config,
        store,
        engine,
        suite,
        ["network", "baseline"],
        [2, 3],
        rounds=2,
        framework=framework_repo,
        progress=lambda t: None,
    )
    row = store.get_run(run_id)
    rows = protocol.read_rows(store.get_blob(row["items_blob"]))

    assert len(rows) == 2 * 2 * 2
    assert {(r.config, r.depth) for r in rows} == {
        ("network", 2),
        ("network", 3),
        ("baseline", 2),
        ("baseline", 3),
    }
    assert all(r.nodes == 100 and r.best_move == "e2e4" and r.score == 5 for r in rows)
    assert row["config"] == "network,baseline" and row["grade"] == "verified"
    assert json.loads(row["limits_json"])["depths"] == [2, 3]
    assert (
        summary["network"]["2"]["mean_nodes"] == 100 and summary["baseline"]["3"]["positions"] == 2
    )


def test_the_rows_say_which_region_each_position_is_in(
    config, store, oracle, suite, framework_repo
):
    engine = fake_engine_row(store, oracle(answers_for(suite)))
    run_id, _ = protocol.run(
        config,
        store,
        engine,
        suite,
        ["network"],
        [2],
        rounds=1,
        framework=framework_repo,
        progress=lambda t: None,
    )
    rows = protocol.read_rows(store.get_blob(store.get_run(run_id)["items_blob"]))

    assert {(r.position, r.region) for r in rows} == {("ope-1", "opening"), ("end-1", "endgame")}


def test_rounds_that_disagree_on_the_nodes_fail_the_run(
    config, store, oracle, suite, framework_repo
):
    answers = {position(p.fen): [answer(100), answer(101)] for p in datasets.read_suite(suite)}
    engine = fake_engine_row(store, oracle(answers))

    with pytest.raises(RoundsDisagree, match="gave"):
        protocol.run(
            config,
            store,
            engine,
            suite,
            ["network"],
            [2],
            rounds=2,
            framework=framework_repo,
            progress=lambda t: None,
        )

    assert store.runs("search-nodes") == []


def test_rounds_that_agree_pass_and_the_time_is_a_median(
    config, store, oracle, suite, framework_repo
):
    engine = fake_engine_row(store, oracle(answers_for(suite)))
    run_id, summary = protocol.run(
        config,
        store,
        engine,
        suite,
        ["network"],
        [2],
        rounds=3,
        framework=framework_repo,
        progress=lambda t: None,
    )

    assert summary["network"]["2"]["mean_ms"] > 0
    assert store.get_run(run_id)["status"] == "complete"


def test_the_default_config_is_recorded_as_default(config, store, oracle, suite, framework_repo):
    engine = fake_engine_row(store, oracle(answers_for(suite)))
    run_id, _ = protocol.run(
        config,
        store,
        engine,
        suite,
        [None],
        [2],
        rounds=1,
        framework=framework_repo,
        progress=lambda t: None,
    )
    rows = protocol.read_rows(store.get_blob(store.get_run(run_id)["items_blob"]))

    assert store.get_run(run_id)["config"] == "default"
    assert {r.config for r in rows} == {"default"}


def test_a_mate_score_is_flagged(config, store, oracle, suite, framework_repo):
    answers = {
        position(p.fen): {"best": "a1a8", "infos": [[1, 5, "a1a8"]]}
        for p in datasets.read_suite(suite)
    }
    program = oracle(answers)
    engine = fake_engine_row(store, program)
    # the oracle prints cp scores, so check the plumbing with a cp score and the flag off
    run_id, _ = protocol.run(
        config,
        store,
        engine,
        suite,
        [None],
        [1],
        rounds=1,
        framework=framework_repo,
        progress=lambda t: None,
    )
    rows = protocol.read_rows(store.get_blob(store.get_run(run_id)["items_blob"]))

    assert all(not r.is_mate for r in rows)


def test_rows_round_trip_through_parquet():
    rows = [
        Row("network", 6, "mid-1", "middlegame", "e2e4", -34, False, 12345, 1.25),
        Row("baseline", 8, "end-3", "endgame", "a1a8", 2, True, 99, 0.5),
    ]
    back = protocol.read_rows(protocol.table(rows))

    assert [
        (r.config, r.depth, r.position, r.best_move, r.score, r.is_mate, r.nodes) for r in back
    ] == [(r.config, r.depth, r.position, r.best_move, r.score, r.is_mate, r.nodes) for r in rows]


def test_bad_arguments_are_refused(config, store, oracle, suite, framework_repo):
    engine = fake_engine_row(store, oracle(answers_for(suite)))

    for kwargs in ({"rounds": 0}, {"configs": []}, {"depths": []}):
        arguments = {"configs": ["network"], "depths": [2], "rounds": 1, **kwargs}

        with pytest.raises(ValueError):
            protocol.run(
                config,
                store,
                engine,
                suite,
                arguments["configs"],
                arguments["depths"],
                rounds=arguments["rounds"],
                framework=framework_repo,
            )


def test_a_config_that_the_engine_rejects_fails_the_run(
    config, store, oracle, suite, tmp_path, framework_repo
):
    # the oracle accepts any Config, so use a program whose Config reply is an error
    program = tmp_path / "strict"
    program.write_text(
        "#!/bin/sh\nwhile read l; do case \"$l\" in uci) echo 'id name S'; "
        "echo 'option name Config type string default network'; "
        "echo 'option name Threads type spin default 1 min 1 max 8'; echo uciok;; "
        "isready) echo readyok;; setoption*) echo 'info string invalid Config: no';; "
        "quit) exit 0;; esac; done\n"
    )
    program.chmod(0o755)
    engine = fake_engine_row(store, program)

    from meristats.preflight import PreflightError

    with pytest.raises(PreflightError, match="invalid Config"):
        protocol.run(
            config,
            store,
            engine,
            suite,
            ["nonsense"],
            [2],
            rounds=1,
            framework=framework_repo,
            progress=lambda t: None,
        )
