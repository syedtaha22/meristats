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

from meristats.build import BuiltEngine
from meristats.protocols import rules as protocol

PERFT = (
    "start\trnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\t20,400,8902\n"
    "other\t8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1\t14,191,2812\n"
)


def position(fen):
    return f"position fen {fen}"


def fake_engine_row(store, binary):
    engine_id = store.add_engine("meridian", "oracle", "c" * 40, "d" * 64, {}, 7, "Oracle 1")
    return BuiltEngine(engine_id, "c" * 40, binary, "d" * 64, 7, "Oracle 1", {})


@pytest.fixture
def perft_file(tmp_path):
    path = tmp_path / "perft.txt"
    path.write_text("# reference\n" + PERFT)
    return path


def perft_answers(wrong=()):
    fens = {
        "start": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "other": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    }
    counts = {"start": (20, 400, 8902), "other": (14, 191, 2812)}
    answers = {}

    for name, fen in fens.items():
        for depth, nodes in enumerate(counts[name], 1):
            answers[f"perft {position(fen)} {depth}"] = nodes + (1 if (name, depth) in wrong else 0)

    return answers


def test_every_depth_is_counted_and_compared(config, store, oracle, perft_file, framework_repo):
    engine = fake_engine_row(store, oracle(perft_answers()))
    run_id, summary = protocol.run(
        config,
        store,
        engine,
        perft_file,
        repetitions=3,
        framework=framework_repo,
        progress=lambda t: None,
    )
    rows = protocol.pq.read_table(
        protocol.io.BytesIO(store.get_blob(store.get_run(run_id)["items_blob"]))
    ).to_pydict()

    assert summary["all_correct"] and summary["counts"] == 6 and summary["wrong"] == []
    assert rows["depth"] == [1, 2, 3, 1, 2, 3] and all(rows["correct"])
    assert len(rows["seconds"][2]) == 3 and len(rows["seconds"][0]) == 1
    assert summary["speed"]["start"]["depth"] == 3 and summary["speed"]["start"]["nodes"] == 8902
    assert summary["speed"]["start"]["nodes_per_second"] > 0


def test_a_wrong_count_is_recorded_and_does_not_stop_the_run(
    config, store, oracle, perft_file, framework_repo
):
    engine = fake_engine_row(store, oracle(perft_answers(wrong={("start", 2)})))
    run_id, summary = protocol.run(
        config,
        store,
        engine,
        perft_file,
        repetitions=1,
        framework=framework_repo,
        progress=lambda t: None,
    )

    assert not summary["all_correct"] and summary["counts"] == 6
    assert summary["wrong"] == ["start depth 2: 401 not 400"]
    assert store.get_run(run_id)["status"] == "complete"


def test_depths_above_the_node_limit_are_skipped(config, store, oracle, perft_file, framework_repo):
    engine = fake_engine_row(store, oracle(perft_answers()))
    run_id, summary = protocol.run(
        config,
        store,
        engine,
        perft_file,
        max_nodes=500,
        repetitions=2,
        framework=framework_repo,
        progress=lambda t: None,
    )

    assert summary["counts"] == 4
    assert summary["speed"]["start"]["depth"] == 2
    assert json.loads(store.get_run(run_id)["limits_json"])["max_nodes"] == 500


def test_a_position_without_a_countable_depth_has_no_speed(
    config, store, oracle, perft_file, framework_repo
):
    engine = fake_engine_row(store, oracle(perft_answers()))
    _, summary = protocol.run(
        config,
        store,
        engine,
        perft_file,
        max_nodes=10,
        repetitions=2,
        framework=framework_repo,
        progress=lambda t: None,
    )

    assert summary["counts"] == 0 and summary["speed"] == {}
