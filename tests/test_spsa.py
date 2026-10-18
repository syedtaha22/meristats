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
import os
import random
from pathlib import Path

import pytest

from meristats.build import BuiltEngine
from meristats.protocols import match
from meristats.protocols import spsa as protocol


def test_parameters_follow_the_techniques_of_the_base():
    names = lambda base: [p[0] for p in protocol.parameters_of(base)]  # noqa: E731

    assert names("full") == []
    assert names("full+razoring") == ["razor_base", "razor_step"]
    assert names("tuned") == [
        "rfp_margin",
        "futility_base",
        "futility_step",
        "null_margin_step",
        "aspiration_delta",
    ]
    assert names("network") == [
        "rfp_margin",
        "futility_base",
        "futility_step",
        "lmr_scale",
        "null_margin_step",
        "aspiration_delta",
    ]
    assert names("network+reverse_futility=0") == [
        "futility_base",
        "futility_step",
        "lmr_scale",
        "null_margin_step",
        "aspiration_delta",
    ]
    assert "iir_depth" in names("full+iir") and "singular_depth" in names("full+singular")


def test_a_value_in_the_base_is_the_start_value():
    found = {p[0]: p for p in protocol.parameters_of("network+rfp_margin=130")}

    assert found["rfp_margin"][1] == 130 and found["futility_base"][1] == 74


def test_preset_string_and_prefix():
    values = {"rfp_margin": 117.6, "futility_base": 63.4}

    assert protocol.preset_string("network", values) == "network+rfp_margin=118+futility_base=63"
    assert protocol.base_prefix("network+see=0+rfp_margin=130", {"rfp_margin"}) == "network+see=0"
    assert protocol.base_prefix("network", {"rfp_margin"}) == "network"


def test_perturbation_shrinks_and_never_falls_below_one():
    assert protocol.perturbation(15, 1) == 15
    assert protocol.perturbation(15, 100) < protocol.perturbation(15, 10) < 15
    assert protocol.perturbation(1, 1000) == 1.0


def test_perturb_stays_inside_the_bounds_and_is_symmetric():
    parameters = [("a", 100, 20, 90, 110), ("b", 5, 4, 1, 9)]
    theta = {"a": 100.0, "b": 5.0}
    signs, plus, minus = protocol.perturb(theta, parameters, 1, random.Random(1))

    assert set(signs.values()) <= {-1, 1}
    assert 90 <= plus["a"] <= 110 and 90 <= minus["a"] <= 110
    assert 1 <= plus["b"] <= 9 and 1 <= minus["b"] <= 9
    assert plus["b"] - 5 == pytest.approx(5 - minus["b"])


def test_update_moves_toward_the_winner_and_not_at_all_on_a_draw():
    parameters = [("a", 100, 20, 0, 1000)]
    theta = {"a": 100.0}
    signs = {"a": 1}

    assert protocol.update(theta, parameters, signs, 0.5, 1) == {"a": 100.0}
    assert protocol.update(theta, parameters, signs, 0.8, 1)["a"] > 100
    assert protocol.update(theta, parameters, signs, 0.2, 1)["a"] < 100
    assert protocol.update(theta, parameters, {"a": -1}, 0.8, 1)["a"] < 100
    assert protocol.update(theta, parameters, signs, 1.0, 1)["a"] == pytest.approx(
        100 + 3 / (1 + 1 / 30) ** 0.602 * 20
    )


def test_the_trajectory_equals_the_one_of_the_script_that_this_replaces():
    # The numbers come from the algorithm of the earlier spsa script with the same seed and
    # the same scripted scores.
    parameters = [
        ("rfp_margin", 107, 15, 40, 300),
        ("futility_base", 74, 15, 20, 400),
        ("futility_step", 214, 20, 30, 400),
        ("lmr_scale", 225, 20, 120, 400),
    ]
    scores = [0.62, 0.31, 0.5, 0.7, 0.45, 0.55, 0.9, 0.1]
    rng = random.Random(protocol.SEED)
    theta = {name: float(start) for name, start, _, _, _ in parameters}
    history = []

    for k, score in enumerate(scores, 1):
        signs, _, _ = protocol.perturb(theta, parameters, k, rng)
        theta = protocol.update(theta, parameters, signs, score, k)
        history.append({n: round(v, 6) for n, v in theta.items()})

    assert history[0] == {
        "rfp_margin": 117.588904,
        "futility_base": 63.411096,
        "futility_step": 228.118539,
        "lmr_scale": 210.881461,
    }
    assert history[-1] == {
        "rfp_margin": 139.828135,
        "futility_base": 70.326684,
        "futility_step": 189.493579,
        "lmr_scale": 280.645771,
    }


@pytest.fixture
def book(tmp_path):
    path = tmp_path / "book.pgn"
    path.write_text('[Event "?"]\n\n1. e4 e5 *\n')
    return path


def built(store, oracle):
    program = oracle({})
    engine_id = store.add_engine("meridian", "oracle", "a" * 40, "1" * 64, {}, 7, "Oracle 1")
    return BuiltEngine(engine_id, "a" * 40, program, "1" * 64, 7, 1000, "Oracle 1", {})


def test_a_run_stores_every_iteration_and_the_tuned_preset(
    config, store, oracle, book, fake_fastchess, framework_repo
):
    import dataclasses

    fastchess = fake_fastchess(["1-0", "0-1"] * 6)
    run_id, summary = protocol.run(
        dataclasses.replace(config, fastchess=fastchess),
        store,
        built(store, oracle),
        "network",
        book,
        match.Limit("st", 0.02),
        iterations=4,
        pairs=6,
        framework=framework_repo,
        check_compliance=False,
        progress=lambda t: None,
    )
    row = store.get_run(run_id)
    args = Path(os.environ["FAKE_ARGS_FILE"]).read_text()

    assert row["protocol"] == "spsa" and row["grade"] == "verified" and row["seed"] == 7
    assert [t["k"] for t in summary["trajectory"]] == [1, 2, 3, 4]
    assert all(t["score"] == 1.0 for t in summary["trajectory"])
    assert summary["base"] == "network" and summary["iterations"] == 4
    assert summary["tuned"].startswith("network+") and "lmr_scale=" in summary["tuned"]
    assert set(summary["final"]) == {
        "rfp_margin",
        "futility_base",
        "futility_step",
        "lmr_scale",
        "null_margin_step",
        "aspiration_delta",
    }
    assert "option.Config=network+" in args and "-srand 1004" in args
    assert "option.MoveOverhead=0" in args and "-sprt" not in args
    assert json.loads(row["limits_json"])["pairs"] == 6


def test_the_parameters_move_toward_the_side_that_wins(
    config, store, oracle, book, fake_fastchess, framework_repo
):
    import dataclasses

    # A, the side that is moved up, wins every game
    fastchess = fake_fastchess(["1-0", "0-1"] * 6)
    _, up = protocol.run(
        dataclasses.replace(config, fastchess=fastchess),
        store,
        built(store, oracle),
        "full+razoring",
        book,
        match.Limit("depth", 2),
        iterations=1,
        pairs=6,
        framework=framework_repo,
        check_compliance=False,
        progress=lambda t: None,
    )
    fastchess = fake_fastchess(["0-1", "1-0"] * 6)
    _, down = protocol.run(
        dataclasses.replace(config, fastchess=fastchess),
        store,
        built(store, oracle),
        "full+razoring",
        book,
        match.Limit("depth", 2),
        iterations=1,
        pairs=6,
        framework=framework_repo,
        check_compliance=False,
        progress=lambda t: None,
    )

    for name in ("razor_base", "razor_step"):
        start = {"razor_base": 800, "razor_step": 300}[name]
        assert (up["final"][name] - start) == pytest.approx(-(down["final"][name] - start))
        assert up["final"][name] != start


def test_a_base_without_parameters_is_refused(config, store, oracle, book, framework_repo):
    with pytest.raises(ValueError, match="nothing to tune"):
        protocol.run(
            config,
            store,
            built(store, oracle),
            "full",
            book,
            match.Limit("st", 0.02),
            framework=framework_repo,
        )
