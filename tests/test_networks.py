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
import textwrap

import pytest

from meristats.build import BuiltEngine
from meristats.config import Config
from meristats.protocols import networks as protocol

SPEC = textwrap.dedent("""
    engine = "HEAD"
    data = "/data/train.txt"
    rounds = 100
    nodes = 5000
    seed = 4

    [[candidate]]
    name = "base"
    init = "deployed"
    epochs = 3

    [[candidate]]
    name = "w256"
    hidden = 256
    lr = 0.001
""")


def write(tmp_path, text=SPEC):
    path = tmp_path / "spec.toml"
    path.write_text(text)
    return path


def test_a_spec_gives_the_data_the_limit_and_the_candidates(tmp_path):
    spec = protocol.load_spec(write(tmp_path))

    assert spec.limit.kind == "nodes" and spec.limit.value == 5000 and spec.rounds == 100
    assert [c.name for c in spec.candidates] == ["base", "w256"]
    assert (spec.candidates[1].hidden, spec.candidates[1].lr) == (256, 0.001)
    assert spec.candidates[0].init == "deployed" and spec.candidates[0].epochs == 3


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("nodes = 5000", "needs one of nodes and st"),
        ('name = "w256"', "two candidates are called"),
        ("hidden = 256", "multiple of 16"),
        ("lr = 0.001", "options that do not exist"),
    ],
)
def test_a_wrong_spec_is_refused_with_the_reason(tmp_path, change, message):
    text = SPEC
    if change == "nodes = 5000":
        text = text.replace(change, "nodes = 5000\nst = 0.05")
    elif change == 'name = "w256"':
        text = text.replace(change, 'name = "base"')
    elif change == "hidden = 256":
        text = text.replace(change, "hidden = 250")
    else:
        text = text.replace(change, "colour = 1")

    with pytest.raises(protocol.NetworksError, match=message):
        protocol.load_spec(write(tmp_path, text))


def test_the_deployed_network_can_start_only_a_network_of_its_own_width(tmp_path):
    text = SPEC.replace('init = "deployed"', 'init = "deployed"\n    hidden = 256')

    with pytest.raises(protocol.NetworksError, match="has 128 values"):
        protocol.load_spec(write(tmp_path, text))


def test_a_network_has_the_numbers_of_its_layers():
    assert protocol.parameters(128) == 1536 * 128 + 128 + 8 * 2 * 128 + 8
    assert protocol.parameters(256) > 2 * protocol.parameters(128) - 3000


class Fakes:
    def __init__(self, tmp_path):
        self.calls = []
        self.tmp_path = tmp_path

    def functions(self):
        return {
            "build": self.build,
            "train": self.train,
            "match": self.match,
            "speed": self.speed,
            "deployed": self.deployed,
        }

    def build(self, config, store, ref, defines=None):
        self.calls.append(("build", defines))
        width = (defines or {}).get("CHESS_NNUE_HIDDEN", "128")
        return BuiltEngine(1, "a" * 40, self.tmp_path / f"engine{width}", "b" * 64, 5, "T", {})

    def deployed(self, repo, sha, destination):
        self.calls.append(("deployed", None))
        destination.write_bytes(b"d")
        return destination

    def train(self, config, store, engine, data, **settings):
        self.calls.append(("train", settings["hidden"], settings["init"] is not None, str(data)))
        return "train-run", {
            "best_held": 0.0123,
            "network": {"path": str(settings["out"])},
        }

    def speed(self, binary, network, suite):
        return 1000 if network is None else (500 if "256" in binary.name else 900)

    def match(self, config, store, a, b, book, limit, **settings):
        self.calls.append(("match", settings["rounds"], limit.kind, settings["sprt"]))
        return "match-run", {"games": 200, "elo": 12.0, "elo_low": 3.0, "elo_high": 21.0}


@pytest.fixture
def setup(tmp_path):
    config = Config(
        engine_repo=tmp_path / "e", data_dir=tmp_path / "d", cache_dir=tmp_path / "c", workers=2
    )
    return config, protocol.load_spec(write(tmp_path))


def run(config, spec, fakes, tmp_path):
    return protocol.run(
        config,
        None,
        spec,
        suite=tmp_path / "suite.txt",
        book=tmp_path / "book.pgn",
        functions=fakes.functions(),
        progress=lambda text: None,
    )


def test_every_candidate_is_built_for_its_width_trained_measured_and_played(setup, tmp_path):
    config, spec = setup
    fakes = Fakes(tmp_path)
    rows = run(config, spec, fakes, tmp_path)

    assert [r["name"] for r in rows] == ["base", "w256"]
    assert ("build", {"CHESS_NNUE_HIDDEN": "256"}) in fakes.calls
    assert ("train", 256, False, "/data/train.txt") in fakes.calls
    assert ("train", 128, True, "/data/train.txt") in fakes.calls
    assert fakes.calls.count(("match", 100, "nodes", None)) == 2
    wide = rows[1]
    assert (wide["hidden"], wide["speed"], wide["elo"]) == (256, 0.5, 12.0)
    assert wide["parameters"] == protocol.parameters(256)
    assert json.loads((spec.folder / "results.json").read_text())[1]["name"] == "w256"


def test_a_candidate_that_is_in_the_results_is_not_run_again(setup, tmp_path):
    config, spec = setup
    run(config, spec, Fakes(tmp_path), tmp_path)
    again = Fakes(tmp_path)
    rows = run(config, spec, again, tmp_path)

    assert len(rows) == 2 and not [c for c in again.calls if c[0] in ("train", "match")]


def test_the_table_shows_the_loss_the_speed_and_the_elo_of_each_candidate(setup, tmp_path):
    config, spec = setup
    table = protocol.format_table(run(config, spec, Fakes(tmp_path), tmp_path))

    assert "w256" in table and "0.50x" in table and "+12.0 (+3.0 to +21.0)" in table
    assert table.splitlines()[0].split() == ["name", "hidden", "loss", "speed", "Elo"]


def test_the_speed_is_the_nodes_of_the_searches_over_their_time(fake_engine, tmp_path):
    program, _ = fake_engine()
    suite = tmp_path / "suite.txt"
    suite.write_text(
        "# a suite\n"
        "opening\ta\tr4rk1/ppp2ppp/1b6/1Q2Pb2/1n1R3q/1PN1P2N/PBP1B1PP/2K4R b\n"
        "opening\tb\tr2qkb1r/pb3ppp/1p3n2/2p5/Pn6/BP1pP3/3P1PPP/R1NQKBNR b KQkq\n"
    )
    plain = protocol.measure_speed(program, None, suite)
    with_network = protocol.measure_speed(program, tmp_path / "n.nnue", suite)

    assert plain > 0 and with_network > 0
