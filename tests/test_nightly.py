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
import time
from pathlib import Path

import numpy as np
import pytest

from meristats import lichess, nightly
from meristats.build import BuiltEngine
from meristats.config import Config
from meristats import nnue


def network(seed=1):
    rng = np.random.default_rng(seed)
    return nnue.Network(
        rng.integers(-50, 50, (nnue.INPUTS, nnue.HIDDEN), dtype=np.int16),
        rng.integers(-50, 50, nnue.HIDDEN, dtype=np.int16),
        rng.integers(-50, 50, (nnue.BUCKETS, 2, nnue.HIDDEN), dtype=np.int16),
        rng.integers(-500, 500, nnue.BUCKETS, dtype=np.int32),
    )


class Fakes:
    """
    The stages of a night, replaced by functions that record that they were called.
    """

    def __init__(self, tmp_path, verdict="H1", games=None):
        self.calls = []
        self.verdict = verdict
        self.games = games or []
        self.tmp_path = tmp_path

    def functions(self):
        return {
            "build": self.build,
            "selfplay": self.selfplay,
            "sync": self.sync,
            "reanalyse": self.reanalyse,
            "train": self.train,
            "match": self.match,
        }

    def build(self, config, store, ref):
        self.calls.append("build")
        return BuiltEngine(1, "c" * 40, Path("/bin/true"), "1" * 64, 7, "Meridian 1.1.0", {})

    def selfplay(self, config, store, engine, out, **settings):
        self.calls.append("selfplay")
        self.selfplay_settings = settings
        out.write_text("fen\t1\t0\t10\nfen\t1\t0\t12\n")
        return "run-selfplay", {"games": 3, "positions": 2}

    def sync(self, data_dir, user):
        self.calls.append("sync")
        path = lichess.games_file(data_dir, user)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(g) + "\n" for g in self.games))
        return self.games

    def reanalyse(self, config, store, engine, todo, user, out, **settings):
        self.calls.append("reanalyse")
        out.mkdir(parents=True, exist_ok=True)
        (out / "positions.txt").write_text("fen\t1\t1\t5\n")
        (out / "mistakes.txt").write_text("fen\t1\t1\t5\n")
        from meristats import experience

        experience.write(out / "experience.bin", {1: (2, 3, 14)})
        return "run-reanalyse", {
            "analysed": len(todo),
            "analysed_ids": [g["id"] for g in todo],
            "mistakes": 1,
            "positions": 1,
            "mean_loss_cp": 12.0,
            "games": len(todo),
        }

    def train(self, config, store, engine, data, **settings):
        self.calls.append("train")
        self.train_settings = settings
        nnue.write_network(network(2), settings["out"])
        return "run-train", {
            "best_held": 0.01,
            "best_epoch": 3,
            "seconds": {"training": 1.0},
            "network": {"sha256": "a" * 64},
        }

    def match(self, config, store, a, b, book, limit, **settings):
        self.calls.append("match")
        self.match_settings = settings
        self.match_limit = limit
        return "run-match", {
            "games": 400,
            "wins": 150,
            "draws": 150,
            "losses": 100,
            "score": 0.55,
            "elo": 35.0,
            "elo_low": 10.0,
            "elo_high": 60.0,
            "sprt": {"verdict": self.verdict, "agree": True},
            "timed_out": False,
        }


@pytest.fixture
def setup(tmp_path, monkeypatch):
    config = Config(
        engine_repo=tmp_path / "engine",
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        workers=2,
        lichess_user="me",
    )

    def committed(repo, sha, destination):
        nnue.write_network(network(1), destination)
        return destination

    monkeypatch.setattr(nightly, "_committed_network", committed)
    return config


def settings(**changes):
    return nightly.Settings(deadline=time.time() + 6 * 3600, night="n1", **changes)


def games(*ids):
    return [{"id": i, "moves": "e4", "players": {}} for i in ids]


def test_a_night_that_passes_names_the_candidate_and_the_deployed_network(setup, tmp_path):
    fakes = Fakes(tmp_path, "H1", games("g1", "g2"))
    summary = nightly.run(
        setup,
        None,
        settings(),
        book=tmp_path / "book.pgn",
        functions=fakes.functions(),
        progress=lambda t: None,
    )
    assert summary["status"] == "passed"
    assert summary["candidate"].endswith("candidate.nnue")
    assert summary["deployed"].endswith("deployed.nnue")
    match = summary["stages"]["match"]
    assert (match["wins"], match["draws"], match["losses"], match["nodes"]) == (
        150,
        150,
        100,
        50000,
    )
    assert fakes.calls == ["build", "selfplay", "sync", "reanalyse", "train", "match"]
    assert (Path(setup.data_dir) / "nightly" / "n1" / "summary.json").exists()


def test_a_network_that_does_not_win_is_rejected_and_nothing_is_committed(setup, tmp_path):
    fakes = Fakes(tmp_path, "H0")
    summary = nightly.run(
        setup,
        None,
        settings(),
        book=tmp_path / "b.pgn",
        functions=fakes.functions(),
        progress=lambda t: None,
    )
    assert summary["status"] == "rejected"
    assert "candidate" not in summary
    assert summary["stages"]["match"]["verdict"] == "H0"


def test_a_stage_that_fails_ends_the_night_with_its_error(setup, tmp_path):
    fakes = Fakes(tmp_path)

    def broken(*args, **kwargs):
        raise RuntimeError("the trainer died")

    functions = fakes.functions() | {"train": broken}
    summary = nightly.run(
        setup,
        None,
        settings(),
        book=tmp_path / "b.pgn",
        functions=functions,
        progress=lambda t: None,
    )
    assert summary["status"] == "failed"
    assert "the trainer died" in summary["error"]
    assert "candidate" not in summary


def test_a_night_past_its_deadline_stops_before_the_next_stage(setup, tmp_path):
    fakes = Fakes(tmp_path)
    late = nightly.Settings(deadline=time.time() + 30, night="n2")
    summary = nightly.run(
        setup,
        None,
        late,
        book=tmp_path / "b.pgn",
        functions=fakes.functions(),
        progress=lambda t: None,
    )
    assert summary["status"] == "late"
    assert summary["stopped_before"] == "self-play"
    assert fakes.calls == ["build"]


def test_the_match_gets_the_candidate_at_a_fixed_number_of_nodes_with_a_time_limit(setup, tmp_path):
    fakes = Fakes(tmp_path, "H0")
    nightly.run(
        setup,
        None,
        settings(match_nodes=12345, match_minutes=10),
        book=tmp_path / "b.pgn",
        functions=fakes.functions(),
        progress=lambda t: None,
    )
    assert fakes.match_limit.kind == "nodes" and fakes.match_limit.value == 12345
    assert fakes.match_settings["a_eval_file"].name == "candidate.nnue"
    assert fakes.match_settings["timeout"] <= 600


def test_training_starts_from_the_deployed_network_and_the_data_has_the_night_and_the_anchor(
    setup, tmp_path
):
    anchor = tmp_path / "anchor.txt"
    anchor.write_text("fen\t1\t0\t1\n" * 5)
    config = Config(**{**setup.__dict__, "anchor_positions": anchor})
    fakes = Fakes(tmp_path, "H0", games("g1"))
    summary = nightly.run(
        config,
        None,
        settings(),
        book=tmp_path / "b.pgn",
        functions=fakes.functions(),
        progress=lambda t: None,
    )
    assert fakes.train_settings["init"].name == "deployed.nnue"
    counts = summary["stages"]["data"]
    assert counts["night"] == 3 and counts["anchor"] == 5 and counts["mistakes"] == 3
    assert lichess.analysed(setup.data_dir, "me") == {"g1"}


def test_the_experience_file_grows_with_the_analysis_of_the_games(setup, tmp_path):
    fakes = Fakes(tmp_path, "H0", games("g1"))
    summary = nightly.run(
        setup,
        None,
        settings(),
        book=tmp_path / "b.pgn",
        functions=fakes.functions(),
        progress=lambda t: None,
    )
    path = Path(setup.data_dir) / "experience" / "experience.bin"
    assert path.exists()
    assert summary["stages"]["experience"] == {"entries": 1, "added": 1}


def test_the_summary_is_written_as_text_for_the_email(setup, tmp_path):
    fakes = Fakes(tmp_path, "H1", games("g1"))
    summary = nightly.run(
        setup,
        None,
        settings(),
        book=tmp_path / "b.pgn",
        functions=fakes.functions(),
        progress=lambda t: None,
    )
    text = nightly.format_summary(summary)
    assert "passed" in text and "Self-play: 3 games" in text and "Match: 400 games" in text
    assert (
        json.loads((Path(setup.data_dir) / "nightly" / "n1" / "summary.json").read_text())["night"]
        == "n1"
    )


def test_the_mix_of_the_training_data_and_the_density_of_self_play_follow_the_settings(
    setup, tmp_path
):
    anchor = tmp_path / "anchor.txt"
    anchor.write_text("fen\t1\t0\t1\n" * 50)
    config = Config(**{**setup.__dict__, "anchor_positions": anchor})
    fakes = Fakes(tmp_path, "H0", games("g1"))
    summary = nightly.run(
        config,
        None,
        settings(anchor_lines=7, selfplay_skip=2, mistake_repeat=2),
        book=tmp_path / "b.pgn",
        functions=fakes.functions(),
        progress=lambda t: None,
    )

    assert fakes.selfplay_settings["skip"] == 2
    assert summary["stages"]["data"]["anchor"] == 7
    assert summary["stages"]["data"]["mistakes"] == 2
