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

import json
import time
from pathlib import Path

import numpy as np
import pytest

from meristats import cycle, nnue, replay
from meristats.build import BuiltEngine
from meristats.config import Config


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
    The steps of a cycle, replaced by functions that record that they were called.
    """

    def __init__(self, verdict="H1", chunk_positions=3):
        self.calls = []
        self.verdict = verdict
        self.chunk_positions = chunk_positions
        self.seeds = []
        self.interrupt_at = None

    def functions(self):
        return {
            "build": self.build,
            "selfplay": self.selfplay,
            "train": self.train,
            "match": self.match,
        }

    def build(self, config, store, ref):
        self.calls.append("build")
        return BuiltEngine(1, "c" * 40, Path("/bin/true"), "1" * 64, 7, "Meridian 1.1.0", {})

    def selfplay(self, config, store, engine, out, **settings):
        self.calls.append("selfplay")
        self.seeds.append(settings["seed"])
        self.selfplay_settings = settings

        if self.interrupt_at == len(self.seeds):
            raise KeyboardInterrupt

        out.write_text("".join(f"fen{len(self.seeds)}-{i}\t1\t{i}\t10\n" for i in range(
            self.chunk_positions)))
        return "run-selfplay", {"games": 2, "positions": self.chunk_positions}

    def train(self, config, store, engine, data, **settings):
        self.calls.append("train")
        self.train_settings = settings
        self.train_data = Path(data).read_text()
        nnue.write_network(network(2), settings["out"])
        return "run-train", {
            "best_held": 0.01,
            "start_held": 0.012,
            "best_epoch": 3,
            "epochs": [{}, {}, {}, {}],
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
def config(tmp_path, monkeypatch):
    def committed(repo, sha, destination):
        nnue.write_network(network(1), destination)
        return destination

    monkeypatch.setattr(cycle, "committed_network", committed)
    return Config(
        engine_repo=tmp_path / "engine",
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        workers=2,
    )


def settings(**changes):
    return cycle.Settings(test_after=5, **changes)


def go(config, fakes, tmp_path, state=None, **changes):
    state = state or cycle.State(test_after=5)
    summary = cycle.run(
        config,
        None,
        settings(**changes),
        state,
        book=tmp_path / "book.pgn",
        functions=fakes.functions(),
        progress=lambda text: None,
    )
    return summary, state


def test_data_is_generated_in_chunks_until_enough_for_a_test(config, tmp_path):
    fakes = Fakes()
    summary, state = go(config, fakes, tmp_path)

    assert fakes.calls == ["build", "selfplay", "selfplay", "train", "match"]
    assert summary["stages"]["selfplay"]["positions"] == 6
    assert summary["stages"]["selfplay"]["chunks"] == 2
    assert summary["stages"]["data"] == {"new": 6, "history": 0}
    assert state.chunks == 2


def test_a_cycle_that_passes_names_the_candidate_and_the_deployed_network(config, tmp_path):
    summary, state = go(config, Fakes("H1"), tmp_path)

    assert summary["status"] == "passed"
    assert summary["candidate"].endswith("candidate.nnue")
    assert summary["deployed"].endswith("deployed.nnue")
    match = summary["stages"]["match"]
    assert (match["wins"], match["draws"], match["losses"], match["nodes"]) == (150, 150, 100, 50000)
    assert state.test_after == 5 and state.cycles == 1
    folder = Path(config.data_dir) / "cycles" / summary["cycle"]
    assert json.loads((folder / "summary.json").read_text())["status"] == "passed"


def test_a_cycle_that_is_rejected_waits_for_more_data_the_next_time(config, tmp_path):
    summary, state = go(config, Fakes("H0"), tmp_path, growth=2.0)

    assert summary["status"] == "rejected"
    assert "candidate" not in summary
    assert state.test_after == 10
    assert state.last_test != ""


def test_the_state_is_saved_so_that_the_next_cycle_starts_where_this_one_ended(config, tmp_path):
    _, state = go(config, Fakes("H0"), tmp_path)
    loaded = cycle.State.load(config.data_dir, 999)

    assert loaded == state and loaded.last_test != ""


def test_the_next_cycle_trains_on_the_new_data_and_a_sample_of_the_old(config, tmp_path):
    fakes = Fakes("H0")
    _, state = go(config, fakes, tmp_path)
    summary, _ = go(config, fakes, tmp_path, state=state)

    assert state.test_after == 10
    assert summary["stages"]["data"]["new"] == 9
    assert summary["stages"]["data"]["history"] == 6
    assert state.cycles == 2


def test_every_chunk_has_its_own_seed_and_so_has_every_cycle(config, tmp_path):
    fakes = Fakes("H0")
    _, state = go(config, fakes, tmp_path)
    go(config, fakes, tmp_path, state=state)

    assert len(fakes.seeds) == 5 and len(set(fakes.seeds)) == 5


def test_a_step_that_fails_ends_the_cycle_and_keeps_the_data_for_the_next_try(config, tmp_path):
    fakes = Fakes()

    def broken(*args, **kwargs):
        raise RuntimeError("the trainer died")

    functions = fakes.functions() | {"train": broken}
    state = cycle.State(test_after=5)
    summary = cycle.run(config, None, settings(), state, book=tmp_path / "b.pgn",
                        functions=functions, progress=lambda t: None)

    assert summary["status"] == "failed"
    assert "the trainer died" in summary["error"]
    assert state.last_test == "" and state.test_after == 5
    again, _ = go(config, fakes, tmp_path, state=state)
    assert fakes.calls.count("selfplay") == 2
    assert again["status"] == "passed"


def test_a_stop_during_the_data_keeps_the_chunks_that_were_finished(config, tmp_path):
    fakes = Fakes()
    fakes.interrupt_at = 2
    state = cycle.State(test_after=50)

    with pytest.raises(KeyboardInterrupt):
        cycle.run(config, None, settings(), state, book=tmp_path / "b.pgn",
                  functions=fakes.functions(), progress=lambda t: None)

    buffer = replay.Buffer(Path(config.data_dir) / "replay")
    assert len(buffer.files(("selfplay",))) == 1
    assert cycle.State.load(config.data_dir, 1).chunks == 1


def test_a_deadline_ends_the_data_and_the_cycle_tests_what_it_has(config, tmp_path):
    fakes = Fakes()
    state = cycle.State(test_after=1000)
    summary = cycle.run(
        config, None, cycle.Settings(test_after=1000, deadline=time.time() + 100), state,
        book=tmp_path / "b.pgn", functions=fakes.functions(), progress=lambda t: None,
    )

    assert summary["status"] == "passed"
    assert fakes.selfplay_settings["max_seconds"] <= 100
    assert fakes.calls.count("selfplay") >= 1


def test_a_deadline_that_has_passed_with_no_data_is_a_failure_and_not_a_test(config, tmp_path):
    fakes = Fakes()
    summary = cycle.run(
        config, None, cycle.Settings(test_after=1000, deadline=time.time() + 10),
        cycle.State(test_after=1000), book=tmp_path / "b.pgn", functions=fakes.functions(),
        progress=lambda t: None,
    )

    assert summary["status"] == "failed" and "no new positions" in summary["error"]
    assert fakes.calls == ["build"]


def test_the_eval_has_no_time_limit_unless_one_is_set(config, tmp_path):
    fakes = Fakes("H0")
    go(config, fakes, tmp_path)
    assert fakes.match_settings["timeout"] is None
    go(config, fakes, tmp_path, match_minutes=10)
    assert fakes.match_settings["timeout"] == 600


def test_the_match_gets_the_candidate_at_a_fixed_number_of_nodes(config, tmp_path):
    fakes = Fakes("H0")
    go(config, fakes, tmp_path, match_nodes=12345)

    assert fakes.match_limit.kind == "nodes" and fakes.match_limit.value == 12345
    assert fakes.match_settings["a_eval_file"].name == "candidate.nnue"


def test_training_starts_from_the_deployed_network_and_stops_early(config, tmp_path):
    fakes = Fakes("H0")
    go(config, fakes, tmp_path, epochs=9, patience=2)

    assert fakes.train_settings["init"].name == "deployed.nnue"
    assert fakes.train_settings["epochs"] == 9 and fakes.train_settings["patience"] == 2


def test_the_anchor_and_leela_positions_join_the_training_data(config, tmp_path):
    anchor = tmp_path / "anchor.txt"
    anchor.write_text("fen\t1\t0\t1\n" * 50)
    leela = tmp_path / "leela.txt"
    leela.write_text("fen\t1\t0\t1\n" * 40)
    config = Config(**{**config.__dict__, "anchor_positions": anchor, "leela_positions": leela})
    summary, _ = go(config, Fakes("H0"), tmp_path, anchor_lines=7, leela_lines=13)

    assert summary["stages"]["data"] == {"new": 6, "history": 0, "anchor": 7, "leela": 13}


def test_the_summary_is_written_as_text(config, tmp_path):
    summary, _ = go(config, Fakes("H1"), tmp_path)
    text = cycle.format_summary(summary)

    assert "passed" in text and "Data: 6 positions" in text and "Match: 400 games" in text


def test_generating_adds_chunks_to_the_buffer_until_the_deadline(config, tmp_path):
    clock = [1000.0]
    fakes = Fakes()
    original = fakes.selfplay

    def selfplay(*args, **kwargs):
        clock[0] += 600
        return original(*args, **kwargs)

    fakes.selfplay = selfplay
    state = cycle.State(test_after=5)
    summary = cycle.generate(
        config, None, cycle.Settings(deadline=2000), state, book=tmp_path / "b.pgn",
        functions=fakes.functions(), progress=lambda t: None, clock=lambda: clock[0],
    )
    buffer = replay.Buffer(Path(config.data_dir) / "replay")

    assert fakes.calls == ["build", "selfplay", "selfplay"]
    assert fakes.selfplay_settings["max_seconds"] == 400
    assert len(buffer.files(("selfplay",))) == 2
    assert summary["chunks"] == 2 and summary["positions"] == 6 and summary["games"] == 4
    assert cycle.State.load(config.data_dir, 1).chunks == 2


def test_generating_gives_every_chunk_its_own_seed_and_the_book(config, tmp_path):
    clock = [0.0]
    fakes = Fakes()
    original = fakes.selfplay

    def selfplay(*args, **kwargs):
        clock[0] += 600
        return original(*args, **kwargs)

    fakes.selfplay = selfplay
    state = cycle.State(test_after=5, chunks=7)
    cycle.generate(
        config, None, cycle.Settings(deadline=1500), state, book=tmp_path / "b.pgn",
        functions=fakes.functions(), progress=lambda t: None, clock=lambda: clock[0],
    )

    assert fakes.seeds == [8, 9, 10]
    assert fakes.selfplay_settings["book"] == tmp_path / "b.pgn"


def test_a_stop_while_generating_keeps_the_chunks_and_returns_the_counts(config, tmp_path):
    fakes = Fakes()
    fakes.interrupt_at = 3
    messages = []
    summary = cycle.generate(
        config, None, cycle.Settings(), cycle.State(test_after=5), book=tmp_path / "b.pgn",
        functions=fakes.functions(), progress=messages.append,
    )
    buffer = replay.Buffer(Path(config.data_dir) / "replay")

    assert summary["chunks"] == 2 and summary["positions"] == 6
    assert "stopped" in messages
    assert len(buffer.files(("selfplay",))) == 2


def test_a_chunk_that_ctrl_c_ended_is_kept_and_ends_the_generation(config, tmp_path):
    fakes = Fakes()
    original = fakes.selfplay

    def selfplay(*args, **kwargs):
        run_id, played = original(*args, **kwargs)
        return run_id, {**played, "interrupted": len(fakes.seeds) == 2}

    fakes.selfplay = selfplay
    summary = cycle.generate(
        config, None, cycle.Settings(), cycle.State(test_after=5), book=tmp_path / "b.pgn",
        functions=fakes.functions(), progress=lambda t: None,
    )
    buffer = replay.Buffer(Path(config.data_dir) / "replay")

    assert fakes.calls.count("selfplay") == 2
    assert len(buffer.files(("selfplay",))) == 2
    assert summary["chunks"] == 1
