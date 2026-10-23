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

import hashlib
import json
from pathlib import Path
import stat
import sys

import pytest

from meristats import nnue
from meristats.build import BuiltEngine
from meristats.preflight import PreflightError
from meristats.protocols import train as protocol
from test_training import write_data

pytest.importorskip("torch")

ENGINE = """
import os, sys
def say(text):
    sys.stdout.write(text + "\\n"); sys.stdout.flush()
for line in sys.stdin:
    words = line.split()
    if not words:
        continue
    if words[0] == "uci":
        say("id name Fake 1")
        say("option name EvalFile type string default <empty>")
        say("uciok")
    elif words[0] == "isready":
        say("readyok")
    elif words[0] == "setoption" and "EvalFile" in words:
        if os.environ.get("FAKE_REFUSE"):
            say("info string cannot read the network")
        else:
            say("info string evaluation: network " + words[-1])
    elif words[0] == "go":
        say("info depth 3 score cp 10 nodes 5 time 1 pv e2e4")
        say("bestmove e2e4")
    elif words[0] == "quit":
        break
"""


@pytest.fixture
def engine(tmp_path, store):
    program = tmp_path / "fake-engine"
    program.write_text(f"#!{sys.executable}\n" + ENGINE)
    program.chmod(program.stat().st_mode | stat.S_IEXEC)
    row = store.add_engine("meridian", "fake", "a" * 40, "1" * 64, {}, 7, "Fake 1")
    return BuiltEngine(row, "a" * 40, program, "1" * 64, 7, "Fake 1", {})


def train(config, store, engine, framework_repo, tmp_path, **settings):
    write_data(tmp_path / "d.txt", count=1500)
    return protocol.run(
        config,
        store,
        engine,
        tmp_path / "d.txt",
        epochs=2,
        batch=64,
        lr=0.01,
        device="cpu",
        workers=2,
        framework=framework_repo,
        progress=lambda text: None,
        **settings,
    )


def test_a_run_stores_the_network_the_log_and_the_numbers(
    config, store, engine, framework_repo, tmp_path
):
    out = tmp_path / "out" / "net.nnue"
    source = tmp_path / "net.inc"
    run_id, summary = train(config, store, engine, framework_repo, tmp_path, out=out, source=source)
    row = store.get_run(run_id)
    roles = dict(
        (r["role"], r["sha256"])
        for r in store.db.execute("SELECT role, sha256 FROM run_blobs WHERE run_id = ?", (run_id,))
    )
    limits = json.loads(row["limits_json"])

    assert row["protocol"] == "train" and row["grade"] == "verified" and row["seed"] == 1
    assert set(roles) == {"network", "tool-log"}
    assert store.get_blob(roles["network"]) == out.read_bytes()
    assert nnue.read_network(out).hidden == 128
    assert source.read_text().startswith("// A network written by meristats")
    assert summary["network"]["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert summary["positions"]["kept"] == 1500 and summary["positions"]["held_out"] > 0
    assert len(summary["epochs"]) == 2 and summary["best_epoch"] in (1, 2)
    assert summary["seconds"]["training"] > 0 and summary["device"] == "cpu"
    assert summary["quantized"]["positions"] > 0 and summary["torch"]
    assert limits["data_sha256"] == hashlib.sha256((tmp_path / "d.txt").read_bytes()).hexdigest()
    assert limits["epochs"] == 2 and limits["batch"] == 64 and limits["init_sha256"] is None
    assert b"epoch" in store.get_blob(roles["tool-log"])


def test_the_default_place_of_the_network_is_the_cache_of_the_run(
    config, store, engine, framework_repo, tmp_path
):
    _, summary = train(config, store, engine, framework_repo, tmp_path)

    assert str(config.cache_dir / "train") in summary["network"]["path"]
    assert nnue.read_network(summary["network"]["path"]).hidden == 128


def test_an_engine_that_refuses_the_network_stops_the_run(
    config, store, engine, framework_repo, tmp_path, monkeypatch
):
    monkeypatch.setenv("FAKE_REFUSE", "1")

    with pytest.raises(PreflightError, match="engine loads the network"):
        train(config, store, engine, framework_repo, tmp_path)

    assert store.runs() == []


def test_a_start_network_is_recorded_by_its_hash(config, store, engine, framework_repo, tmp_path):
    first, _ = train(config, store, engine, framework_repo, tmp_path, out=tmp_path / "a.nnue")
    run_id, _ = train(
        config,
        store,
        engine,
        framework_repo,
        tmp_path,
        out=tmp_path / "b.nnue",
        init=tmp_path / "a.nnue",
    )
    limits = json.loads(store.get_run(run_id)["limits_json"])

    assert limits["init_sha256"] == hashlib.sha256((tmp_path / "a.nnue").read_bytes()).hexdigest()
    assert first != run_id


def test_the_training_runs_in_the_python_that_the_settings_name(
    config, store, engine, framework_repo, tmp_path
):
    from dataclasses import replace

    named = replace(config, trainer_python=Path(sys.executable))
    _, summary = train(named, store, engine, framework_repo, tmp_path)

    assert summary["positions"]["kept"] == 1500


def test_a_trainer_that_fails_is_an_error_with_its_last_lines(
    config, store, engine, framework_repo, tmp_path
):
    from dataclasses import replace

    broken = tmp_path / "python"
    broken.write_text("#!/bin/sh\necho 'no torch here' >&2\nexit 3\n")
    broken.chmod(0o755)
    named = replace(config, trainer_python=broken)

    with pytest.raises(protocol.TrainingError, match="no torch here"):
        train(named, store, engine, framework_repo, tmp_path)
