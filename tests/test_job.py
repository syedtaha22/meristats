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
import shutil

import pytest

from meristats.build import build_engine
from meristats.protocols import job
from meristats.protocols.job import JobError

pytestmark = pytest.mark.skipif(
    not (shutil.which("cmake") and shutil.which("g++")), reason="needs cmake and g++"
)


@pytest.fixture
def setup(tmp_path, tool_repo, store, framework_repo):
    repo, commit, tools = tool_repo
    sha = commit(77)
    from meristats.config import Config

    config = Config(
        engine_repo=repo, data_dir=tmp_path / "data", cache_dir=tmp_path / "cache", workers=2
    )
    engine = build_engine(config, store, sha)
    return config, engine, tools, framework_repo


def test_a_tool_is_built_against_the_engine_library_and_run(setup, store, tmp_path):
    config, engine, tools, framework = setup
    data = tmp_path / "positions.txt"
    data.write_text("first line of data\nsecond\n")
    run_id, summary = job.run(
        config,
        store,
        engine,
        "build_book",
        ["--data", str(data), "--out", "net.nnue"],
        inputs=[data],
        outputs=["net.nnue"],
        tools_dir=tools,
        framework=framework,
        progress=lambda t: None,
    )
    row = store.get_run(run_id)

    assert (row["protocol"], row["status"], row["grade"]) == ("job", "complete", "verified")
    assert summary["exit_code"] == 0 and summary["tool"] == "build_book"
    assert summary["engine_commit"] == engine.commit_sha
    output = summary["outputs"][0]
    assert output["name"] == "net.nnue" and output["role"] == "network"
    assert store.get_blob(output["sha256"]) == b"core 42 first line [first line of data]\n"
    assert summary["inputs"][0]["bytes"] == len(data.read_bytes())
    assert len(summary["inputs"][0]["sha256"]) == 64 and len(summary["tool_sha256"]) == 64
    roles = {
        r["role"]
        for r in store.db.execute("SELECT role FROM run_blobs WHERE run_id = ?", (run_id,))
    }
    assert roles == {"network", "tool-log"}
    assert json.loads(row["limits_json"])["args"] == ["--data", str(data), "--out", "net.nnue"]


def test_the_same_tool_gives_the_same_output_at_the_same_commit(setup, store, tmp_path):
    config, engine, tools, framework = setup
    data = tmp_path / "d.txt"
    data.write_text("x\n")
    first = job.run(
        config,
        store,
        engine,
        "build_book",
        ["--data", str(data)],
        inputs=[data],
        outputs=["out.txt"],
        tools_dir=tools,
        framework=framework,
        progress=lambda t: None,
    )[1]
    second = job.run(
        config,
        store,
        engine,
        "build_book",
        ["--data", str(data)],
        inputs=[data],
        outputs=["out.txt"],
        tools_dir=tools,
        framework=framework,
        progress=lambda t: None,
    )[1]

    assert first["outputs"][0]["sha256"] == second["outputs"][0]["sha256"]
    assert first["tool_sha256"] == second["tool_sha256"]


def test_a_tool_that_fails_is_stored_as_a_failed_run(setup, store):
    config, engine, tools, framework = setup
    run_id, summary = job.run(
        config,
        store,
        engine,
        "build_book",
        ["--exit", "3"],
        tools_dir=tools,
        framework=framework,
        progress=lambda t: None,
    )
    row = store.get_run(run_id)

    assert row["status"] == "failed" and summary["exit_code"] == 3
    assert row["notes"] == "exit 3"


def test_a_missing_output_is_a_failed_run_and_names_the_file(setup, store):
    config, engine, tools, framework = setup
    run_id, summary = job.run(
        config,
        store,
        engine,
        "build_book",
        ["--out", "none"],
        outputs=["net.nnue"],
        tools_dir=tools,
        framework=framework,
        progress=lambda t: None,
    )

    assert store.get_run(run_id)["status"] == "failed"
    assert (
        summary["missing"] == ["net.nnue"] and "missing net.nnue" in store.get_run(run_id)["notes"]
    )


def test_a_tool_that_does_not_exist_is_refused(setup, store):
    config, engine, tools, framework = setup

    with pytest.raises(JobError, match="not a tool"):
        job.build_tool(config, engine, "rm", tools)

    with pytest.raises(JobError, match="no source"):
        job.build_tool(config, engine, "train_policy", tools)


def test_a_tool_that_does_not_compile_is_a_build_error(setup, store):
    from meristats.build import BuildError

    config, engine, tools, framework = setup
    (tools / "train_policy.cpp").write_text("int main( {")

    with pytest.raises(BuildError, match="train_policy"):
        job.build_tool(config, engine, "train_policy", tools)


def test_an_input_that_is_not_a_file_is_refused(setup, store, tmp_path):
    config, engine, tools, framework = setup

    with pytest.raises(JobError, match="is not a file"):
        job.run(
            config,
            store,
            engine,
            "build_book",
            [],
            inputs=[tmp_path / "missing"],
            tools_dir=tools,
            framework=framework,
            progress=lambda t: None,
        )


def test_roles_follow_the_file_suffix():
    assert job._role("net.nnue") == "network" and job._role("tuned.params") == "params"
    assert job._role("positions.txt") == "output"


def test_the_real_tool_sources_and_the_cmake_hook_are_in_the_repository():
    for tool in job.TOOLS:
        assert (job.REPOSITORY_TOOLS / f"{tool}.cpp").exists(), tool

    assert (job.REPOSITORY_TOOLS / "common.hpp").exists()

    for tool in job.TOOLS:
        text = (job.REPOSITORY_TOOLS / f"{tool}.cpp").read_text()
        assert '#include "common.hpp"' in text and "../bench" not in text, tool
    assert job.CMAKE_INCLUDE.exists()
