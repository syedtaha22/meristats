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

import shutil

import pytest

from conftest import run_git
from meristats.build import (
    BuildError,
    build_engine,
    export_source,
    file_sha256,
    resolve_commit,
    run_bench,
)

pytestmark = pytest.mark.skipif(
    not (shutil.which("cmake") and shutil.which("g++")), reason="needs cmake and g++"
)


def test_resolve_commit_gives_the_full_sha_for_refs(tiny_repo):
    repo, commit = tiny_repo
    sha = commit(100)

    assert resolve_commit(repo, "main") == sha
    assert resolve_commit(repo, sha[:10]) == sha
    assert resolve_commit(repo, "HEAD") == sha


def test_resolve_commit_refuses_what_is_not_a_commit(tiny_repo):
    repo, commit = tiny_repo
    commit(1)

    with pytest.raises(BuildError, match="not a commit"):
        resolve_commit(repo, "no-such-branch")


def test_export_source_writes_the_committed_files_only(tiny_repo, tmp_path):
    repo, commit = tiny_repo
    sha = commit(5)
    (repo / "untracked.txt").write_text("x")
    (repo / "main.cpp").write_text("changed but not committed")
    export_source(repo, sha, tmp_path / "out")

    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == [
        "CMakeLists.txt",
        "main.cpp",
    ]
    assert "NODES" not in (tmp_path / "out" / "main.cpp").read_text()
    assert "changed but not committed" not in (tmp_path / "out" / "main.cpp").read_text()


def test_build_records_commit_binary_hash_and_bench_nodes(config, tiny_repo, store):
    repo, commit = tiny_repo
    sha = commit(777, version="2.0")
    built = build_engine(config, store, "main")

    assert built.commit_sha == sha
    assert built.bench_nodes == 777
    assert built.version_text == "Tiny 2.0"
    assert built.binary_sha256 == file_sha256(built.binary)
    assert built.build["build_type"] == "Release"
    assert "g++" in built.build["compiler"] or "c++" in built.build["compiler"]
    row = store.find_engines(sha)[0]
    assert row["bench_nodes"] == 777 and row["binary_sha256"] == built.binary_sha256


def test_the_working_tree_never_enters_the_build(config, tiny_repo, store):
    repo, commit = tiny_repo
    commit(111)
    (repo / "main.cpp").write_text("this does not compile")

    assert build_engine(config, store, "HEAD").bench_nodes == 111


def test_building_a_commit_twice_gives_one_engine_row(config, tiny_repo, store):
    repo, commit = tiny_repo
    sha = commit(5)
    first = build_engine(config, store, sha)
    second = build_engine(config, store, sha)

    assert first.engine_id == second.engine_id
    assert len(store.find_engines(sha)) == 1


def test_two_commits_give_two_engines_with_their_own_counts(config, tiny_repo, store):
    repo, commit = tiny_repo
    one, two = commit(10), commit(20)

    assert build_engine(config, store, one).bench_nodes == 10
    assert build_engine(config, store, two).bench_nodes == 20
    assert build_engine(config, store, one).engine_id != build_engine(config, store, two).engine_id


def test_a_different_count_for_the_same_commit_is_refused(config, tiny_repo, store):
    repo, commit = tiny_repo
    sha = commit(50)
    store.add_engine("meridian", "meridian", sha, "0" * 64, {}, 999, "Tiny 1.0")

    with pytest.raises(BuildError, match="same source must give the same count"):
        build_engine(config, store, sha)


def test_a_source_that_does_not_build_is_reported_with_its_output(config, tiny_repo, store):
    repo, commit = tiny_repo
    (repo / "main.cpp").write_text("int main( {")
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-q", "-m", "broken")

    with pytest.raises(BuildError, match="failed"):
        build_engine(config, store, "HEAD")


def test_a_commit_without_the_target_is_reported(config, tiny_repo, store):
    repo, commit = tiny_repo
    commit(1)

    with pytest.raises(BuildError):
        build_engine(config, store, "HEAD", target="nothing")


def test_run_bench_reports_none_for_a_program_without_bench(tmp_path):
    script = tmp_path / "prog"
    script.write_text("#!/bin/sh\necho nothing here\n")
    script.chmod(0o755)

    assert run_bench(script) is None
    assert run_bench(tmp_path / "missing") is None


def test_export_of_an_unknown_commit_fails_with_the_git_message(tiny_repo, tmp_path):
    repo, commit = tiny_repo
    commit(1)

    with pytest.raises(BuildError, match="git archive"):
        export_source(repo, "f" * 40, tmp_path / "out")


def test_exporting_over_an_existing_folder_replaces_it(tiny_repo, tmp_path):
    repo, commit = tiny_repo
    sha = commit(1)
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale.txt").write_text("old")
    export_source(repo, sha, out)

    assert not (out / "stale.txt").exists() and (out / "main.cpp").exists()


def test_a_build_without_the_program_is_reported(config, tiny_repo, store):
    repo, commit = tiny_repo
    (repo / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\nproject(tiny CXX)\n"
        "add_custom_target(meridian COMMAND echo nothing)\n"
    )
    commit(1)

    with pytest.raises(BuildError, match="produced no meridian"):
        build_engine(config, store, "HEAD")


def test_helper_functions_give_unavailable_when_they_cannot_read(tmp_path):
    from meristats import build

    assert build._cache_value(tmp_path / "missing", "KEY") == "unavailable"
    assert build._first_line(["/nonexistent/program"]) == "unavailable"
    (tmp_path / "cache").write_text("A:STRING=1\nB:PATH=two\n")
    assert build._cache_value(tmp_path / "cache", "B") == "two"
    assert build._cache_value(tmp_path / "cache", "C") == "unavailable"
