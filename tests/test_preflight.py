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

from conftest import run_git
from meristats import preflight
from meristats.build import BuiltEngine
from meristats.preflight import Check, PreflightError, require


def fake_built(tmp_path, nodes_printed, nodes_recorded, sha="a" * 40):
    program = tmp_path / "prog"
    program.write_text(f"#!/bin/sh\necho 'Nodes searched  : {nodes_printed}'\n")
    program.chmod(0o755)
    return BuiltEngine(1, sha, program, "0" * 64, nodes_recorded, "v", {})


def test_require_passes_checks_through_and_raises_for_a_failed_one():
    checks = [Check("a", True), Check("b", True, "fine")]

    assert require(checks) == checks

    with pytest.raises(PreflightError, match="b: broken"):
        require([Check("a", True), Check("b", False, "broken")])


def test_a_clean_framework_repo_is_verified(tmp_path):
    run_git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "f").write_text("x")
    run_git(tmp_path, "add", "-A")
    run_git(tmp_path, "commit", "-q", "-m", "m")
    check, commit, dirty, grade = preflight.framework_clean(tmp_path)

    assert check.ok and dirty is False and grade == "verified" and len(commit) == 40


def test_a_dirty_framework_repo_fails_unless_allowed(tmp_path):
    run_git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "f").write_text("x")
    run_git(tmp_path, "add", "-A")
    run_git(tmp_path, "commit", "-q", "-m", "m")
    (tmp_path / "f").write_text("changed")

    check, _, dirty, grade = preflight.framework_clean(tmp_path)
    assert not check.ok and dirty is True and grade == "unverified"

    check, _, dirty, grade = preflight.framework_clean(tmp_path, allow_dirty=True)
    assert check.ok and dirty is True and grade == "unverified"


def test_an_ignored_file_does_not_make_the_repo_dirty(tmp_path):
    run_git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / ".gitignore").write_text("junk\n")
    run_git(tmp_path, "add", "-A")
    run_git(tmp_path, "commit", "-q", "-m", "m")
    (tmp_path / "junk").write_text("x")

    assert preflight.framework_clean(tmp_path)[0].ok


def test_no_checkout_or_no_repository_is_unverified(tmp_path):
    assert not preflight.framework_clean(None)[0].ok
    assert not preflight.framework_clean(tmp_path)[0].ok
    assert preflight.framework_clean(None)[3] == "unverified"


def test_engine_from_a_sha(tmp_path):
    assert preflight.engine_from_sha(fake_built(tmp_path, 1, 1)).ok
    assert not preflight.engine_from_sha(fake_built(tmp_path, 1, 1, sha="")).ok
    assert not preflight.engine_from_sha(fake_built(tmp_path, 1, 1, sha="abc")).ok


def test_fingerprint_compares_a_fresh_bench_with_the_record(tmp_path):
    assert preflight.fingerprint(fake_built(tmp_path, 640, 640)).ok
    assert not preflight.fingerprint(fake_built(tmp_path, 641, 640)).ok
    assert not preflight.fingerprint(fake_built(tmp_path, 640, None)).ok


def test_dataset_hash(tmp_path):
    import hashlib

    path = tmp_path / "d.tsv"
    path.write_bytes(b"rows")
    good = hashlib.sha256(b"rows").hexdigest()

    assert preflight.dataset_hash(path, good).ok
    assert not preflight.dataset_hash(path, "0" * 64).ok


def test_engine_replies_flag_an_invalid_option():
    assert preflight.engine_replies(["Config: network", "evaluation: built-in"]).ok
    assert not preflight.engine_replies(["Config: x", "invalid Config: nope"]).ok
    assert preflight.engine_replies([]).ok


def test_one_thread_and_move_overhead():
    assert preflight.one_thread({"Threads": 1}).ok and preflight.one_thread({}).ok
    assert not preflight.one_thread({"Threads": 2}).ok
    assert preflight.move_overhead({"MoveOverhead": 0}, timed=True).ok
    assert not preflight.move_overhead({"MoveOverhead": 30}, timed=True).ok
    assert not preflight.move_overhead({}, timed=True).ok
    assert preflight.move_overhead({"MoveOverhead": 30}, timed=False).ok
