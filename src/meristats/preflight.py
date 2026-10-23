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
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from meristats import provenance
from meristats.build import BuiltEngine, run_bench


class PreflightError(Exception):
    """
    A check that must pass before a run did not pass.
    """


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str = ""


def require(checks: Iterable[Check]) -> list[Check]:
    """
    Returns the checks, and raises `PreflightError` when any of them failed.
    """
    checks = list(checks)
    failed = [check for check in checks if not check.ok]

    if failed:
        raise PreflightError("; ".join(f"{check.name}: {check.detail}" for check in failed))

    return checks


def framework_clean(root: Path | None, allow_dirty: bool = False):
    """
    Checks that the working tree of the meristats repository has no changes. Returns the check,
    the commit, whether the tree was dirty, and the grade a run gets: `verified`, or `unverified`
    when `allow_dirty` lets a dirty tree through.
    """
    if root is None:
        return (
            Check("framework clean", False, "meristats does not run from a git checkout"),
            None,
            None,
            "unverified",
        )

    commit, dirty = provenance.git_state(root)

    if commit is None:
        return (
            Check("framework clean", False, f"{root} is not a git repository"),
            None,
            None,
            "unverified",
        )

    if dirty and not allow_dirty:
        return (
            Check("framework clean", False, "the working tree has changes"),
            commit,
            True,
            "unverified",
        )

    return (
        Check("framework clean", True, "dirty, allowed" if dirty else "clean"),
        commit,
        dirty,
        "unverified" if dirty else "verified",
    )


def engine_from_sha(engine: BuiltEngine) -> Check:
    ok = bool(engine.commit_sha) and len(engine.commit_sha) == 40
    return Check("engine from a SHA", ok, engine.commit_sha or "no commit")


def fingerprint(engine: BuiltEngine, depth: int = 10) -> Check:
    """
    Runs `bench` again and compares the node count with the one recorded for the build.
    """
    if engine.bench_nodes is None:
        return Check("fingerprint", False, "the engine gives no bench node count")

    nodes = run_bench(engine.binary, depth)
    return Check(
        "fingerprint",
        nodes == engine.bench_nodes,
        f"{nodes} nodes now, {engine.bench_nodes} recorded",
    )


def dataset_hash(path: Path, expected: str) -> Check:
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    actual = digest.hexdigest()
    return Check("dataset hash", actual == expected, f"{path.name}: {actual[:12]}")


def engine_replies(replies: Iterable[str]) -> Check:
    """
    Checks that no reply to a `setoption` command reports an invalid value.
    """
    bad = [reply for reply in replies if reply.lower().startswith("invalid")]
    return Check("engine options", not bad, "; ".join(bad) if bad else "accepted")


def one_thread(options: dict) -> Check:
    threads = str(options.get("Threads", "1"))
    return Check("one thread", threads == "1", f"Threads {threads}")


def move_overhead(options: dict, timed: bool) -> Check:
    """
    A protocol with a time limit must send `MoveOverhead 0`, so that `movetime` is the time
    that the search takes.
    """
    if not timed:
        return Check("move overhead", True, "no time limit")

    value = str(options.get("MoveOverhead", ""))
    return Check("move overhead", value == "0", f"MoveOverhead {value or 'not set'}")
