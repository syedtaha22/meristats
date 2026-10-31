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

"""
Commits the network of a cycle that passed, pushes it, and keeps the pull request up to date.

The steps:

1. The ledger is brought up to date with the state of its pull requests.
2. When every earlier network is in `main` (or its pull request was closed), the branch is reset
   to `origin/main`. Otherwise the branch stays as it is and the network goes on top of it.
3. The network is written into `src/nnue_net.inc`, the minor version of the engine is raised in
   `include/version.hpp`, the engine is built and its tests run, and the commit is made by the
   author of the target, as author and committer.
4. The branch is pushed with the token of the GitHub App, and a row is added to the ledger.
5. The pull request is opened when none is open, and its description is written from the ledger.

The commit is returned. A failure before the push leaves the clone as it was. When only the pull
request fails, `PullRequestError` carries the commit. Nothing is merged: that is left to the owner.
"""

import os
import re
import subprocess
from pathlib import Path

from meristats import nnue
from meristats.release import github_app, pull_request
from meristats.release.common import ReleaseError, auth_environment, git  # noqa: F401

SOURCE = "src/nnue_net.inc"
VERSION_FILE = "include/version.hpp"
VERSION = re.compile(r'(kEngineVersion = ")(\d+)\.(\d+)\.(\d+)(")')


class PullRequestError(ReleaseError):
    """
    The network is committed and pushed, and only the pull request could not be opened or updated.
    `commit` is the commit of the network.
    """

    def __init__(self, commit, reason):
        super().__init__(f"the network is committed ({commit[:9]}), the pull request failed: {reason}")
        self.commit = commit


def bump_version(engine):
    """
    Raises the minor version of the engine, `kEngineVersion` in include/version.hpp, and returns
    the new version. A network that passed its test changes the moves of the engine, which
    VERSIONING.md of the engine counts as a minor change, and the version changes in the commit
    of that change.
    """
    path = Path(engine) / VERSION_FILE
    text = path.read_text() if path.exists() else ""
    found = VERSION.search(text)

    if found is None:
        raise ReleaseError(f"{path} has no kEngineVersion of the form X.Y.Z")

    new = f"{found.group(2)}.{int(found.group(3)) + 1}.0"
    path.write_text(VERSION.sub(lambda m: f'{m.group(1)}{new}{m.group(5)}', text, count=1))
    return new


def build_and_test(engine, build):
    for command in (
        ["cmake", "-S", str(engine), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"],
        ["cmake", "--build", str(build), "-j", str(os.cpu_count() or 1)],
        ["ctest", "--test-dir", str(build), "--output-on-failure"],
    ):
        done = subprocess.run(command, capture_output=True, text=True)

        if done.returncode != 0:
            raise ReleaseError(f"{' '.join(command[:2])} failed: {(done.stdout + done.stderr)[-500:]}")


def promote(summary, target, api, auth=None, run_tests=True):
    """
    Runs the steps for one cycle that passed. `summary` is the content of its summary file,
    `target` says where (see settings.py), `api` makes the calls of the GitHub API, and `auth` is
    the environment for the commands of git that reach the remote (none for a remote that needs
    no token). Returns the commit.
    """
    engine = Path(target.engine)
    ledger = pull_request.Ledger(target.ledger)
    rows = pull_request.refresh(ledger.load(), api, target.repository)
    ledger.save(rows)

    if git(engine, "status", "--porcelain"):
        raise ReleaseError(f"{engine} has changes")

    git(engine, "fetch", "--quiet", "origin", env=auth)

    if pull_request.waiting(rows):
        git(engine, "checkout", "--quiet", target.branch)
        last = pull_request.waiting(rows)[-1]["commit"]

        known = subprocess.run(["git", "-C", str(engine), "merge-base", "--is-ancestor", last, "HEAD"])

        if known.returncode != 0:
            raise ReleaseError(f"the branch {target.branch} does not hold the commit {last[:9]} of the ledger")
    else:
        git(engine, "checkout", "--quiet", "-B", target.branch, f"origin/{target.base}")

    before = git(engine, "rev-parse", "HEAD")
    pushed = git(engine, "rev-parse", "--verify", "--quiet", f"origin/{target.branch}", check=False)
    source = engine / SOURCE

    try:
        source.write_text(nnue.source_text(nnue.read_network(Path(summary["candidate"]))))

        version = bump_version(engine)

        if run_tests:
            build_and_test(engine, Path(target.build))

        git(engine, "add", SOURCE, VERSION_FILE)
        git(engine, "-c", f"user.name={target.author_name}", "-c", f"user.email={target.author_email}",
            "commit", "--quiet", "-m", pull_request.message(summary["stages"]["match"]))
        commit = git(engine, "rev-parse", "HEAD")
        git(engine, "push", "--quiet", f"--force-with-lease={target.branch}:{pushed}", "origin",
            f"{target.branch}:{target.branch}", env=auth)
    except BaseException:
        git(engine, "reset", "--quiet", "--hard", before)
        raise

    row = pull_request.row_of(summary, commit)
    row["version"] = version
    rows.append(row)
    ledger.save(rows)
    try:
        pull_request.ensure_pull_request(api, target.repository, target.branch, target.base, rows,
                                       target.author_name)
    except (github_app.GitHubError, RuntimeError) as error:
        raise PullRequestError(commit, error) from error
    finally:
        ledger.save(rows)
    return commit
