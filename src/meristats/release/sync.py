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
Brings the branch of the networks of an engine clone up to date with `main`, before a cycle.

The cycle builds the engine at the head of the branch, so a change that is only on `main` has to
be on the branch first. What happens depends on the ledger of `promote.py`:

- When every network of the ledger is in `main` (or its pull request was closed), the branch is
  set to `origin/main`.
- When a network is not merged yet, the branch keeps it, and `origin/main` is merged into it.
  A merge that cannot be done is stopped and the clone is left as it was.

Nothing is pushed. The result is `reset`, `merged` or `current`.
"""

import subprocess
from pathlib import Path

from meristats.release import pull_request
from meristats.release.common import ReleaseError, git


def sync(target, api, auth=None):
    """
    Updates the ledger from the state of its pull requests and the branch from `main`, and returns
    what was done.
    """
    engine = Path(target.engine)
    ledger = pull_request.Ledger(target.ledger)
    rows = pull_request.refresh(ledger.load(), api, target.repository)
    ledger.save(rows)

    if git(engine, "status", "--porcelain"):
        raise ReleaseError(f"{engine} has changes")

    git(engine, "fetch", "--quiet", "origin", env=auth)
    base = f"origin/{target.base}"

    if not pull_request.waiting(rows):
        git(engine, "checkout", "--quiet", "-B", target.branch, base)
        return "reset"

    git(engine, "checkout", "--quiet", target.branch)
    current = subprocess.run(["git", "-C", str(engine), "merge-base", "--is-ancestor", base, "HEAD"])

    if current.returncode == 0:
        return "current"

    merged = subprocess.run(
        ["git", "-C", str(engine), "-c", f"user.name={target.author_name}",
         "-c", f"user.email={target.author_email}", "merge", "--no-edit", "--quiet", base],
        capture_output=True, text=True)

    if merged.returncode != 0:
        git(engine, "merge", "--abort", check=False)
        raise ReleaseError(f"{base} cannot be merged into {target.branch}: "
                           f"{(merged.stdout + merged.stderr)[-300:]}")

    return "merged"
