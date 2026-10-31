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
Puts the matches of the cycles on the site of meristats, and pushes the site as the App.

Every cycle under the folder of the cycles that has a summary with a match gets a pin on the page
`matches`, labelled with its time, unless it has one. Then the pinned runs are written into the
folder of the site, which is the worktree of the branch `site`, and the change is committed by the
App and pushed with its token. Nothing is committed when the site does not change. The folder of
the site is made from `origin/site` when it is missing, and it is set back to `origin/site`
before each publication, so that it holds nothing but what was published.
"""

import json
from datetime import datetime
from pathlib import Path

from meristats.publish import publish as write_site
from meristats.release.common import git

PAGE = "matches"
BRANCH = "site"


def label_of(cycle):
    """
    The label of a cycle on the page: `Cycle 2026-10-31 03:15 UTC` for an id that is a UTC time,
    and `Cycle <id>` for any other.
    """
    try:
        moment = datetime.strptime(cycle, "%Y%m%dT%H%M%SZ")
    except ValueError:
        return f"Cycle {cycle}"

    return f"Cycle {moment:%Y-%m-%d %H:%M} UTC"


def cycles(folder):
    """
    The cycles under `folder` that played a match, as (label, run id), oldest first.
    """
    found = []

    for path in sorted(Path(folder).glob("*/summary.json")):
        summary = json.loads(path.read_text())
        run = summary.get("stages", {}).get("match", {}).get("run")

        if run:
            found.append((label_of(summary["cycle"]), run))

    return found


def message(added):
    if len(added) == 1:
        return f"Publish the match of {added[0][0].lower()}"

    if added:
        return f"Publish the matches of {len(added)} cycles"

    return "Publish the site"


def prepare(site, remote, auth=None):
    """
    Makes the folder of the site the worktree of the branch `site`, at `origin/site`.
    """
    directory = Path(site.directory)
    clone = directory.parent

    git(clone, "fetch", "--quiet", remote, f"+{BRANCH}:refs/remotes/origin/{BRANCH}", env=auth)

    if not (directory / ".git").exists():
        git(clone, "worktree", "add", "--quiet", "-B", BRANCH, str(directory), f"origin/{BRANCH}")

    git(directory, "checkout", "--quiet", BRANCH)
    git(directory, "reset", "--quiet", "--hard", f"origin/{BRANCH}")
    git(directory, "clean", "--quiet", "-fdx")


def publish(site, store, auth=None):
    """
    Pins the new cycles, writes the site, and commits and pushes it. Returns the commit, or None
    when the site did not change.
    """
    directory = Path(site.directory)
    remote = site.remote or f"https://github.com/{site.repository}.git"
    have = {pin["label"] for pin in store.pins() if pin["page"] == PAGE}
    added = [(label, run) for label, run in cycles(site.cycles) if label not in have]

    for label, run in added:
        store.pin(PAGE, label, run)

    prepare(site, remote, auth)
    write_site(directory, store)
    git(directory, "add", "-A")

    if not git(directory, "status", "--porcelain"):
        return None

    git(directory, "-c", f"user.name={site.author_name}", "-c", f"user.email={site.author_email}",
        "commit", "--quiet", "-m", message(added))
    git(directory, "push", "--quiet", remote, f"{BRANCH}:{BRANCH}", env=auth)
    return git(directory, "rev-parse", "HEAD")

