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
Tests of the sync of the branch of the networks with main.
"""

import unittest

from meristats.release import sync
from meristats.release.common import ReleaseError

from . import test_promote as base

run = base.run


class SyncTest(unittest.TestCase):
    """
    The clone, the remote, the ledger and the fake API of the tests of the commit of a network.
    """

    setUp = base.PromoteTest.setUp
    tearDown = base.PromoteTest.tearDown
    night = base.PromoteTest.night
    origin_head = base.PromoteTest.origin_head
    merge_into_main = base.PromoteTest.merge_into_main

    def advance_main(self, name="extra.txt", content="x\n"):
        """
        A commit on main that someone else made, pushed from another clone.
        """
        work = self.root / f"other-{name}"
        run("git", "clone", "-q", str(self.origin), str(work))
        for key, value in (("user.name", "other"), ("user.email", "o@example.com")):
            run("git", "config", key, value, cwd=work)
        (work / name).write_text(content)
        run("git", "add", "-A", cwd=work)
        run("git", "commit", "-q", "-m", f"change {name}", cwd=work)
        run("git", "push", "-q", "origin", "main", cwd=work)

    def sync(self):
        return sync.sync(self.args, self.api)

    def head(self, ref="HEAD"):
        return run("git", "rev-parse", ref, cwd=self.engine)

    def test_without_a_network_that_waits_the_branch_is_set_to_main(self):
        self.advance_main()

        self.assertEqual(self.sync(), "reset")
        self.assertEqual(self.head(), self.head("origin/main"))
        self.assertEqual(run("git", "status", "--porcelain", cwd=self.engine), "")
        self.assertEqual(run("git", "branch", "--show-current", cwd=self.engine), "nightly-networks")

    def test_a_network_that_waits_stays_and_main_is_merged_into_the_branch(self):
        network = self.night("n1", 12.3, "network 1\n")
        self.advance_main()

        self.assertEqual(self.sync(), "merged")
        merged = self.head()
        parents = run("git", "show", "-s", "--format=%P", merged, cwd=self.engine).split()
        self.assertEqual(len(parents), 2)
        self.assertEqual(run("git", "merge-base", "--is-ancestor", network, merged, cwd=self.engine), "")
        self.assertEqual(run("git", "merge-base", "--is-ancestor", "origin/main", merged,
                             cwd=self.engine), "")
        self.assertEqual(run("git", "log", "-1", "--format=%an|%cn", cwd=self.engine),
                         "ce-meridian-bot[bot]|ce-meridian-bot[bot]")
        self.assertEqual(run("git", "show", f"{merged}:src/nnue_net.inc", cwd=self.engine),
                         base.source_of("network 1\n"))

    def test_a_branch_that_has_main_already_is_left_as_it_is(self):
        self.night("n1", 12.3, "network 1\n")
        before = self.head()

        self.assertEqual(self.sync(), "current")
        self.assertEqual(self.head(), before)

    def test_a_network_whose_pull_request_was_merged_no_longer_holds_the_branch(self):
        self.night("n1", 12.3, "network 1\n")
        self.merge_into_main()
        self.api.merge(1)
        self.advance_main()

        self.assertEqual(self.sync(), "reset")
        self.assertEqual(self.head(), self.head("origin/main"))

    def test_a_merge_that_cannot_be_done_leaves_the_clone_as_it_was(self):
        self.night("n1", 12.3, "network 1\n")
        before = self.head()
        self.advance_main("src/nnue_net.inc", "another network\n")

        with self.assertRaisesRegex(ReleaseError, "cannot be merged"):
            self.sync()

        self.assertEqual(self.head(), before)
        self.assertEqual(run("git", "status", "--porcelain", cwd=self.engine), "")

    def test_a_clone_with_changes_is_refused(self):
        (self.engine / "stray.txt").write_text("x")

        with self.assertRaisesRegex(ReleaseError, "has changes"):
            self.sync()
