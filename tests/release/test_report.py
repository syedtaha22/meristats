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
Tests of the mail of the training.
"""

import unittest

from meristats.release import report

PROMOTED = {
    "cycle": "20261023T180000Z", "status": "passed", "seconds": 7200,
    "engine": {"version": "Meridian 1.1.0", "commit": "c" * 40},
    "stages": {
        "selfplay": {"games": 900, "positions": 31000},
        "data": {"new": 33000, "history": 100, "anchor": 1000000},
        "train": {"best_held": 0.0123, "best_epoch": 3, "seconds": {"training": 840.2}},
        "match": {"games": 900, "elo": 12.3, "elo_low": 3.1, "elo_high": 21.5, "verdict": "H1",
                  "timed_out": False, "wins": 300, "draws": 350, "losses": 250, "nodes": 50000},
    },
}


class ReportTest(unittest.TestCase):
    def test_a_passed_cycle_lists_every_stage_and_the_commit(self):
        text = report.text_of(PROMOTED, "d" * 40)
        for part in ("passed", "900 games, 31000 positions",
                     "new 33000, history 100, anchor 1000000", "held-out loss 0.01230", "+12.3 Elo", "verdict H1", "dddddddddddd", "120 min"):
            self.assertIn(part, text)

    def test_a_failed_commit_shows_its_error_in_the_mail(self):
        text = report.text_of(PROMOTED, None, "the engine has changes")
        self.assertIn("Commit failed", text)
        self.assertIn("the engine has changes", text)
        self.assertNotIn("Commit  ", text)
        page = report.page_of(PROMOTED, "cid:logo", None, "<b>x</b>")
        self.assertIn("&lt;b&gt;x&lt;/b&gt;", page)

    def test_a_failed_pull_request_is_named_as_such_next_to_the_commit(self):
        text = report.text_of(PROMOTED, "d" * 40, "403 no permission")
        self.assertRegex(text, r"Commit +dddddddddddd")
        self.assertRegex(text, r"Pull request failed +403 no permission")
        self.assertNotIn("Commit failed", text)

    def test_a_cycle_without_a_commit_has_no_commit_row(self):
        self.assertNotIn("Commit  ", report.text_of(PROMOTED))

    def test_the_mail_has_no_sentence_of_its_own(self):
        text = report.text_of(PROMOTED, "d" * 40)
        self.assertNotIn(".", text.splitlines()[0])

    def test_a_failed_cycle_gives_its_error(self):
        text = report.text_of({"cycle": "n", "status": "failed", "error": "RuntimeError: x"})
        self.assertIn("Error", text)
        self.assertIn("RuntimeError: x", text)

    def test_the_subject_has_the_cycle_and_the_status(self):
        self.assertEqual(report.subject_of(PROMOTED),
                         "Meridian training 20261023T180000Z: passed")

    def test_the_page_escapes_the_text_of_an_error(self):
        page = report.page_of(
            {"cycle": "n", "status": "failed", "error": "<b>bad</b>"}, "cid:logo")
        self.assertIn("&lt;b&gt;bad&lt;/b&gt;", page)
        self.assertNotIn("<b>bad</b>", page)
