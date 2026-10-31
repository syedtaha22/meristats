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
Tests of the commit of a network and of its pull request.
"""

import base64
import dataclasses
import hashlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from meristats import nnue
from meristats.release import github_app, promote, pull_request
from meristats.release.common import ReleaseError, auth_environment
from meristats.release.settings import Target

REPOSITORY = "owner/engine"
VERSION_HPP = """namespace chess {{
    constexpr const char *kEngineVersion = "{}";
}}
"""


def network_of(label):
    """
    A network that is the same for the same label.
    """
    rng = np.random.default_rng(int(hashlib.sha256(label.encode()).hexdigest()[:8], 16))
    return nnue.Network(
        rng.integers(-50, 50, (nnue.INPUTS, nnue.HIDDEN), dtype=np.int16),
        rng.integers(-50, 50, nnue.HIDDEN, dtype=np.int16),
        rng.integers(-50, 50, (nnue.BUCKETS, 2, nnue.HIDDEN), dtype=np.int16),
        rng.integers(-500, 500, nnue.BUCKETS, dtype=np.int32),
    )


def source_of(label):
    return nnue.source_text(network_of(label)).strip()


def run(*command, cwd=None):
    done = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def summary_of(night, elo, candidate):
    return {
        "cycle": night, "candidate": str(candidate),
        "stages": {
            "selfplay": {"positions": 60000},
            "train": {"best_held": 0.0161},
            "match": {"elo": elo, "elo_low": elo - 8, "elo_high": elo + 8, "games": 4000,
                      "wins": 1300, "draws": 1500, "losses": 1200, "nodes": 50000},
        },
    }


class FakeApi:
    """
    The part of the GitHub API that the pull request uses, with the pull requests in memory.
    """

    def __init__(self):
        self.pulls = {}
        self.calls = []
        self.fail_create = False
        self.login = "ce-meridian-bot[bot]"

    def request(self, method, path, body=None):
        self.calls.append((method, path.split("?")[0]))

        if method == "GET" and "/pulls?" in path:
            return [{"number": n, **p} for n, p in self.pulls.items() if p["state"] == "open"]

        if method == "GET":
            number = int(path.rsplit("/", 1)[1])
            return {"number": number, **self.pulls[number]}

        if method == "POST":
            if self.fail_create:
                raise github_app.GitHubError("POST: 502")

            number = len(self.pulls) + 1
            self.pulls[number] = {"state": "open", "merged": False, "title": body["title"],
                                  "body": body["body"], "head": body["head"]}
            return {"number": number, "user": {"login": self.login}}

        number = int(path.rsplit("/", 1)[1])
        self.pulls[number]["body"] = body["body"]
        return {"number": number}

    def merge(self, number):
        self.pulls[number].update(state="closed", merged=True)


class PromoteTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        seed = self.root / "seed"
        seed.mkdir()
        run("git", "init", "-q", "-b", "main", cwd=seed)
        (seed / "src").mkdir()
        (seed / "include").mkdir()
        (seed / "src" / "nnue_net.inc").write_text("network 0\n")
        (seed / "include" / "version.hpp").write_text(VERSION_HPP.format("1.1.0"))
        for key, value in (("user.name", "owner"), ("user.email", "owner@example.com")):
            run("git", "config", key, value, cwd=seed)
        run("git", "add", "-A", cwd=seed)
        run("git", "commit", "-q", "-m", "first", cwd=seed)
        self.origin = self.root / "origin.git"
        run("git", "clone", "-q", "--bare", str(seed), str(self.origin))
        run("git", "--git-dir", str(self.origin), "branch", "nightly-networks", "main")
        self.engine = self.root / "engine"
        run("git", "clone", "-q", str(self.origin), str(self.engine))
        run("git", "checkout", "-q", "nightly-networks", cwd=self.engine)
        self.args = Target(
            engine=self.engine, repository=REPOSITORY, ledger=self.root / "ledger.json",
            branch="nightly-networks", base="main", build=self.root / "build",
            author_name="ce-meridian-bot[bot]",
            author_email="346834764+ce-meridian-bot[bot]@users.noreply.github.com")
        self.api = FakeApi()

    def tearDown(self):
        self.directory.cleanup()

    def night(self, name, elo, content):
        candidate = self.root / f"{name}.nnue"
        nnue.write_network(network_of(content), candidate)
        return promote.promote(summary_of(name, elo, candidate), self.args, self.api,
                                       run_tests=False)

    def origin_head(self, branch="nightly-networks"):
        return run("git", "--git-dir", str(self.origin), "rev-parse", branch)

    def test_the_first_night_commits_pushes_and_opens_the_pull_request(self):
        commit = self.night("n1", 12.3, "network 1\n")
        self.assertEqual(self.origin_head(), commit)
        self.assertEqual(run("git", "log", "-1", "--format=%s", cwd=self.engine),
                         "Update the network, +12.3 Elo")
        self.assertEqual(run("git", "log", "-1", "--format=%an <%ae>|%cn", cwd=self.engine),
                         "ce-meridian-bot[bot] <346834764+ce-meridian-bot[bot]@users.noreply.github.com>"
                         "|ce-meridian-bot[bot]")
        self.assertEqual(run("git", "log", "-1", "--format=%b", cwd=self.engine), "")
        self.assertEqual(len(self.api.pulls), 1)
        pull = self.api.pulls[1]
        self.assertEqual(pull["title"], "Trained networks")
        self.assertIn("| n1 | +12.3 | +4.3&nbsp;to&nbsp;+20.3 |", pull["body"])
        self.assertEqual(json.loads(self.args.ledger.read_text())[0]["commit"], commit)

    def version_in(self, commit):
        text = run("git", "show", f"{commit}:include/version.hpp", cwd=self.engine)
        return text.split('kEngineVersion = "')[1].split('"')[0]

    def test_every_commit_of_a_network_raises_the_minor_version_in_the_same_commit(self):
        first = self.night("n1", 12.3, "network 1\n")
        second = self.night("n2", 8.0, "network 2\n")
        changed = run("git", "show", "--name-only", "--format=", first, cwd=self.engine).split()

        self.assertEqual(sorted(changed), ["include/version.hpp", "src/nnue_net.inc"])
        self.assertEqual((self.version_in(first), self.version_in(second)), ("1.2.0", "1.3.0"))
        self.assertEqual(json.loads(self.args.ledger.read_text())[1]["version"], "1.3.0")

    def test_after_a_merge_the_next_commit_goes_on_from_the_version_of_main(self):
        self.night("n1", 12.3, "network 1\n")
        self.merge_into_main()
        self.api.merge(1)
        third = self.night("n2", 9.0, "network 2\n")

        self.assertEqual(self.version_in(third), "1.3.0")

    def test_an_engine_without_a_version_is_refused_and_left_as_it_was(self):
        run("git", "checkout", "-q", "main", cwd=self.engine)
        (self.engine / "include" / "version.hpp").unlink()
        run("git", "add", "-A", cwd=self.engine)
        run("git", "-c", "user.name=t", "-c", "user.email=t@e", "commit", "-q", "-m", "x",
            cwd=self.engine)
        run("git", "push", "-q", "origin", "main", cwd=self.engine)
        run("git", "checkout", "-q", "nightly-networks", cwd=self.engine)

        with self.assertRaisesRegex(ReleaseError, "no kEngineVersion"):
            self.night("n1", 12.3, "network 1\n")

        # The branch was set to main, and the failed change is gone from the clone.
        self.assertEqual(run("git", "rev-parse", "HEAD", cwd=self.engine),
                         run("git", "rev-parse", "origin/main", cwd=self.engine))
        self.assertEqual(run("git", "status", "--porcelain", cwd=self.engine), "")
        self.assertEqual(self.api.pulls, {})

    def test_a_later_night_goes_on_top_and_updates_the_same_pull_request(self):
        first = self.night("n1", 12.3, "network 1\n")
        second = self.night("n2", 8.0, "network 2\n")
        self.assertEqual(run("git", "rev-parse", f"{second}~1", cwd=self.engine), first)
        self.assertEqual(self.origin_head(), second)
        self.assertEqual(len(self.api.pulls), 1)
        body = self.api.pulls[1]["body"]
        self.assertIn("| n1 |", body)
        self.assertIn("| n2 |", body)
        self.assertEqual([c for c in self.api.calls if c[0] == "POST"],
                         [("POST", f"/repos/{REPOSITORY}/pulls")])

    def merge_into_main(self, squash=False):
        work = self.root / "merger"
        run("git", "clone", "-q", str(self.origin), str(work))
        for key, value in (("user.name", "owner"), ("user.email", "owner@example.com")):
            run("git", "config", key, value, cwd=work)
        if squash:
            run("git", "merge", "-q", "--squash", "origin/nightly-networks", cwd=work)
            run("git", "commit", "-q", "-m", "squashed", cwd=work)
        else:
            run("git", "merge", "-q", "--no-ff", "-m", "merge", "origin/nightly-networks", cwd=work)
        run("git", "push", "-q", "origin", "main", cwd=work)

    def test_after_a_merge_the_branch_starts_again_from_main_with_a_new_pull_request(self):
        self.night("n1", 12.3, "network 1\n")
        self.merge_into_main()
        self.api.merge(1)
        commit = self.night("n2", 9.0, "network 2\n")
        parent = run("git", "rev-parse", f"{commit}~1", cwd=self.engine)
        self.assertEqual(parent, run("git", "rev-parse", "origin/main", cwd=self.engine))
        self.assertEqual(len(self.api.pulls), 2)
        body = self.api.pulls[2]["body"]
        self.assertNotIn("| n1 |", body)
        self.assertIn("| n2 |", body)

    def test_a_squash_merge_is_handled_like_any_other_merge(self):
        self.night("n1", 12.3, "network 1\n")
        self.merge_into_main(squash=True)
        self.api.merge(1)
        commit = self.night("n2", 9.0, "network 2\n")
        self.assertEqual(run("git", "show", f"{commit}:src/nnue_net.inc", cwd=self.engine),
                         source_of("network 2\n"))
        self.assertEqual(run("git", "rev-parse", f"{commit}~1", cwd=self.engine),
                         run("git", "rev-parse", "origin/main", cwd=self.engine))

    def test_a_pull_request_that_is_closed_without_a_merge_does_not_keep_its_network(self):
        self.night("n1", 12.3, "network 1\n")
        self.api.pulls[1]["state"] = "closed"
        commit = self.night("n2", 9.0, "network 2\n")
        self.assertEqual(run("git", "rev-parse", f"{commit}~1", cwd=self.engine),
                         run("git", "rev-parse", "origin/main", cwd=self.engine))
        self.assertNotIn("| n1 |", self.api.pulls[2]["body"])

    def test_a_failed_pull_request_keeps_the_commit_and_reports_it(self):
        self.api.fail_create = True

        with self.assertRaises(promote.PullRequestError) as raised:
            self.night("n1", 12.3, "network 1\n")

        pushed = self.origin_head()
        self.assertEqual(raised.exception.commit, pushed)
        self.assertEqual(json.loads(self.args.ledger.read_text())[0]["commit"], pushed)
        self.api.fail_create = False
        commit = self.night("n2", 9.0, "network 2\n")
        self.assertEqual(run("git", "rev-parse", f"{commit}~1", cwd=self.engine), pushed)
        self.assertEqual(len(self.api.pulls), 1)
        self.assertIn("| n1 |", self.api.pulls[1]["body"])

    def test_a_clone_with_changes_is_refused_and_nothing_is_changed(self):
        (self.engine / "stray.txt").write_text("x")

        with self.assertRaises(ReleaseError):
            self.night("n1", 12.3, "network 1\n")

        self.assertEqual(self.api.pulls, {})

    def test_a_failed_build_leaves_the_clone_as_it_was(self):
        before = run("git", "rev-parse", "HEAD", cwd=self.engine)
        candidate = self.root / "c.nnue"
        nnue.write_network(network_of("network 1\n"), candidate)
        broken = dataclasses.replace(self.args, build=self.root / "no-such" / "build")

        with self.assertRaises(ReleaseError):
            promote.promote(summary_of("n1", 5.0, candidate), broken, self.api,
                                    run_tests=True)

        self.assertEqual(run("git", "rev-parse", "HEAD", cwd=self.engine), before)
        self.assertEqual((self.engine / "src" / "nnue_net.inc").read_text(), "network 0\n")
        self.assertEqual(self.origin_head(), before)

    def test_a_pull_request_that_the_app_did_not_open_is_an_error(self):
        self.api.login = "someone-else"

        with self.assertRaisesRegex(RuntimeError, "not by ce-meridian-bot"):
            self.night("n1", 12.3, "network 1\n")

    def test_the_token_is_given_to_git_by_the_environment_and_not_by_the_arguments(self):
        environment = auth_environment("tok")
        self.assertEqual(environment["GIT_CONFIG_COUNT"], "1")
        self.assertIn(base64.b64encode(b"x-access-token:tok").decode(),
                      environment["GIT_CONFIG_VALUE_0"])


class GitHubAppTest(unittest.TestCase):
    def test_the_token_of_the_app_is_signed_with_the_key_and_the_exchange_asks_for_the_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            key, public = Path(directory) / "k.pem", Path(directory) / "p.pem"
            run("openssl", "genrsa", "-out", str(key), "2048")
            run("openssl", "rsa", "-in", str(key), "-pubout", "-out", str(public))
            token = github_app.app_jwt(5439742, key, now=1000)
            header, payload, signature = token.split(".")
            claims = json.loads(base64.urlsafe_b64decode(payload + "=="))
            self.assertEqual(claims, {"iat": 940, "exp": 1540, "iss": "5439742"})
            signed = Path(directory) / "sig"
            signed.write_bytes(base64.urlsafe_b64decode(signature + "=="))
            checked = subprocess.run(
                ["openssl", "dgst", "-sha256", "-verify", str(public), "-signature", str(signed)],
                input=f"{header}.{payload}".encode(), capture_output=True)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            seen = []

            def opener(request, timeout):
                seen.append((request.get_method(), request.full_url,
                             request.get_header("Authorization")))

                if request.full_url.endswith("/installation"):
                    return io.BytesIO(b'{"id": 77}')

                return io.BytesIO(b'{"token": "installation-token"}')

            result = github_app.installation_token(5439742, key, "owner/engine", opener)

        self.assertEqual(result, "installation-token")
        self.assertEqual(
            [call[:2] for call in seen],
            [("GET", "https://api.github.com/repos/owner/engine/installation"),
             ("POST", "https://api.github.com/app/installations/77/access_tokens")])
        self.assertTrue(all(call[2].startswith("Bearer ") for call in seen))

    def test_the_identity_of_the_app_is_read_with_the_signed_token(self):
        with tempfile.TemporaryDirectory() as directory:
            key = Path(directory) / "k.pem"
            run("openssl", "genrsa", "-out", str(key), "2048")
            seen = []

            def opener(request, timeout):
                seen.append((request.get_method(), request.full_url))
                return io.BytesIO(b'{"slug": "ce-meridian-bot", "id": 5439742}')

            identity = github_app.app_identity(5439742, key, opener)

        self.assertEqual(identity, ("ce-meridian-bot", 5439742))
        self.assertEqual(seen, [("GET", "https://api.github.com/app")])



class LedgerTest(unittest.TestCase):
    def test_a_merged_row_leaves_the_table_and_is_counted(self):
        old = pull_request.row_of(summary_of("n1", 5.0, "c"), "a" * 40)
        new = pull_request.row_of(summary_of("n2", 7.0, "c"), "b" * 40)
        old.update(pr=3, state="merged")
        body = pull_request.table([old, new], REPOSITORY)
        self.assertNotIn("| n1 |", body)
        self.assertIn("| n2 |", body)

    def test_the_numbers_of_a_row_have_no_space_to_break_a_line_at(self):
        row = pull_request.row_of(summary_of("n1", 12.3, "c"), "a" * 40)
        cells = pull_request.cells(row)
        self.assertEqual(len(cells), 8)
        self.assertTrue(all(" " not in cell for cell in cells))
        self.assertEqual(cells[4], "1300&nbsp;/&nbsp;1500&nbsp;/&nbsp;1200")
        self.assertEqual(cells[5], "50000")

    def test_the_table_lists_the_newest_network_first_and_has_the_legend_after_it(self):
        rows = [pull_request.row_of(summary_of(n, e, "c"), c * 40)
                for n, e, c in (("n1", 5.0, "a"), ("n2", 7.0, "b"), ("n3", 9.0, "c"))]
        body = pull_request.table(rows, REPOSITORY)
        order = [body.index(f"| {n} |") for n in ("n3", "n2", "n1")]
        self.assertEqual(order, sorted(order))
        self.assertEqual(sum(1 for line in body.splitlines() if line.startswith("- **")), 4)
        self.assertTrue(body.startswith("| Cycle |"))
        self.assertNotIn("earlier", body)

    def test_the_commit_message_is_one_short_line_with_the_gain(self):
        text = pull_request.message({"elo": 12.34})
        self.assertEqual(text, "Update the network, +12.3 Elo")
        self.assertLessEqual(len(text), 50)


if __name__ == "__main__":
    unittest.main()
