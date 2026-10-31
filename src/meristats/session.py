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
Training for a time, or until it is stopped: cycles one after the other, and at the end the
matches of the cycles go on the site.

Before a cycle the branch of the networks is brought up to date with `main`. After a cycle that
passed, the network is committed and put in the pull request of the engine. After every cycle a
mail reports it. The release steps need the settings of the GitHub App and of the repositories in
meristats.toml, the mail needs `mail_settings`, and a step whose settings are missing is skipped
with a note. Two cycles in a row that fail stop the training.
"""

import json
import re
import signal
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from meristats import cycle
from meristats.config import Config
from meristats.release import github_app, mail, promote, report, settings, site, sync
from meristats.release.common import ReleaseError, auth_environment
from meristats.store import Store

UNITS = {"m": 60, "h": 3600, "d": 86400}
FAILURES = 2


def seconds(text: str) -> float:
    """
    A duration such as `8h`, `90m`, `2d` or `1.5h`, in seconds. A number alone counts hours.
    """
    found = re.fullmatch(r"(\d+(?:\.\d+)?)([mhd]?)", text.strip().lower())

    if not found or float(found.group(1)) <= 0:
        raise ValueError(f"not a duration: {text} (examples: 8h, 90m, 2d)")

    return float(found.group(1)) * UNITS[found.group(2) or "h"]


def release_ready(config: Config) -> bool:
    return not settings.missing(
        config, "engine_repository", "app_id", "app_key", "commit_name", "commit_email"
    )


def site_ready(config: Config) -> bool:
    return not settings.missing(
        config, "site_repository", "site_worktree", "app_id", "app_key", "commit_name",
        "commit_email"
    )


def token_of(config: Config, repository: str) -> str:
    return github_app.installation_token(config.app_id, config.app_key, repository)


def results(folder: Path, since: float) -> list[str]:
    """
    One line for each cycle whose summary was written after `since`.
    """
    lines = []

    for path in sorted(Path(folder).glob("*/summary.json")):
        if path.stat().st_mtime < since:
            continue

        data = json.loads(path.read_text())
        match = data.get("stages", {}).get("match")
        line = f"{data['cycle']}  {data['status']:<9}"

        if match and match.get("elo") is not None:
            line += (
                f" {match['elo']:+6.1f} Elo ({match['elo_low']:+.1f} to "
                f"{match['elo_high']:+.1f}), {match['games']} games"
            )

        lines.append(line)

    return lines


def run(
    config: Config,
    store: Store,
    base: cycle.Settings,
    *,
    duration: float | None,
    book: Path,
    publish: bool = True,
    send_mail: bool = True,
    functions: dict[str, Callable] | None = None,
    progress: Callable[[str], None] = lambda text: print(text, flush=True),
    clock: Callable[[], float] = time.time,
) -> int:
    """
    Trains until `duration` seconds have passed, or until Ctrl-C when it is None, and returns the
    exit status. The deadline stops the data of the cycle that is running, which then trains and
    tests what it has, so that the run ends a little after its duration. Ctrl-C ends the cycle that
    is running at once, and the chunks that were finished stay.
    """
    start = clock()
    end = None if duration is None else start + duration
    state = cycle.State.load(config.data_dir, base.test_after)
    failed = 0
    ready = release_ready(config)

    if not ready:
        progress("release: not configured, so no branch sync, commit or pull request")

    try:
        while failed < FAILURES:
            if end is not None and end - clock() < 60:
                break

            target = settings.target_of(config) if ready else None
            auth = api = None

            if ready:
                token = token_of(config, config.engine_repository)
                auth, api = auth_environment(token), github_app.Api(token)
                progress(f"engine clone: {sync.sync(target, api, auth)}")

            summary = cycle.run(
                config, store, replace(base, deadline=end), state, book=book,
                functions=functions, progress=progress, clock=clock,
            )
            progress(cycle.format_summary(summary))
            failed = failed + 1 if summary["status"] == "failed" else 0
            commit = error = None

            if summary["status"] == "passed" and ready:
                try:
                    commit = promote.promote(summary, target, api, auth)
                    progress(f"committed {commit[:9]}")
                except promote.PullRequestError as problem:
                    commit, error = problem.commit, str(problem)
                except ReleaseError as problem:
                    error = str(problem)

                if error:
                    progress(error)

            if send_mail and config.mail_settings:
                try:
                    mail.send(
                        report.subject_of(summary),
                        report.text_of(summary, commit, error),
                        report.page_of(summary, "cid:logo", commit, error),
                        mail.read_settings(config.mail_settings),
                    )
                except Exception as problem:  # noqa: BLE001 - a mail never stops the training
                    progress(f"the mail could not be sent: {problem}")
    except KeyboardInterrupt:
        progress("stopped")

    if failed >= FAILURES:
        progress(f"{FAILURES} cycles in a row failed, so the training stops")

    status = 0
    signal.signal(signal.SIGINT, lambda *_: progress("publishing, please wait"))
    progress("results of this run")
    progress("\n".join(results(Path(config.data_dir) / "cycles", start)) or "no cycle finished")

    if publish and site_ready(config):
        try:
            token = token_of(config, config.site_repository)
            commit = site.publish(settings.site_of(config), store, auth_environment(token))
            progress(f"site: {commit or 'nothing changed'}")
        except Exception as problem:  # noqa: BLE001 - the pins stay, and the step can be run again
            progress(f"the site was not published: {problem}")
            status = 1
    elif publish:
        progress("site: not configured")

    return status

