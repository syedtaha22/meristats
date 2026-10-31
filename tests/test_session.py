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

from pathlib import Path

import pytest

from meristats import cycle, session
from meristats.config import Config
from meristats.release import mail, promote

from test_cycle import Fakes, config as _config  # noqa: F401 - the fixture


class Clock:
    def __init__(self):
        self.time = 1_000_000.0

    def __call__(self):
        return self.time


@pytest.fixture
def setup(_config, tmp_path):  # noqa: F811
    clock = Clock()
    fakes = Fakes("H0")
    original = fakes.selfplay

    def selfplay(config, store, engine, out, **settings):
        clock.time += 600
        return original(config, store, engine, out, **settings)

    fakes.selfplay = selfplay
    return _config, clock, fakes


def go(setup, duration, tmp_path, **options):
    config, clock, fakes = setup
    messages = []
    status = session.run(
        config, None, cycle.Settings(test_after=5), duration=duration, book=tmp_path / "b.pgn",
        functions=fakes.functions(), progress=messages.append, clock=clock, **options,
    )
    return status, messages, fakes


def test_durations():
    assert session.seconds("8h") == 28800
    assert session.seconds("90m") == 5400
    assert session.seconds("2d") == 172800
    assert session.seconds("1.5") == 5400

    for bad in ("", "0h", "h", "-1h", "8x"):
        with pytest.raises(ValueError):
            session.seconds(bad)


def test_cycles_follow_each_other_until_the_time_is_over(setup, tmp_path):
    status, messages, fakes = go(setup, 3600 * 3, tmp_path)

    assert status == 0
    assert fakes.calls.count("match") >= 2
    assert any("results of this run" in m for m in messages)


def test_without_a_duration_it_runs_until_ctrl_c_and_then_publishes(setup, tmp_path):
    config, clock, fakes = setup
    fakes.interrupt_at = 9
    status, messages, _ = go(setup, None, tmp_path)

    assert status == 0
    assert "stopped" in messages
    assert fakes.calls.count("match") >= 2
    assert "site: not configured" in messages


def test_two_failed_cycles_in_a_row_stop_the_training(setup, tmp_path):
    config, clock, fakes = setup

    def broken(*args, **kwargs):
        raise RuntimeError("the trainer died")

    fakes.train = broken
    status, messages, _ = go(setup, None, tmp_path)

    assert any("2 cycles in a row failed" in m for m in messages)
    assert fakes.calls.count("build") == 2


def test_the_site_is_left_alone_when_asked(setup, tmp_path):
    status, messages, _ = go(setup, 3600, tmp_path, publish=False)

    assert not any(m.startswith("site:") for m in messages)


def test_a_mail_follows_every_cycle(setup, tmp_path, monkeypatch):
    config, clock, fakes = setup
    sent = []
    settings_file = tmp_path / "mail.env"
    settings_file.write_text("SMTP_USER=a@x\nSMTP_PASSWORD=pw\n")
    config = Config(**{**config.__dict__, "mail_settings": settings_file})
    monkeypatch.setattr(mail, "send", lambda subject, text, page, settings: sent.append(subject))
    go((config, clock, fakes), 3600 * 2, tmp_path)

    assert sent and all(subject.startswith("Meridian training") for subject in sent)


def test_a_network_that_passed_is_committed_and_the_commit_is_in_the_mail(setup, tmp_path,
                                                                           monkeypatch):
    config, clock, fakes = setup
    fakes.verdict = "H1"
    texts = []
    config = Config(**{
        **config.__dict__, "engine_repository": "o/e", "app_id": 1, "app_key": Path("k"),
        "commit_name": "bot", "commit_email": "b@x", "mail_settings": tmp_path / "m.env",
    })
    (tmp_path / "m.env").write_text("SMTP_USER=a@x\nSMTP_PASSWORD=pw\n")
    monkeypatch.setattr(session, "token_of", lambda config, repository: "tok")
    monkeypatch.setattr(session.sync, "sync", lambda target, api, auth: "reset")
    monkeypatch.setattr(promote, "promote", lambda summary, target, api, auth: "c" * 40)
    monkeypatch.setattr(mail, "send", lambda subject, text, page, settings: texts.append(text))
    status, messages, _ = go((config, clock, fakes), 3600, tmp_path)

    assert "engine clone: reset" in messages
    assert any("c" * 12 in text for text in texts)
