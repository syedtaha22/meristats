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
Tests of the sending of the mail.
"""

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from meristats.release import mail


class FakeServer:
    sent = []

    def __init__(self, host, port, timeout):
        self.address = (host, port)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self, context):
        self.tls = True

    def login(self, user, password):
        self.login = (user, password)

    def send_message(self, message):
        FakeServer.sent.append((self.address, self.login, message))


class MailTest(unittest.TestCase):
    def setUp(self):
        FakeServer.sent = []

    def send(self, settings):
        with mock.patch.object(mail.smtplib, "SMTP", FakeServer), redirect_stdout(StringIO()):
            return mail.send("Subject", "Text", '<p><img src="cid:logo"></p>', settings)

    def test_a_mail_has_the_text_the_page_and_the_logo_inline(self):
        sent = self.send({"SMTP_USER": "a@x", "SMTP_PASSWORD": "pw", "MAIL_TO": "b@x"})
        self.assertTrue(sent)
        address, login, message = FakeServer.sent[0]
        self.assertEqual(address, ("smtp.gmail.com", 587))
        self.assertEqual(login, ("a@x", "pw"))
        self.assertEqual((message["From"], message["To"], message["Subject"]), ("a@x", "b@x", "Subject"))
        parts = [part.get_content_type() for part in message.walk()]
        self.assertIn("text/plain", parts)
        self.assertIn("text/html", parts)
        self.assertIn("image/png", parts)

    def test_the_server_and_the_recipient_can_be_set(self):
        self.send({"SMTP_USER": "a@x", "SMTP_PASSWORD": "pw", "SMTP_HOST": "mail.example",
                   "SMTP_PORT": "25"})
        address, _, message = FakeServer.sent[0]
        self.assertEqual((address, message["To"]), (("mail.example", 25), "a@x"))

    def test_nothing_is_sent_without_the_settings(self):
        self.assertFalse(self.send({"SMTP_USER": "a@x"}))
        self.assertEqual(FakeServer.sent, [])

    def test_the_settings_are_read_from_a_file_of_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mail.env"
            path.write_text('# a comment\nSMTP_USER=a@x\n\nSMTP_PASSWORD="pw w"\nMAIL_TO=b@x\n')
            settings = mail.read_settings(path)

        self.assertEqual(settings, {"SMTP_USER": "a@x", "SMTP_PASSWORD": "pw w", "MAIL_TO": "b@x"})

    def test_text_is_escaped_and_the_logo_is_a_png(self):
        self.assertEqual(mail.esc("<b>&"), "&lt;b&gt;&amp;")
        self.assertTrue(mail.logo_uri().startswith("data:image/png;base64,"))
        self.assertIn("<td", mail.facts([("a", "b")]))
