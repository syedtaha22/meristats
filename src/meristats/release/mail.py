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
The mail of the training: the pieces of an HTML page that every mail program shows (tables and
inline styles, no script and no image to fetch), the logo, and the sending through SMTP.

The page is written for the light theme, and a `prefers-color-scheme: dark` block restates the
colours for the dark theme. SMTP_USER, SMTP_PASSWORD and MAIL_TO come from a settings file of
lines `NAME=value`, and SMTP_HOST and SMTP_PORT there can change the server (smtp.gmail.com, 587).
"""

import base64
import html
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets"

STYLE = """
body, table, td, a { -webkit-text-size-adjust:100%; }
@media (prefers-color-scheme: dark) {
  .page { background:#0d1117 !important; }
  .card { background:#161b22 !important; border-color:#30363d !important; }
  .text { color:#e6edf3 !important; }
  .muted { color:#8b949e !important; }
  .line { border-color:#30363d !important; }
}
[data-ogsc] .page { background:#0d1117 !important; }
[data-ogsc] .card { background:#161b22 !important; }
[data-ogsc] .text { color:#e6edf3 !important; }
[data-ogsc] .muted { color:#8b949e !important; }
"""


def esc(value):
    return html.escape(str(value))


def section(title, body):
    return (f'<tr><td style="padding:18px 28px 0 28px;">'
            f'<div class="text" style="font-size:14px;font-weight:700;color:#1f2328;margin-bottom:10px;">'
            f'{esc(title)}</div>{body}</td></tr>')


def facts(pairs):
    """
    A table of (label, value) pairs.
    """
    rows = "".join(
        f'<tr><td class="muted" style="color:#656d76;padding:3px 14px 3px 0;white-space:nowrap;">{esc(k)}</td>'
        f'<td class="text" style="color:#1f2328;padding:3px 0;">{esc(v)}</td></tr>' for k, v in pairs)
    return f'<table role="presentation" cellpadding="0" cellspacing="0" style="font-size:13px;">{rows}</table>'


def logo_uri():
    """
    The logo as a data URI, for the saved page.
    """
    return "data:image/png;base64," + base64.b64encode((ASSETS / "logo.png").read_bytes()).decode()


def read_settings(path):
    """
    The settings in a file of lines `NAME=value`. Blank lines and lines that start with # are
    skipped, and a value may be in quotes.
    """
    settings = {}

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        name, value = line.split("=", 1)
        settings[name.strip()] = value.strip().strip("\"'")

    return settings


def send(subject, text, page, settings):
    """
    Mails the text and the page, with the logo as an inline image under the address `cid:logo`.
    `settings` is the dictionary of `read_settings`. Returns False, and says why, when the
    settings are not complete.
    """
    user = settings.get("SMTP_USER", "")
    password = settings.get("SMTP_PASSWORD", "")
    recipient = settings.get("MAIL_TO", user)

    if not user or not password or not recipient:
        print("not sent: SMTP_USER, SMTP_PASSWORD and MAIL_TO are not all set")
        return False

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = user
    message["To"] = recipient
    message.set_content(text)
    message.add_alternative(page, subtype="html")
    message.get_payload()[1].add_related((ASSETS / "logo.png").read_bytes(), "image", "png",
                                         cid="<logo>", disposition="inline", filename="logo.png")

    with smtplib.SMTP(settings.get("SMTP_HOST", "smtp.gmail.com"),
                      int(settings.get("SMTP_PORT", "587")), timeout=60) as server:
        server.starttls(context=ssl.create_default_context())
        server.login(user, password)
        server.send_message(message)

    print(f"sent to {recipient}")
    return True
