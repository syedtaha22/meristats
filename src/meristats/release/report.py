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
The mail of the training: the numbers of one cycle as text and as a page.
"""

from meristats.release import mail


def lines_of(summary, commit=None, promotion_error=None):
    """
    The facts of the cycle as (label, value) pairs.
    """
    stages = summary.get("stages", {})
    facts = []

    if "engine" in stages or "engine" in summary:
        engine = summary.get("engine", {})
        facts.append(("Engine", f"{engine.get('version', '')} {str(engine.get('commit', ''))[:10]}"))

    if "selfplay" in stages:
        s = stages["selfplay"]
        facts.append(("Self-play", f"{s['games']} games, {s['positions']} positions"))

    if "data" in stages:
        facts.append(("Training data", ", ".join(f"{k} {v}" for k, v in stages["data"].items())))

    if "train" in stages:
        s = stages["train"]
        facts.append(("Training", f"held-out loss {s['best_held']:.5f} at epoch {s['best_epoch']}, "
                      f"{s['seconds']['training']:.0f} s"))

    if "match" in stages:
        s = stages["match"]
        elo = ""
        if s.get("elo") is not None:
            elo = f", {s['elo']:+.1f} Elo ({s['elo_low']:+.1f} to {s['elo_high']:+.1f})"
        stop = ", stopped at the time limit" if s.get("timed_out") else ""
        facts.append(("Match", f"{s['games']} games{elo}, verdict {s['verdict']}{stop}"))

    if commit:
        facts.append(("Commit", commit[:12]))

    if promotion_error:
        facts.append(("Pull request failed" if commit else "Commit failed", promotion_error))

    if "stopped_before" in summary:
        facts.append(("Stopped before", summary["stopped_before"]))

    if "error" in summary:
        facts.append(("Error", summary["error"]))

    facts.append(("Time", f"{summary.get('seconds', 0) // 60} min"))
    return facts


def subject_of(summary):
    return f"Meridian training {summary['cycle']}: {summary['status']}"


def text_of(summary, commit=None, promotion_error=None):
    body = "\n".join(f"  {label:<22}{value}" for label, value in lines_of(summary, commit, promotion_error))
    return f"Meridian training, cycle {summary['cycle']}: {summary['status']}\n\n{body}\n"


def page_of(summary, logo, commit=None, promotion_error=None):
    """
    The page of the mail.
    """
    colour = {"passed": "#1a7f37", "rejected": "#9a6700"}.get(summary["status"], "#c62828")
    head = (
        '<tr><td align="center" style="padding:26px 28px 6px 28px;text-align:center;">'
        f'<img src="{logo}" width="120" alt="Meridian" '
        'style="display:block;margin:0 auto;border:0;height:auto;">'
        '<div class="text" style="font-size:22px;font-weight:700;color:#1f2328;margin-top:14px;">'
        f'Training, cycle {mail.esc(summary["cycle"])}</div>'
        f'<div style="font-size:14px;font-weight:600;color:{colour};margin-top:6px;">'
        f'{mail.esc(summary["status"])}</div></td></tr>'
    )
    body = mail.section("The cycle", mail.facts(lines_of(summary, commit, promotion_error)))
    footer = ('<tr><td style="padding:18px 28px 24px 28px;"><div class="muted" '
              'style="font-size:11px;color:#8c959f;">Sent by the training of '
              'meristats.</div></td></tr>')
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>Meridian training</title>
<style>{mail.STYLE}</style>
</head>
<body class="page" style="margin:0;padding:0;background:#f6f8fa;font-family:-apple-system,'Segoe UI',Helvetica,Arial,sans-serif;">
<table role="presentation" class="page" width="100%" cellpadding="0" cellspacing="0" style="background:#f6f8fa;">
<tr><td align="center" style="padding:24px 12px;">
<table role="presentation" class="card" width="640" cellpadding="0" cellspacing="0"
  style="max-width:640px;width:100%;background:#ffffff;border:1px solid #d0d7de;border-radius:12px;">
{head}
{body}
{footer}
</table>
</td></tr>
</table>
</body>
</html>
"""

