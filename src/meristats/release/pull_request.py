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
The ledger of the networks that passed their test, and the pull request that shows it.

The ledger is a JSON file on the machine that trains, with one row for each network that was
committed. The description of the pull request is a table of every row, written again from the
ledger each time, so nothing is read back from GitHub except the state of a pull request. A
merged pull request is closed for good on GitHub, so the next network opens a new one, and its
table still holds the earlier rows, marked with the pull request that merged them.
"""

import json
import os
from pathlib import Path

TITLE = "Trained networks"
OPEN, MERGED, CLOSED = "open", "merged", "closed"


def message(match):
    """
    The commit message of a network: one line with the gain.
    """
    return f"Update the network, {match['elo']:+.1f} Elo"


def row_of(summary, commit):
    """
    The row of the ledger for a cycle that passed, from its summary.
    """
    stages = summary["stages"]
    match = stages["match"]
    return {
        "cycle": summary["cycle"],
        "elo": match["elo"], "elo_low": match["elo_low"], "elo_high": match["elo_high"],
        "games": match["games"], "wins": match["wins"], "draws": match["draws"],
        "losses": match["losses"], "nodes": match["nodes"],
        "selfplay_positions": stages.get("selfplay", {}).get("positions"),
        "held_out_loss": stages.get("train", {}).get("best_held"),
        "commit": commit, "pr": None, "state": OPEN,
    }


class Ledger:
    def __init__(self, path):
        self.path = Path(path)

    def load(self):
        return json.loads(self.path.read_text()) if self.path.exists() else []

    def save(self, rows):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".part")
        temporary.write_text(json.dumps(rows, indent=1))
        os.replace(temporary, self.path)


NB = "&nbsp;"
HEADER = ("| Cycle | Elo | 95% interval | Games | W / D / L | Nodes | Self-play | Loss | "
          "Commit |")
LEGEND = (
    "- **Elo:** the gain over the network before it, with its 95% interval",
    "- **Nodes:** nodes searched for each move of the test games",
    "- **Self-play:** positions of the cycle's games of the engine against itself",
    "- **Loss:** held-out loss of the training",
)


def cells(row):
    """
    The cells of a row, without the commit. A cell has no space that a browser could break a line
    at.
    """
    return [
        row["cycle"],
        f"{row['elo']:+.1f}",
        f"{row['elo_low']:+.1f}{NB}to{NB}{row['elo_high']:+.1f}",
        str(row["games"]),
        f"{row['wins']}{NB}/{NB}{row['draws']}{NB}/{NB}{row['losses']}",
        str(row["nodes"]),
        "-" if row["selfplay_positions"] is None else str(row["selfplay_positions"]),
        "-" if row["held_out_loss"] is None else f"{row['held_out_loss']:.5f}",
    ]


def table(rows, repository):
    """
    The description of the pull request: a table with one line for each network that is not in
    `main` yet, newest first, and a list of what the columns are.
    """
    lines = [HEADER, "|:--|--:|:--|--:|:--|--:|--:|--:|:--|"]

    for row in reversed(waiting(rows)):
        link = f"[{row['commit'][:9]}](https://github.com/{repository}/commit/{row['commit']})"
        lines.append("| " + " | ".join(cells(row) + [link]) + " |")

    return "\n".join(lines + [""] + list(LEGEND)) + "\n"


def refresh(rows, api, repository):
    """
    Sets the state of the rows whose pull request was open when they were last seen.
    """
    for row in rows:
        if row["pr"] is not None and row["state"] == OPEN:
            pull = api.request("GET", f"/repos/{repository}/pulls/{row['pr']}")
            row["state"] = MERGED if pull.get("merged") else (CLOSED if pull["state"] == "closed" else OPEN)

    return rows


def waiting(rows):
    """
    The rows that are not in `main` yet: those with an open pull request, and those that have no
    pull request because opening it failed.
    """
    return [row for row in rows if row["pr"] is None or row["state"] == OPEN]


def ensure_pull_request(api, repository, branch, base, rows, author=None):
    """
    Opens the pull request of the branch when none is open, gives its number to the rows that
    have none, and writes the table into its description. Returns the number. A pull request that
    this call opens has to be opened by `author` (the login of the App), or it is an error.
    """
    owner = repository.split("/")[0]
    found = api.request(
        "GET", f"/repos/{repository}/pulls?state=open&head={owner}:{branch}&base={base}")

    if found:
        number = found[0]["number"]
    else:
        created = api.request("POST", f"/repos/{repository}/pulls", {
            "title": TITLE, "head": branch, "base": base, "body": table(rows, repository)})
        number = created["number"]
        opened_by = created.get("user", {}).get("login")

        if author is not None and opened_by != author:
            raise RuntimeError(f"the pull request #{number} was opened by {opened_by}, not by {author}")

    for row in rows:
        if row["pr"] is None:
            row["pr"], row["state"] = number, OPEN

    api.request("PATCH", f"/repos/{repository}/pulls/{number}", {"body": table(rows, repository)})
    return number
