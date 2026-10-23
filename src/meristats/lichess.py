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
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""
The games of a Lichess account, read through the public API and kept in a file of the data folder.

Only finished games of standard chess are kept, and only the games that the account played. A
game is stored once, as the API gives it, and a stored game never changes. The API call needs no
token for public games.
"""

import json
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path

import chess

API = "https://lichess.org/api/games/user/"
UNFINISHED = {"created", "started", "aborted", "noStart", "unknownFinish"}


class LichessError(RuntimeError):
    pass


def games_file(data_dir: Path, user: str) -> Path:
    """
    The file that holds the stored games of an account, one JSON object to a line.
    """
    return Path(data_dir) / "lichess" / f"{user}.jsonl"


def stored(path: Path) -> list[dict]:
    """
    The games in the file, in the order that they were stored.
    """
    if not Path(path).exists():
        return []

    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def keeps(game: dict, user: str) -> bool:
    """
    Whether a game is a finished game of standard chess that `user` played.
    """
    players = game.get("players", {})
    names = {
        side: players.get(side, {}).get("user", {}).get("name", "").lower()
        for side in ("white", "black")
    }
    return (
        game.get("variant") == "standard"
        and game.get("status") not in UNFINISHED
        and bool(game.get("moves"))
        and user.lower() in names.values()
    )


def fetch(
    user: str,
    *,
    since_ms: int | None = None,
    until_ms: int | None = None,
    opener: Callable = urllib.request.urlopen,
) -> list[dict]:
    """
    The games of `user` that the API gives for a span of time, oldest first. `opener` is the
    function that opens a request, which a test replaces.
    """
    query = {
        "moves": "true",
        "tags": "false",
        "clocks": "false",
        "evals": "false",
        "opening": "false",
    }

    if since_ms is not None:
        query["since"] = str(since_ms)

    if until_ms is not None:
        query["until"] = str(until_ms)

    request = urllib.request.Request(
        API + urllib.parse.quote(user) + "?" + urllib.parse.urlencode(query),
        headers={"Accept": "application/x-ndjson", "User-Agent": "meristats"},
    )

    try:
        with opener(request, timeout=120) as response:
            lines = [line for line in response.read().decode("utf-8").splitlines() if line.strip()]
    except OSError as error:
        raise LichessError(f"cannot read the games of {user}: {error}") from error

    games = [json.loads(line) for line in lines]
    return sorted((g for g in games if keeps(g, user)), key=lambda g: g.get("createdAt", 0))


def sync(data_dir: Path, user: str, *, opener: Callable = urllib.request.urlopen) -> list[dict]:
    """
    Stores the games that the file does not have yet, and returns them. The API is asked only for
    the games after the newest one that is stored.
    """
    path = games_file(data_dir, user)
    known = stored(path)
    newest = max((g.get("createdAt", 0) for g in known), default=0)
    have = {g["id"] for g in known}
    new = [
        g
        for g in fetch(user, since_ms=newest + 1 if newest else None, opener=opener)
        if g["id"] not in have
    ]

    if new:
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "a", encoding="utf-8") as handle:
            for game in new:
                handle.write(json.dumps(game, separators=(",", ":")) + "\n")

    return new


def analysed_file(data_dir: Path, user: str) -> Path:
    """
    The file with the ids of the games that were analysed, one to a line.
    """
    return Path(data_dir) / "lichess" / f"{user}.analysed"


def analysed(data_dir: Path, user: str) -> set[str]:
    path = analysed_file(data_dir, user)
    return set(path.read_text().split()) if path.exists() else set()


def mark_analysed(data_dir: Path, user: str, ids: Iterable[str]) -> None:
    path = analysed_file(data_dir, user)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as handle:
        for game_id in ids:
            handle.write(game_id + "\n")


def side_of(game: dict, user: str) -> bool:
    """
    The colour that `user` played in a game: True for White.
    """
    white = game["players"]["white"].get("user", {}).get("name", "").lower()
    return white == user.lower()


def result_of(game: dict) -> float:
    """
    The points of White: 1, 0.5 or 0.
    """
    winner = game.get("winner")
    return 1.0 if winner == "white" else 0.0 if winner == "black" else 0.5


def boards_of(game: dict) -> Iterable[tuple[chess.Board, chess.Move | None]]:
    """
    The positions of a game with the move that was played in each, from the first position to the
    last. The last position has no move.
    """
    board = chess.Board()

    for san in game["moves"].split():
        move = board.parse_san(san)
        yield board.copy(), move
        board.push(move)

    yield board.copy(), None
