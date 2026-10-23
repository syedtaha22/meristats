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


import io
import json

import pytest

from meristats import lichess


def game(
    game_id, created, moves="e4 e5 Nf3", winner="white", white="ce_meridian", black="other", **extra
):
    return {
        "id": game_id,
        "createdAt": created,
        "variant": "standard",
        "status": "mate",
        "winner": winner,
        "moves": moves,
        "players": {"white": {"user": {"name": white}}, "black": {"user": {"name": black}}},
        **extra,
    }


class Reply(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def opener_of(games, seen):
    def opener(request, timeout):
        seen.append(request.full_url)
        return Reply("\n".join(json.dumps(g) for g in games).encode())

    return opener


def test_only_finished_standard_games_of_the_account_are_kept():
    assert lichess.keeps(game("a", 1), "ce_meridian")
    assert lichess.keeps(game("a", 1), "CE_Meridian")
    assert not lichess.keeps(game("a", 1, variant="chess960"), "ce_meridian")
    assert not lichess.keeps(game("a", 1, status="started"), "ce_meridian")
    assert not lichess.keeps(game("a", 1, status="aborted"), "ce_meridian")
    assert not lichess.keeps(game("a", 1, moves=""), "ce_meridian")
    assert not lichess.keeps(game("a", 1, white="x", black="y"), "ce_meridian")


def test_a_sync_stores_new_games_once_and_asks_only_for_later_ones(tmp_path):
    seen = []
    first = lichess.sync(
        tmp_path, "ce_meridian", opener=opener_of([game("b", 20), game("a", 10)], seen)
    )
    assert [g["id"] for g in first] == ["a", "b"]
    assert "since" not in seen[0]
    second = lichess.sync(
        tmp_path, "ce_meridian", opener=opener_of([game("b", 20), game("c", 30)], seen)
    )
    assert [g["id"] for g in second] == ["c"]
    assert "since=21" in seen[1]
    stored = lichess.stored(lichess.games_file(tmp_path, "ce_meridian"))
    assert [g["id"] for g in stored] == ["a", "b", "c"]


def test_a_failed_request_is_an_error_of_the_module(tmp_path):
    def opener(request, timeout):
        raise OSError("no network")

    with pytest.raises(lichess.LichessError):
        lichess.sync(tmp_path, "ce_meridian", opener=opener)


def test_the_analysed_games_are_remembered(tmp_path):
    assert lichess.analysed(tmp_path, "ce_meridian") == set()
    lichess.mark_analysed(tmp_path, "ce_meridian", ["a", "b"])
    lichess.mark_analysed(tmp_path, "ce_meridian", ["c"])
    assert lichess.analysed(tmp_path, "ce_meridian") == {"a", "b", "c"}


def test_the_side_the_result_and_the_positions_of_a_game():
    white = game("a", 1)
    black = game("b", 1, white="other", black="ce_meridian", winner="black")
    assert lichess.side_of(white, "ce_meridian") is True
    assert lichess.side_of(black, "ce_meridian") is False
    assert lichess.result_of(white) == 1.0
    assert lichess.result_of(black) == 0.0
    assert lichess.result_of(game("c", 1, winner=None)) == 0.5
    boards = list(lichess.boards_of(white))
    assert len(boards) == 4
    assert [m.uci() if m else None for _, m in boards] == ["e2e4", "e7e5", "g1f3", None]
