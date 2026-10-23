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


import chess
import pytest

from meristats import experience, zobrist
from meristats.protocols import reanalyse
from meristats.uci import GoResult, Info


class Searcher:
    """
    A stand-in for the engine: it answers every position from a table of `fen -> (move, kind, value)`.
    """

    def __init__(self, answers):
        self.answers = answers
        self.asked = []

    def new_game(self):
        pass

    def position(self, fen, moves):
        self.fen = fen

    def go(self, command, timeout=0):
        move, kind, value = self.answers[self.fen]
        self.asked.append((self.fen, command))
        return GoResult(move, [Info(depth=12, score_kind=kind, score=value)], [], 0.0)


def game(moves, winner=None, white="me", black="you", game_id="g1"):
    return {
        "id": game_id,
        "moves": moves,
        "winner": winner,
        "players": {"white": {"user": {"name": white}}, "black": {"user": {"name": black}}},
    }


def answers_for(moves, table):
    """
    The table of answers for the positions of a game: `table` maps the ply to (move, kind, value).
    """
    board = chess.Board()
    found = {}

    for ply, san in enumerate(moves.split() + [None]):
        if ply in table:
            found[board.fen(en_passant="fen")] = table[ply]

        if san:
            board.push_san(san)

    return found


def test_a_move_that_gives_away_material_is_a_mistake_of_the_difference_of_the_scores():
    moves = "e4 e5 Nf3"
    table = {
        0: ("e2e4", "cp", 30),
        1: ("e7e5", "cp", -30),
        2: ("g1f3", "cp", 200),
        3: ("b8c6", "cp", -190),
    }
    result = reanalyse.analyse_game(Searcher(answers_for(moves, table)), game(moves), "me", 12)
    rows = result["rows"]
    assert [r["mine"] for r in rows] == [True, False, True, False]
    assert rows[0]["loss"] == 0
    assert rows[1]["loss"] is None
    assert rows[2]["loss"] == 0
    assert result["id"] == "g1"


def test_the_loss_is_the_score_before_plus_the_score_after():
    moves = "e4 e5 Qh5"
    table = {
        0: ("e2e4", "cp", 30),
        1: ("e7e5", "cp", -30),
        2: ("g1f3", "cp", 40),
        3: ("g7g6", "cp", 300),
    }
    rows = reanalyse.analyse_game(Searcher(answers_for(moves, table)), game(moves), "me", 12)[
        "rows"
    ]
    assert rows[2]["played"] == "d1h5"
    assert rows[2]["loss"] == 40 + 300


def test_a_checkmate_ends_the_game_and_counts_as_the_worst_score_for_the_side_that_is_mated():
    moves = "f3 e5 g4 Qh4#"
    table = {
        0: ("e2e4", "cp", 0),
        1: ("e7e5", "cp", 0),
        2: ("d2d4", "cp", 0),
        3: ("d8h4", "mate", 1),
    }
    rows = reanalyse.analyse_game(
        Searcher(answers_for(moves, table)), game(moves, "black"), "me", 12
    )["rows"]
    assert rows[-1]["terminal"] and rows[-1]["kind"] == "mate"
    assert rows[2]["loss"] == reanalyse.CAP_CP


def test_the_lines_of_a_game_are_its_quiet_positions_with_the_score_of_white():
    moves = "e4 e5"
    table = {0: ("g1f3", "cp", 30), 1: ("g8f6", "cp", 20), 2: ("g1f3", "cp", 25)}
    result = reanalyse.analyse_game(
        Searcher(answers_for(moves, table)), game(moves, "white"), "me", 12
    )
    lines = reanalyse.lines_of(result, 4).splitlines()
    fields = [line.split("\t") for line in lines]
    assert [f[1:] for f in fields] == [["1", "4", "30"], ["1", "4", "-20"], ["1", "4", "25"]]


def test_a_capture_as_the_best_move_is_not_a_quiet_position():
    moves = "e4 d5"
    table = {0: ("g1f3", "cp", 30), 1: ("d7d5", "cp", -30), 2: ("e4d5", "cp", 40)}
    result = reanalyse.analyse_game(Searcher(answers_for(moves, table)), game(moves), "me", 12)
    assert [r["quiet"] for r in result["rows"]] == [True, True, False]


def test_the_experience_has_the_positions_where_the_account_was_to_move():
    moves = "e4 e5"
    table = {0: ("e2e4", "cp", 30), 1: ("e7e5", "cp", -20), 2: ("g1f3", "mate", 3)}
    rows = reanalyse.analyse_game(Searcher(answers_for(moves, table)), game(moves), "me", 12)[
        "rows"
    ]
    mine = [r for r in rows if r["mine"] and not r["terminal"]]
    assert len(mine) == 2
    assert mine[0]["key"] == zobrist.engine_key(chess.Board())
    assert mine[0]["best_data"] == zobrist.move_data(chess.Board(), chess.Move.from_uci("e2e4"))
    assert experience.engine_score(mine[1]["kind"], mine[1]["value"]) == 29995


def test_an_illegal_best_move_is_an_error():
    moves = "e4"
    table = {0: ("e2e5", "cp", 30), 1: ("e7e5", "cp", 0)}

    with pytest.raises(reanalyse.ReanalyseError):
        reanalyse.analyse_game(Searcher(answers_for(moves, table)), game(moves), "me", 12)
