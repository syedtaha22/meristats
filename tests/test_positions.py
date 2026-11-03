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

import hashlib
import random
import stat
import sys
from pathlib import Path

import chess
import pytest

from meristats.build import BuiltEngine
from meristats.preflight import PreflightError
from meristats.protocols import positions as protocol
from meristats.uci import GoResult, Info, UciEngine

PLAYER = """
import os, sys
import chess
board = chess.Board()
white_score = int(os.environ.get("PLAYER_WHITE_SCORE", "20"))
mate = os.environ.get("PLAYER_MATE")
def say(text):
    sys.stdout.write(text + "\\n"); sys.stdout.flush()
for line in sys.stdin:
    words = line.split()
    if not words:
        continue
    if words[0] == "uci":
        say("id name Player 1")
        say("option name Hash type spin default 16 min 1 max 64")
        say("option name Threads type spin default 1 min 1 max 64")
        say("option name Config type string default network")
        say("option name EvalFile type string default <empty>")
        say("uciok")
    elif words[0] == "isready":
        say("readyok")
    elif words[0] == "position":
        board = chess.Board()
        if "moves" in words:
            for move in words[words.index("moves") + 1:]:
                board.push_uci(move)
    elif words[0] == "go":
        legal = sorted(board.legal_moves, key=lambda m: m.uci())
        wanted = os.environ.get("PLAYER_PREFER")
        pick = next((m for m in legal if m.uci() == wanted), legal[0]) if wanted else legal[0]
        score = white_score if board.turn == chess.WHITE else -white_score
        text = "mate 3" if mate else "cp %d" % score
        say("info depth 5 score %s nodes 100 time 1 pv %s" % (text, pick.uci()))
        say("bestmove " + pick.uci())
    elif words[0] == "quit":
        break
"""


@pytest.fixture
def player(tmp_path):
    """
    Makes a player program. Its settings are written into the program, because the processes of a
    run do not share the environment of the test.
    """
    count = []

    def make(**environment) -> Path:
        program = tmp_path / f"player{len(count)}"
        count.append(program)
        settings = "import os\nos.environ.update(%r)\n" % {
            key: str(value) for key, value in environment.items()
        }
        program.write_text(f"#!{sys.executable}\n" + settings + PLAYER)
        program.chmod(program.stat().st_mode | stat.S_IEXEC)
        return program

    return make


def play(program, game=0, **settings):
    settings = {"seed": 1, "depth": 5, **settings}

    with UciEngine(program).start() as uci:
        return protocol.play_game(uci, game, **settings)


def test_a_game_ends_when_one_side_is_far_ahead_for_four_moves(player):
    result, kept = play(player(PLAYER_WHITE_SCORE=1500))

    assert result == 1.0
    assert len(kept) <= 4


def test_a_game_won_by_black_is_scored_zero(player):
    assert play(player(PLAYER_WHITE_SCORE=-1500))[0] == 0.0


def test_a_mate_score_is_never_kept(player):
    _, kept = play(player(PLAYER_MATE=1))

    assert kept == []


def test_positions_are_kept_every_few_plies_with_the_score_of_white(player):
    program = player(PLAYER_WHITE_SCORE=40)
    result, kept = play(program, skip=3)
    fens = [fen for fen, _ in kept]

    assert result in (0.5, 0.0, 1.0)
    assert len(kept) >= 3 and all(score == 40 for _, score in kept)
    assert len(set(fens)) == len(fens)
    assert all(chess.Board(fen).is_valid() and not chess.Board(fen).is_check() for fen in fens)


def test_a_game_is_a_function_of_its_number_and_the_seed(player):
    program = player()

    assert play(program, game=3) == play(program, game=3)
    assert play(program, game=3) != play(program, game=4)
    assert play(program, game=3, seed=1) != play(program, game=3, seed=2)


def test_a_position_with_a_capture_or_a_check_as_best_move_is_not_kept(player):
    program = player()
    _, kept = play(program, skip=1)
    kept_with_capture = [
        fen
        for fen, _ in kept
        if any(
            chess.Board(fen).is_capture(move)
            for move in [min(chess.Board(fen).legal_moves, key=lambda m: m.uci())]
        )
    ]

    assert kept_with_capture == []


def test_the_lines_have_the_fen_the_result_the_game_and_the_score():
    text = protocol.format_lines(7, 0.5, [("8/8/8/8/8/8/8/K6k w - - 0 1", -12)])

    assert text == "8/8/8/8/8/8/8/K6k w - - 0 1\t0.5\t7\t-12\n"
    assert protocol.format_lines(1, 1.0, [("f", 1)]) == "f\t1\t1\t1\n"
    assert protocol.format_lines(2, 0.0, [("f", 1)]) == "f\t0\t2\t1\n"
    assert protocol.format_lines(3, 1.0, []) == ""


def test_an_illegal_move_of_the_engine_is_an_error(player, tmp_path):
    program = player(PLAYER_PREFER="a1a1")
    bad = tmp_path / "bad"
    bad.write_text(
        program.read_text().replace(
            "pick = next", "pick = chess.Move.from_uci('a1h8') if wanted else next"
        )
    )
    bad.chmod(bad.stat().st_mode | stat.S_IEXEC)

    with pytest.raises(protocol.PositionsError, match="illegal move"):
        play(bad)


def engine_of(store, program):
    row = store.add_engine("meridian", "player", "a" * 40, "1" * 64, {}, 7, "Player 1")
    return BuiltEngine(row, "a" * 40, program, "1" * 64, 7, "Player 1", {})


def test_a_run_writes_the_games_in_order_and_records_the_file(
    config, store, player, framework_repo, tmp_path
):
    program = player(PLAYER_WHITE_SCORE=1500)
    out = tmp_path / "data" / "positions.txt"
    run_id, summary = protocol.run(
        config,
        store,
        engine_of(store, program),
        out,
        games=6,
        depth=5,
        seed=2,
        workers=2,
        framework=framework_repo,
        progress=lambda text: None,
    )
    lines = out.read_text().splitlines()
    games = [int(line.split("\t")[2]) for line in lines]
    row = store.get_run(run_id)

    assert games == sorted(games) and summary["positions"] == len(lines) > 0
    assert summary["white_wins"] == 6 and summary["draws"] == 0 and summary["black_wins"] == 0
    assert summary["output"]["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert row["protocol"] == "positions" and row["seed"] == 2 and row["grade"] == "verified"


def test_a_run_gives_the_same_file_for_the_same_seed(
    config, store, player, framework_repo, tmp_path
):
    program = player()
    engine = engine_of(store, program)
    digests = []

    for name in ("a", "b"):
        _, summary = protocol.run(
            config,
            store,
            engine,
            tmp_path / name,
            games=4,
            depth=5,
            seed=3,
            workers=2,
            framework=framework_repo,
            progress=lambda text: None,
        )
        digests.append(summary["output"]["sha256"])

    assert digests[0] == digests[1]


def test_a_network_file_is_recorded_by_its_hash(config, store, player, framework_repo, tmp_path):
    program = player()
    network = tmp_path / "net.nnue"
    network.write_bytes(b"x")
    _, summary = protocol.run(
        config,
        store,
        engine_of(store, program),
        tmp_path / "p.txt",
        games=2,
        depth=5,
        eval_file=network,
        workers=1,
        framework=framework_repo,
        progress=lambda text: None,
    )
    row = store.db.execute("SELECT limits_json FROM runs").fetchone()

    assert hashlib.sha256(b"x").hexdigest() in row["limits_json"]
    assert summary["positions"] > 0


def test_a_run_without_positions_is_refused(config, store, player, framework_repo, tmp_path):
    program = player(PLAYER_MATE=1)

    with pytest.raises(PreflightError, match="0 positions"):
        protocol.run(
            config,
            store,
            engine_of(store, program),
            tmp_path / "p.txt",
            games=2,
            depth=5,
            workers=1,
            framework=framework_repo,
            progress=lambda text: None,
        )


def test_a_run_with_a_time_limit_stops_and_counts_the_games_that_it_finished(
    config, store, player, framework_repo, tmp_path
):
    out = tmp_path / "data" / "limited.txt"
    run_id, summary = protocol.run(
        config,
        store,
        engine_of(store, player()),
        out,
        games=100000,
        depth=3,
        workers=2,
        max_seconds=1,
        framework=framework_repo,
        progress=lambda text: None,
    )

    assert 1 <= summary["games"] < 100000
    assert len(out.read_text().splitlines()) == summary["positions"]
    assert store.get_run(run_id)["status"] == "complete"


class Scripted:
    """
    An engine that answers every search with the next score of a list, and records what it was
    asked.
    """

    def __init__(self, scores, bestmove="e2e4"):
        self.scores = list(scores)
        self.bestmove = bestmove
        self.commands = []
        self.played = []

    def new_game(self):
        pass

    def position(self, fen, moves=()):
        self.played.append(list(moves))

    def go(self, command):
        self.commands.append(command)
        score = self.scores.pop(0) if self.scores else 0
        info = Info(depth=1, score_kind="cp", score=score)
        return GoResult(self.bestmove, [info], [], 0.0)


def test_a_book_is_read_as_its_first_plies_without_repeats(tmp_path):
    book = tmp_path / "book.pgn"
    book.write_text(
        '[Event "a"]\n\n1. e4 e5 2. Nf3 Nc6 *\n\n[Event "b"]\n\n1. e4 e5 2. Nf3 Nc6 *\n\n'
        '[Event "c"]\n\n1. d4 d5 2. c4 e6 3. Nc3 *\n'
    )

    assert protocol.load_book(book, plies=3) == (
        ("e2e4", "e7e5", "g1f3"),
        ("d2d4", "d7d5", "c2c4"),
    )


def test_an_opening_from_the_book_continues_with_at_most_two_random_moves():
    book = (("e2e4", "e7e5", "g1f3", "b8c6"),)
    lengths = set()

    for game in range(40):
        rng = random.Random(game)
        moves, _ = protocol.draw_opening(
            Scripted([]), rng, book=book, book_share=1.0, min_random_plies=4, random_plies=8
        )
        assert moves[:4] == list(book[0])
        lengths.add(len(moves))

    assert lengths == {4, 5, 6}


def test_a_random_opening_has_a_length_between_the_bounds():
    lengths = set()

    for game in range(60):
        moves, _ = protocol.draw_opening(
            Scripted([]), random.Random(game), book=(), book_share=1.0,
            min_random_plies=4, random_plies=8,
        )
        lengths.add(len(moves))

    assert lengths == {4, 5, 6, 7, 8}


def test_a_lopsided_opening_is_drawn_again():
    uci = Scripted([900, -700, 50])
    moves, _ = protocol.draw_opening(
        uci, random.Random(1), book=(), book_share=0.0, min_random_plies=4, random_plies=8
    )

    assert len(uci.commands) == 3 and uci.scores == []
    assert all(command == "depth 6" for command in uci.commands)


def test_the_last_opening_stands_when_every_try_is_lopsided():
    uci = Scripted([900] * 20)
    protocol.draw_opening(
        uci, random.Random(1), book=(), book_share=0.0, min_random_plies=4, random_plies=8
    )

    assert len(uci.commands) == protocol.OPENING_TRIES - 1


def test_another_move_is_played_only_when_a_shallow_search_scores_it_near_the_best():
    board = chess.Board()
    best = chess.Move.from_uci("e2e4")

    near = Scripted([-30])
    move = protocol.other_move(
        near, random.Random(1), board, [], best, 20, depth=9, margin=60
    )

    assert move != best and near.commands[0].startswith("depth 6 searchmoves ")
    assert near.commands[0].endswith(move.uci())

    far = Scripted([-500, -500, -500])

    assert protocol.other_move(
        far, random.Random(1), board, [], best, 20, depth=9, margin=60
    ) == best
    assert len(far.commands) == protocol.EXPLORE_TRIES


def test_exploring_games_with_the_same_opening_part_ways(player):
    program = player(PLAYER_WHITE_SCORE=30)

    with UciEngine(program).start() as uci:
        plain = protocol.play_game(uci, 0, seed=1, depth=5, explore=0.0, skip=1)
        explored = protocol.play_game(uci, 0, seed=1, depth=5, explore=1.0, skip=1)

    assert plain != explored


def test_a_position_is_written_once_in_a_run(config, store, player, framework_repo, tmp_path):
    program = player(PLAYER_WHITE_SCORE=1500)
    out = tmp_path / "p.txt"
    _, summary = protocol.run(
        config,
        store,
        engine_of(store, program),
        out,
        games=8,
        depth=5,
        random_plies=0,
        min_random_plies=0,
        explore=0.0,
        skip=1,
        workers=2,
        framework=framework_repo,
        progress=lambda text: None,
    )
    keys = [line.split("\t")[0].rsplit(" ", 2)[0] for line in out.read_text().splitlines()]

    assert len(keys) == len(set(keys)) == summary["positions"]
    assert summary["duplicates"] > 0


def test_a_book_is_recorded_by_its_hash(config, store, player, framework_repo, tmp_path):
    book = tmp_path / "book.pgn"
    book.write_text('[Event "a"]\n\n1. e4 e5 2. Nf3 Nc6 *\n')
    _, summary = protocol.run(
        config,
        store,
        engine_of(store, player()),
        tmp_path / "p.txt",
        games=2,
        depth=5,
        book=book,
        book_share=1.0,
        workers=1,
        framework=framework_repo,
        progress=lambda text: None,
    )
    row = store.db.execute("SELECT limits_json FROM runs").fetchone()

    assert hashlib.sha256(book.read_bytes()).hexdigest() in row["limits_json"]
    assert summary["positions"] > 0


def test_ctrl_c_ends_a_run_and_keeps_the_games_that_were_finished(
    config, store, player, framework_repo, tmp_path, monkeypatch
):
    out = tmp_path / "p.txt"
    written = []
    original = protocol.format_lines

    def format_lines(game, result, kept):
        if written:
            raise KeyboardInterrupt

        written.append(game)
        return original(game, result, kept)

    monkeypatch.setattr(protocol, "format_lines", format_lines)
    _, summary = protocol.run(
        config,
        store,
        engine_of(store, player(PLAYER_WHITE_SCORE=1500)),
        out,
        games=6,
        depth=5,
        workers=2,
        framework=framework_repo,
        progress=lambda text: None,
    )

    assert summary["interrupted"] is True and summary["games"] == 1
    assert out.read_text().count("\t") == 3 * summary["positions"]
