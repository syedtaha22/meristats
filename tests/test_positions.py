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
import stat
import sys
from pathlib import Path

import chess
import pytest

from meristats.build import BuiltEngine
from meristats.preflight import PreflightError
from meristats.protocols import positions as protocol
from meristats.uci import UciEngine

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
def player(tmp_path, monkeypatch):
    program = tmp_path / "player"
    program.write_text(f"#!{sys.executable}\n" + PLAYER)
    program.chmod(program.stat().st_mode | stat.S_IEXEC)

    def make(**environment) -> Path:
        for key, value in environment.items():
            monkeypatch.setenv(key, str(value))

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
