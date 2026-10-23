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


import os
import random
import struct
import subprocess
from pathlib import Path

import chess
import pytest

from meristats import zobrist


def test_the_key_depends_on_the_side_the_castling_rights_and_the_pieces():
    board = chess.Board()
    start = zobrist.engine_key(board)
    assert (
        zobrist.engine_key(chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"))
        != start
    )
    assert (
        zobrist.engine_key(chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w Qkq - 0 1"))
        != start
    )
    board.push_uci("e2e4")
    assert zobrist.engine_key(board) != start


def test_the_hash_of_a_position_does_not_depend_on_the_moves_that_reached_it():
    one = chess.Board()
    two = chess.Board()

    for move in ("g1f3", "g8f6", "b1c3", "b8c6"):
        one.push_uci(move)

    for move in ("b1c3", "b8c6", "g1f3", "g8f6"):
        two.push_uci(move)

    assert zobrist.engine_key(one) == zobrist.engine_key(two)


def test_an_en_passant_square_counts_only_when_a_pawn_can_capture():
    board = chess.Board()
    board.push_uci("e2e4")
    without = zobrist.engine_key(board)
    capturable = chess.Board("rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3")
    fen_without = chess.Board("rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 3")
    assert zobrist.engine_key(capturable) != zobrist.engine_key(fen_without)
    assert without == zobrist.engine_key(chess.Board(board.fen(en_passant="legal")))


@pytest.mark.parametrize(
    ("fen", "uci", "flag"),
    [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e3", zobrist.QUIET),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4", zobrist.DOUBLE_PUSH),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", zobrist.KING_CASTLE),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1", zobrist.QUEEN_CASTLE),
        ("4k3/8/8/8/8/8/p7/4K3 b - - 0 1", "a2a1q", 11),
        ("4k3/8/8/8/8/8/p7/1R2K3 b - - 0 1", "a2b1n", 12),
        (
            "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3",
            "d4e3",
            zobrist.EN_PASSANT,
        ),
        ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", zobrist.CAPTURE),
    ],
)
def test_a_move_is_packed_with_its_flag(fen, uci, flag):
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    data = zobrist.move_data(board, move)
    assert data & 63 == move.from_square
    assert (data >> 6) & 63 == move.to_square
    assert data >> 12 == flag


@pytest.mark.slow
def test_the_keys_and_the_moves_are_the_ones_of_the_engine(tmp_path):
    from meristats.build import build_engine
    from meristats.config import Config
    from meristats.store import Store

    repo = os.environ.get("MERISTATS_ENGINE_REPO")

    if not repo:
        pytest.skip("MERISTATS_ENGINE_REPO does not name an engine repository")

    config = Config(
        engine_repo=Path(repo), data_dir=tmp_path / "data", cache_dir=tmp_path / "cache", workers=2
    )

    with Store(config.data_dir) as store:
        engine = build_engine(config, store, "HEAD")

    rng = random.Random(5)
    checked = 0

    for _ in range(25):
        board = chess.Board()

        for _ in range(rng.randint(8, 90)):
            moves = list(board.legal_moves)

            if not moves:
                break

            special = [
                m for m in moves if m.promotion or board.is_castling(m) or board.is_en_passant(m)
            ]
            board.push(rng.choice(special) if special and rng.random() < 0.5 else rng.choice(moves))

        moves = list(board.legal_moves)

        if not moves:
            continue

        move = rng.choice(moves)
        before = chess.Board()
        played = [m.uci() for m in board.move_stack]

        for m in board.move_stack:
            before.push(m)

        path = tmp_path / "experience.bin"
        path.write_bytes(
            b"MRDEXP01"
            + struct.pack("<Q", 1)
            + struct.pack(
                "<QHhH", zobrist.engine_key(board), zobrist.move_data(board, move), 17, 99
            )
        )
        commands = (
            f"setoption name ExperienceFile value {path}\n"
            f"position startpos moves {' '.join(played)}\ngo depth 2\nquit\n"
        )
        out = subprocess.run(
            [engine.binary], input=commands, capture_output=True, text=True, check=True
        ).stdout
        assert "info string experience move" in out
        assert f"bestmove {move.uci()}" in out
        checked += 1

    assert checked >= 15
