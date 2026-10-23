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
The position hash and the move encoding of the engine, for files that the engine reads by hash.

The engine keys its tables by a Zobrist hash of the placement, the side to move, the castling
rights and a capturable en passant square, with keys from a fixed seed. An experience file holds
these hashes, so they have to be the engine's own.
"""

import chess

MASK = (1 << 64) - 1
SEED = 0xD1B54A32D192ED03
STEP = 0x9E3779B97F4A7C15

QUIET, DOUBLE_PUSH, KING_CASTLE, QUEEN_CASTLE, CAPTURE, EN_PASSANT = 0, 1, 2, 3, 4, 5
PROMOTION = 8


def _keys() -> tuple[list[list[int]], list[int], list[int], int]:
    state = SEED

    def draw() -> int:
        nonlocal state
        state = (state + STEP) & MASK
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    pieces = [[draw() for _ in range(64)] for _ in range(12)]
    castling = [draw() for _ in range(16)]
    en_passant = [draw() for _ in range(8)]
    return pieces, castling, en_passant, draw()


PIECE_KEYS, CASTLING_KEYS, EN_PASSANT_KEYS, SIDE_KEY = _keys()


def castling_mask(board: chess.Board) -> int:
    """
    The castling rights as the engine holds them: 1 and 2 for White's king and queen side, 4 and 8
    for Black's. They are read from the FEN, as the engine reads them.
    """
    letters = board.fen().split()[2]
    return sum(bit for letter, bit in (("K", 1), ("Q", 2), ("k", 4), ("q", 8)) if letter in letters)


def engine_key(board: chess.Board) -> int:
    """
    The hash that the engine gives to the position of the board.
    """
    key = 0

    for square, piece in board.piece_map().items():
        key ^= PIECE_KEYS[(0 if piece.color == chess.WHITE else 6) + piece.piece_type - 1][square]

    key ^= CASTLING_KEYS[castling_mask(board)]

    if board.turn == chess.BLACK:
        key ^= SIDE_KEY

    square = board.ep_square

    if square is not None:
        capturers = chess.BB_PAWN_ATTACKS[not board.turn][square] & int(
            board.pieces(chess.PAWN, board.turn)
        )

        if capturers:
            key ^= EN_PASSANT_KEYS[chess.square_file(square)]

    return key


def move_data(board: chess.Board, move: chess.Move) -> int:
    """
    The 16 bits that the engine holds for a move in the position of the board: the origin, the
    destination and a flag in the four high bits.
    """
    if move.promotion:
        flag = PROMOTION + move.promotion - chess.KNIGHT + (4 if board.is_capture(move) else 0)
    elif board.is_castling(move):
        flag = KING_CASTLE if chess.square_file(move.to_square) > 4 else QUEEN_CASTLE
    elif board.is_en_passant(move):
        flag = EN_PASSANT
    elif board.is_capture(move):
        flag = CAPTURE
    elif (
        board.piece_type_at(move.from_square) == chess.PAWN
        and abs(move.to_square - move.from_square) == 16
    ):
        flag = DOUBLE_PUSH
    else:
        flag = QUIET

    return move.from_square | (move.to_square << 6) | (flag << 12)
