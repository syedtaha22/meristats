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
The network file of the engine: its layout, its input features, and its integer evaluation.

The engine loads a network with the option EvalFile and refuses a file whose magic or header does
not match its build, so a layout that differs is found at the first load.
"""

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = b"MRDNNUE3"
BASE_INPUTS = 768
KING_BUCKETS = 2
INPUTS = BASE_INPUTS * KING_BUCKETS
HIDDEN = 128
BUCKETS = 8
QUANT_INPUT = 255
QUANT_OUTPUT = 128
OUTPUT_SCALE = 400
MAX_PIECES = 32

PIECES = "PNBRQKpnbrqk"


class NetworkError(ValueError):
    pass


@dataclass
class Network:
    """
    The weights of a network in the integers that the engine uses.
    """

    feature_weights: np.ndarray  # int16, INPUTS x hidden
    feature_bias: np.ndarray  # int16, hidden
    output_weights: np.ndarray  # int16, BUCKETS x 2 x hidden
    output_bias: np.ndarray  # int32, BUCKETS

    @property
    def hidden(self) -> int:
        return self.feature_bias.shape[0]


def king_context(perspective: int, king_square: int) -> tuple[int, int]:
    """
    The king bucket and the mirror of one point of view. Black sees the board flipped, and a king
    on the files e to h mirrors the board so that it stands on the files a to d.
    """
    f = king_square if perspective == 0 else king_square ^ 56
    mirror = 1 if (f & 7) >= 4 else 0
    m = f ^ 7 if mirror else f
    return (1 if (m >> 3) >= 4 else 0), mirror


def king_feature(perspective: int, piece: int, square: int, bucket: int, mirror: int) -> int:
    """
    The input of a piece on a square for one point of view. A piece is `color * 6 + type`.
    """
    slot = piece if perspective == 0 else (piece + 6) % 12
    sq = (square if perspective == 0 else square ^ 56) ^ (7 if mirror else 0)
    return bucket * BASE_INPUTS + slot * 64 + sq


def bucket_of(pieces: int) -> int:
    """
    The output layer for a number of pieces: 2 to 5 pieces the first, 6 to 9 the second, and so on.
    """
    b = (pieces - 2) // (32 // BUCKETS)
    return 0 if b < 0 else (BUCKETS - 1 if b >= BUCKETS else b)


def parse_board(fen: str) -> tuple[list[tuple[int, int]], int]:
    """
    The pieces of a position as (piece, square) pairs, with a1 as square 0, and the side to move,
    0 for White. The piece is `color * 6 + type`, with the pawn, knight, bishop, rook, queen and
    king in this order.
    """
    fields = fen.split()
    pieces = []
    ranks = fields[0].split("/")

    if len(ranks) != 8:
        raise ValueError(f"the board of {fen!r} has {len(ranks)} ranks")

    for index, text in enumerate(ranks):
        rank, file = 7 - index, 0

        for char in text:
            if char.isdigit():
                file += int(char)
            else:
                if file > 7:
                    raise ValueError(f"the board of {fen!r} has too many squares")

                pieces.append((PIECES.index(char), rank * 8 + file))
                file += 1

        if file != 8:
            raise ValueError(f"the board of {fen!r} has a rank of {file} squares")

    return pieces, 0 if fields[1] == "w" else 1


KNIGHT_STEPS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
ROOK_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))
BISHOP_STEPS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def side_in_check(pieces: list[tuple[int, int]], side: int) -> bool:
    """
    Whether the king of `side` is attacked, from the pieces of a position as `parse_board` gives
    them. The squares are numbered with a1 as 0.
    """
    board = {square: piece for piece, square in pieces}
    king = next((square for piece, square in pieces if piece == side * 6 + 5), None)

    if king is None:
        return False

    enemy = 6 * (1 - side)
    file, rank = king % 8, king // 8

    def at(f: int, r: int) -> int | None:
        return board.get(r * 8 + f) if 0 <= f < 8 and 0 <= r < 8 else None

    forward = 1 if side == 0 else -1  # the pawns of the enemy stand on this side of the king

    for df in (-1, 1):
        if at(file + df, rank + forward) == enemy:
            return True

    for df, dr in KNIGHT_STEPS:
        if at(file + df, rank + dr) == enemy + 1:
            return True

    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if (df or dr) and at(file + df, rank + dr) == enemy + 5:
                return True

    for steps, kinds in (
        (ROOK_STEPS, (enemy + 3, enemy + 4)),
        (BISHOP_STEPS, (enemy + 2, enemy + 4)),
    ):
        for df, dr in steps:
            f, r = file + df, rank + dr

            while 0 <= f < 8 and 0 <= r < 8:
                piece = board.get(r * 8 + f)

                if piece is not None:
                    if piece in kinds:
                        return True

                    break

                f, r = f + df, r + dr

    return False


def read_network(path: Path) -> Network:
    """
    Reads a network file. The header of the file has the sizes and the scales of the build that
    wrote it, and the scales have to be those of the engine.
    """
    data = Path(path).read_bytes()

    if data[:8] != MAGIC:
        raise NetworkError(f"{path} is not a network file")

    inputs, hidden, quant_input, quant_output, output_scale, buckets = struct.unpack(
        "<6i", data[8:32]
    )

    if (inputs, quant_input, quant_output, output_scale, buckets) != (
        INPUTS,
        QUANT_INPUT,
        QUANT_OUTPUT,
        OUTPUT_SCALE,
        BUCKETS,
    ):
        raise NetworkError(f"{path} has the sizes or the scales of another build")

    expected = 32 + 2 * (INPUTS * hidden + hidden + BUCKETS * 2 * hidden) + 4 * BUCKETS

    if len(data) != expected:
        raise NetworkError(f"{path} has {len(data)} bytes and {expected} are expected")

    offset = 32
    weights = np.frombuffer(data, np.int16, INPUTS * hidden, offset).reshape(INPUTS, hidden)
    offset += 2 * INPUTS * hidden
    bias = np.frombuffer(data, np.int16, hidden, offset)
    offset += 2 * hidden
    output = np.frombuffer(data, np.int16, BUCKETS * 2 * hidden, offset).reshape(BUCKETS, 2, hidden)
    offset += 2 * BUCKETS * 2 * hidden
    output_bias = np.frombuffer(data, np.int32, BUCKETS, offset)
    return Network(weights.copy(), bias.copy(), output.copy(), output_bias.copy())


def write_network(network: Network, path: Path) -> None:
    """
    Writes a network file in the layout that the engine reads.
    """
    header = struct.pack(
        "<6i", INPUTS, network.hidden, QUANT_INPUT, QUANT_OUTPUT, OUTPUT_SCALE, BUCKETS
    )
    Path(path).write_bytes(
        MAGIC
        + header
        + network.feature_weights.astype("<i2").tobytes()
        + network.feature_bias.astype("<i2").tobytes()
        + network.output_weights.astype("<i2").tobytes()
        + network.output_bias.astype("<i4").tobytes()
    )


def evaluate(network: Network, fen: str) -> int:
    """
    The evaluation of a position in centipawns for the side to move, with the integer arithmetic
    of the engine.
    """
    pieces, side = parse_board(fen)
    white_king = next(square for piece, square in pieces if piece == 5)
    black_king = next(square for piece, square in pieces if piece == 11)
    accumulators = []

    for perspective, king in ((0, white_king), (1, black_king)):
        bucket, mirror = king_context(perspective, king)
        total = network.feature_bias.astype(np.int64).copy()

        for piece, square in pieces:
            total += network.feature_weights[
                king_feature(perspective, piece, square, bucket, mirror)
            ]

        accumulators.append(np.clip(total, 0, QUANT_INPUT))

    us, them = accumulators[side], accumulators[1 - side]
    layer = bucket_of(len(pieces))
    total = int(network.output_bias[layer])
    total += int(us @ network.output_weights[layer, 0].astype(np.int64))
    total += int(them @ network.output_weights[layer, 1].astype(np.int64))
    # C++ integer division truncates toward zero.
    scaled = total * OUTPUT_SCALE
    quotient = abs(scaled) // (QUANT_INPUT * QUANT_OUTPUT)
    return quotient if scaled >= 0 else -quotient


def quantize(
    feature_weights: np.ndarray,
    feature_bias: np.ndarray,
    output_weights: np.ndarray,
    output_bias: np.ndarray,
) -> Network:
    """
    Turns floating point weights into the integers of a network. The accumulator has the scale
    QUANT_INPUT, the weights of the output layer have the scale QUANT_OUTPUT, and a value rounds
    half away from zero, as the engine's tools round.
    """

    def scaled(values: np.ndarray, scale: float) -> np.ndarray:
        x = values.astype(np.float64) * scale
        return np.clip(np.sign(x) * np.floor(np.abs(x) + 0.5), -32767, 32767).astype(np.int16)

    x = output_bias.astype(np.float64) * QUANT_INPUT * QUANT_OUTPUT
    bias = (np.sign(x) * np.floor(np.abs(x) + 0.5)).astype(np.int32)
    return Network(
        scaled(feature_weights, QUANT_INPUT),
        scaled(feature_bias, QUANT_INPUT),
        scaled(output_weights, QUANT_OUTPUT),
        bias,
    )


def to_float(network: Network) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    The weights of a network as floating point numbers, to start a training from it.
    """
    return (
        network.feature_weights.astype(np.float32) / QUANT_INPUT,
        network.feature_bias.astype(np.float32) / QUANT_INPUT,
        network.output_weights.astype(np.float32) / QUANT_OUTPUT,
        network.output_bias.astype(np.float32) / (QUANT_INPUT * QUANT_OUTPUT),
    )


def source_text(network: Network) -> str:
    """
    The network as C++ source for the engine's `src/nnue_net.inc`, 12 numbers to a line.
    """

    def numbers(name: str, values: np.ndarray, kind: str = "int16_t") -> str:
        flat = [int(v) for v in values.reshape(-1)]
        lines = []

        for start in range(0, len(flat), 12):
            lines.append("   " + "".join(f" {v}," for v in flat[start : start + 12]))

        return f"constexpr {kind} {name}[] = {{\n" + "\n".join(lines) + "\n};\n\n"

    return (
        f"// A network written by meristats: {INPUTS} inputs, {network.hidden} values for each "
        f"side,\n// {BUCKETS} output layers.\n\n"
        + numbers("kBuiltinFeatureWeights", network.feature_weights)
        + numbers("kBuiltinFeatureBias", network.feature_bias)
        + numbers("kBuiltinOutputWeights", network.output_weights)
        + "constexpr int32_t kBuiltinOutputBias[] = {"
        + ", ".join(str(int(v)) for v in network.output_bias)
        + "};\n"
    )
