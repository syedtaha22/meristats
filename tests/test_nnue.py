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

import struct

import numpy as np
import pytest

from meristats import nnue
from meristats.nnue import Network, NetworkError

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def blank(hidden=nnue.HIDDEN) -> Network:
    return Network(
        np.zeros((nnue.INPUTS, hidden), np.int16),
        np.zeros(hidden, np.int16),
        np.zeros((nnue.BUCKETS, 2, hidden), np.int16),
        np.zeros(nnue.BUCKETS, np.int32),
    )


def random_network(seed=1, hidden=nnue.HIDDEN) -> Network:
    rng = np.random.default_rng(seed)
    return Network(
        rng.integers(-60, 60, (nnue.INPUTS, hidden)).astype(np.int16),
        rng.integers(-20, 20, hidden).astype(np.int16),
        rng.integers(-90, 90, (nnue.BUCKETS, 2, hidden)).astype(np.int16),
        rng.integers(-3000, 3000, nnue.BUCKETS).astype(np.int32),
    )


def test_the_board_is_read_with_a1_as_square_zero():
    pieces, side = nnue.parse_board(START)

    assert side == 0 and len(pieces) == 32
    assert (5, 4) in pieces and (11, 60) in pieces  # the kings on e1 and e8
    assert (0, 8) in pieces and (6, 48) in pieces  # a pawn on a2 and on a7
    assert nnue.parse_board("8/8/8/8/8/8/8/K6k b - - 0 1")[1] == 1


@pytest.mark.parametrize(
    "perspective,king,expected",
    [
        (0, 4, (0, 1)),  # e1: the board is mirrored, and the king stays in the lower half
        (0, 0, (0, 0)),  # a1
        (0, 7, (0, 0 + 1)),  # h1 mirrors to a1
        (0, 32, (1, 0)),  # a5 is in the upper half
        (1, 60, (0, 1)),  # the black king on e8 sees e1
        (1, 56, (0, 0)),
    ],
)
def test_the_king_decides_the_bucket_and_the_mirror(perspective, king, expected):
    assert nnue.king_context(perspective, king) == expected


def test_every_piece_of_a_position_has_its_own_input_below_the_limit():
    pieces, _ = nnue.parse_board(START)

    for perspective, king in ((0, 4), (1, 60)):
        bucket, mirror = nnue.king_context(perspective, king)
        inputs = [nnue.king_feature(perspective, p, s, bucket, mirror) for p, s in pieces]

        assert len(set(inputs)) == len(inputs) and all(0 <= i < nnue.INPUTS for i in inputs)


def test_black_sees_the_board_flipped_and_the_sides_exchanged():
    white = nnue.king_feature(0, 0, 8, 0, 0)  # a white pawn on a2, seen by White
    black = nnue.king_feature(1, 6, 48, 0, 0)  # a black pawn on a7, seen by Black

    assert white == black == 8


def test_the_output_layer_follows_the_number_of_pieces():
    assert [nnue.bucket_of(n) for n in (2, 5, 6, 9, 10, 17, 30, 32, 40, 1)] == [
        0, 0, 1, 1, 2, 3, 7, 7, 7, 0,
    ]  # fmt: skip


def test_a_network_round_trips_through_its_file(tmp_path):
    network = random_network()
    path = tmp_path / "net.nnue"
    nnue.write_network(network, path)
    back = nnue.read_network(path)
    data = path.read_bytes()

    assert data[:8] == b"MRDNNUE3"
    assert struct.unpack("<6i", data[8:32]) == (1536, 128, 255, 128, 400, 8)
    assert len(data) == 32 + 2 * (1536 * 128 + 128 + 8 * 2 * 128) + 4 * 8
    for name in ("feature_weights", "feature_bias", "output_weights", "output_bias"):
        assert np.array_equal(getattr(back, name), getattr(network, name))


def test_a_file_of_another_size_is_read_with_its_own_width(tmp_path):
    network = random_network(hidden=64)
    nnue.write_network(network, tmp_path / "n")

    assert nnue.read_network(tmp_path / "n").hidden == 64


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda d: b"XRDNNUE3" + d[8:], "not a network file"),
        (lambda d: d[:8] + struct.pack("<i", 7) + d[12:], "another build"),
        (lambda d: d[:20] + struct.pack("<i", 99) + d[24:], "another build"),
        (lambda d: d[:-4], "bytes"),
    ],
)
def test_a_file_that_the_engine_would_refuse_is_refused(tmp_path, mutate, message):
    nnue.write_network(random_network(), tmp_path / "n")
    (tmp_path / "bad").write_bytes(mutate((tmp_path / "n").read_bytes()))

    with pytest.raises(NetworkError, match=message):
        nnue.read_network(tmp_path / "bad")


def test_a_network_of_only_a_bias_gives_that_bias_in_centipawns_truncated_toward_zero():
    for bias, expected in ((255 * 128, 400), (-255 * 128, -400), (10_000, 122), (-10_000, -122)):
        network = blank()
        network.output_bias[:] = bias

        assert nnue.evaluate(network, START) == expected


def test_the_evaluation_is_the_clipped_sum_over_the_pieces():
    network = blank()
    network.feature_bias[:] = 300  # clipped to 255 in every unit
    network.output_weights[:, 0, :] = 1
    network.output_weights[:, 1, :] = 2
    value = (255 * 128 * 1 + 255 * 128 * 2) * 400 // (255 * 128)

    assert nnue.evaluate(network, START) == value == 1200
    network.feature_bias[:] = -5  # clipped to 0
    assert nnue.evaluate(network, START) == 0


def test_the_side_to_move_and_the_other_side_use_their_own_output_weights():
    network = blank()
    network.feature_bias[:] = 255
    network.output_weights[:, 0, :] = 1  # weights of the side to move
    first = nnue.evaluate(network, START)
    second = nnue.evaluate(network, START.replace(" w ", " b "))

    assert first == second == 400 // 1 * 1


def test_a_piece_adds_its_row_to_the_accumulator_of_its_own_point_of_view_only():
    fen = "4k3/8/8/8/8/8/P7/4K3 w - - 0 1"
    network = blank()
    network.output_weights[:, :, :] = 1
    bucket, mirror = nnue.king_context(0, 4)
    network.feature_weights[nnue.king_feature(0, 0, 8, bucket, mirror)] = 255

    assert nnue.evaluate(network, fen) == 400  # the side to move is White
    assert nnue.evaluate(network, fen.replace(" w ", " b ")) == 400  # and as the other side
    network.output_weights[:, 0, :] = 0
    assert nnue.evaluate(network, fen) == 0
    assert nnue.evaluate(network, fen.replace(" w ", " b ")) == 400


def test_quantizing_rounds_half_away_from_zero_and_clips():
    weights = np.array([[0.5 / 255, -0.5 / 255, 0.4 / 255, 1000.0, -1000.0]], np.float32)
    network = nnue.quantize(
        weights,
        np.zeros(5, np.float32),
        np.zeros((8, 2, 5), np.float32),
        np.array([1.0] + [0.0] * 7, np.float32),
    )

    assert network.feature_weights.tolist() == [[1, -1, 0, 32767, -32767]]
    assert network.output_bias[0] == 255 * 128


def test_quantized_weights_come_back_within_one_step_as_floats():
    network = random_network()
    weights, bias, output, output_bias = nnue.to_float(network)
    again = nnue.quantize(weights, bias, output, output_bias)

    assert np.array_equal(again.feature_weights, network.feature_weights)
    assert np.array_equal(again.output_weights, network.output_weights)
    assert np.array_equal(again.output_bias, network.output_bias)


def test_the_source_has_the_layout_of_the_file_of_the_engine():
    network = blank(hidden=4)
    network.feature_weights[0, :] = [-10, -1, 18, -6]
    network.output_bias[:] = np.arange(8)
    text = nnue.source_text(network)
    lines = text.splitlines()

    assert lines[0].startswith("// A network written by meristats: 1536 inputs, 4 values")
    assert "constexpr int16_t kBuiltinFeatureWeights[] = {" in text
    assert "    -10, -1, 18, -6, 0, 0, 0, 0, 0, 0, 0, 0," in lines
    assert text.endswith("constexpr int32_t kBuiltinOutputBias[] = {0, 1, 2, 3, 4, 5, 6, 7};\n")
    assert all(len(line) <= 100 for line in lines)


@pytest.mark.slow
def test_the_integer_evaluation_equals_the_engine_exactly(tmp_path, monkeypatch):
    import os
    import subprocess
    from pathlib import Path

    from meristats.build import build_engine
    from meristats.config import Config
    from meristats.protocols import job
    from meristats.store import Store
    from meristats.uci import UciEngine
    from test_training import positions

    repo = os.environ.get("MERISTATS_ENGINE_REPO")

    if not repo:
        pytest.skip("MERISTATS_ENGINE_REPO does not name an engine repository")

    config = Config(
        engine_repo=Path(repo), data_dir=tmp_path / "data", cache_dir=tmp_path / "cache", workers=2
    )

    with Store(config.data_dir) as store:
        engine = build_engine(config, store, "HEAD")

    monkeypatch.setattr(job, "TOOLS", (*job.TOOLS, "nnue_eval"))
    program = job.build_tool(config, engine, "nnue_eval", Path(__file__).parent / "cpp")
    network = tmp_path / "net.nnue"
    nnue.write_network(random_network(seed=4), network)
    fens = positions(300, seed=7) + [
        START,
        "4k3/8/8/8/8/8/8/4K3 b - - 0 1",
    ]
    done = subprocess.run(
        [str(program), str(network)], input="\n".join(fens) + "\n", capture_output=True, text=True
    )
    engine_values = [int(line) for line in done.stdout.split()]
    mine = [nnue.evaluate(nnue.read_network(network), fen) for fen in fens]

    assert done.returncode == 0 and len(engine_values) == len(fens)
    assert engine_values == mine and len(set(mine)) > 50

    with UciEngine(engine.binary).start() as uci:
        replies = uci.set_option("EvalFile", str(network))

        assert any("evaluation: network" in reply for reply in replies)
        uci.position(None, [])
        assert uci.go("depth 2").bestmove


def test_check_is_found_as_python_chess_finds_it_in_random_positions():
    import random

    import chess

    rng = random.Random(3)
    checks = 0

    for _ in range(4000):
        board = chess.Board(None)
        squares = rng.sample(range(64), rng.randint(2, 14))
        board.set_piece_at(squares[0], chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(squares[1], chess.Piece(chess.KING, chess.BLACK))

        for square in squares[2:]:
            kind = rng.choice([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
            board.set_piece_at(square, chess.Piece(kind, rng.random() < 0.5))

        board.turn = rng.random() < 0.5
        pieces, side = nnue.parse_board(board.fen())
        found = nnue.side_in_check(pieces, side)
        checks += found

        assert found == board.is_check(), board.fen()

    assert 300 < checks < 3700


def test_a_board_with_too_many_squares_or_a_wrong_piece_is_refused():
    for fen in ("9/8/8/8/8/8/8/8 w - - 0 1", "8/8/8/8/8/8/8/8/8 w - - 0 1", "x7/8/8/8/8/8/8/8 w"):
        with pytest.raises((ValueError, IndexError)):
            nnue.parse_board(fen)
