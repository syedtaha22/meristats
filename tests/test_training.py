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

import random

import chess
import numpy as np
import pytest

from meristats import nnue, training
from meristats.training import TrainingError

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
IN_CHECK = "4k3/8/8/8/8/8/4q3/4K3 w - - 0 1"


def positions(count, seed=1):
    """
    Random legal positions that are not in check, from short random games.
    """
    rng = random.Random(seed)
    found = []

    while len(found) < count:
        board = chess.Board()

        for _ in range(rng.randint(4, 60)):
            moves = list(board.legal_moves)

            if not moves:
                break

            board.push(rng.choice(moves))

            if not board.is_check() and not board.is_game_over():
                found.append(board.fen())

    return found[:count]


def write_data(path, count=300, seed=1):
    rng = random.Random(seed)
    lines = []

    for index, fen in enumerate(positions(count, seed)):
        lines.append(
            f"{fen}\t{rng.choice(['0', '0.5', '1'])}\t{index // 6}\t{rng.randint(-300, 300)}"
        )

    path.write_text("\n".join(lines) + "\n")
    return lines


def test_the_target_mixes_the_expected_points_of_the_score_with_the_result():
    assert training.target_of(0, 1.0, 0, 0.7) == pytest.approx(0.7 * 0.5 + 0.3 * 1.0)
    assert training.target_of(0, 1.0, 1, 0.7) == pytest.approx(0.7 * 0.5 + 0.3 * 0.0)
    assert training.target_of(400, 0.5, 0, 1.0) == pytest.approx(
        1 / (1 + 10 ** (-1.25 * 400 / 400))
    )
    assert training.target_of(400, 0.5, 1, 1.0) == pytest.approx(1 / (1 + 10 ** (1.25 * 400 / 400)))
    assert training.target_of(100, 0.0, 0, 0.0) == 0.0


def test_a_data_file_is_read_into_arrays(tmp_path):
    path = tmp_path / "d.txt"
    path.write_text(f"{START}\t1\t3\t30\n")
    data = training.read_dataset(path, workers=1)

    assert len(data) == 1 and data.pieces.shape == (1, 32) and data.pieces.dtype == np.int16
    assert data.count[0] == 32 and data.context[0].tolist() == [0, 0, 0, 1, 1]
    assert data.pieces[0, 0] == 3 * 64 + 0  # the white rook on a1 comes first
    assert data.pieces[0, 31] == 9 * 64 + 63  # the black rook on h8 comes last
    assert data.target[0] == pytest.approx(training.target_of(30, 1.0, 0, 0.7))
    assert not data.held[0]


def test_the_pieces_are_in_the_order_of_the_squares_and_padded_with_minus_one(tmp_path):
    path = tmp_path / "d.txt"
    path.write_text("4k3/8/8/8/8/8/P7/4K3 w - - 0 1\t0.5\t1\t0\n")
    data = training.read_dataset(path, workers=1)

    assert data.count[0] == 3
    row = data.pieces[0]
    assert (row[:3] // 64).tolist() == [5, 0, 11] and (row[:3] % 64).tolist() == [4, 8, 60]
    assert (row[3:] == -1).all()


def test_lines_without_a_quiet_score_are_left_out(tmp_path):
    path = tmp_path / "d.txt"
    path.write_text(
        "\n".join(
            [
                f"{START}\t1\t1\t10",  # kept
                f"{START}\t1\t1\t3000",  # a score beyond the limit
                f"{IN_CHECK}\t1\t1\t10",  # in check
                f"{START}\t1\t1\t",  # no score
                f"{START}\t1\t1",  # too few fields
                "not a fen\t1\t1\t10",  # not a position
                f"{START}\t0\t2\t-2500",  # at the limit
            ]
        )
        + "\n"
    )
    data = training.read_dataset(path, workers=1)

    assert len(data) == 2 and data.read == 7 and data.skipped == 5


def test_every_fifth_game_is_held_out(tmp_path):
    path = tmp_path / "d.txt"
    lines = write_data(path, count=60)
    data = training.read_dataset(path, workers=1)
    games = [int(line.split("\t")[2]) for line in lines]

    assert data.held.tolist() == [g % 5 == 0 for g in games]
    assert data.held_fens == [
        line.split("\t")[0] for line in lines if int(line.split("\t")[2]) % 5 == 0
    ]


def test_a_limit_keeps_the_first_positions(tmp_path):
    path = tmp_path / "d.txt"
    write_data(path, count=60)
    full = training.read_dataset(path, workers=1)
    part = training.read_dataset(path, workers=1, limit=25)

    assert len(part) == 25 and np.array_equal(part.pieces, full.pieces[:25])
    assert np.array_equal(part.target, full.target[:25])


def test_chunks_of_the_file_give_the_same_arrays_as_one_read(tmp_path, monkeypatch):
    path = tmp_path / "d.txt"
    write_data(path, count=120)
    whole = training.read_dataset(path, workers=1)

    for size in (97, 256, 1000):
        monkeypatch.setattr(training, "CHUNK_BYTES", size)
        chunked = training.read_dataset(path, workers=3)

        assert len(chunked) == len(whole) == 120
        assert np.array_equal(chunked.pieces, whole.pieces)
        assert np.array_equal(chunked.target, whole.target)
        assert np.array_equal(chunked.held, whole.held)


def test_a_file_without_positions_is_an_error(tmp_path):
    (tmp_path / "e.txt").write_text("")

    with pytest.raises(TrainingError, match="no positions"):
        training.read_dataset(tmp_path / "e.txt", workers=1)


# ----------------------------------------------------------------------------- the training


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def small_data(tmp_path, count=1500):
    write_data(tmp_path / "d.txt", count=count)
    return training.read_dataset(tmp_path / "d.txt", workers=2)


def test_training_lowers_the_held_out_loss_and_the_network_is_a_valid_file(torch, tmp_path):
    data = small_data(tmp_path)
    result = training.train(
        data, epochs=4, batch=64, lr=0.01, hidden=16, device="cpu", log=lambda text: None
    )
    nnue.write_network(result.network, tmp_path / "n.nnue")
    back = nnue.read_network(tmp_path / "n.nnue")

    assert result.best_held == min(e["held"] for e in result.epochs)
    assert result.epochs[-1]["train"] < result.epochs[0]["train"]
    assert back.hidden == 16 and result.checked == len(data.held_fens) > 0
    assert result.quantized_difference < 60


def test_the_same_seed_gives_the_same_network_on_the_cpu(torch, tmp_path):
    data = small_data(tmp_path)
    runs = [
        training.train(
            data, epochs=2, batch=64, lr=0.01, hidden=8, seed=5, device="cpu", log=lambda t: None
        )
        for _ in range(2)
    ]

    assert np.array_equal(runs[0].network.feature_weights, runs[1].network.feature_weights)
    assert np.array_equal(runs[0].network.output_bias, runs[1].network.output_bias)


def test_a_network_can_be_the_start_of_a_training(torch, tmp_path):
    data = small_data(tmp_path)
    first = training.train(
        data, epochs=2, batch=64, lr=0.01, hidden=8, device="cpu", log=lambda t: None
    )
    second = training.train(
        data, epochs=1, batch=64, lr=0.0001, init=first.network, device="cpu", log=lambda t: None
    )

    assert second.network.hidden == 8
    assert second.epochs[0]["held"] <= first.best_held * 1.2


def test_too_few_positions_are_an_error(torch, tmp_path):
    data = small_data(tmp_path, count=40)

    with pytest.raises(TrainingError, match="too few"):
        training.train(data, epochs=1, batch=1024, device="cpu", log=lambda t: None)


def test_the_float_forward_matches_the_integer_evaluation_closely(torch, tmp_path):
    data = small_data(tmp_path)
    result = training.train(
        data, epochs=3, batch=64, lr=0.01, hidden=16, device="cpu", log=lambda t: None
    )

    assert result.quantized_difference < 5 + 0.05 * result.quantized_magnitude


def test_training_runs_on_the_gpu_when_there_is_one(torch, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("there is no GPU")

    data = small_data(tmp_path)
    result = training.train(
        data, epochs=3, batch=64, lr=0.01, hidden=16, device="cuda", log=lambda text: None
    )

    assert result.device == "cuda"
    assert result.epochs[-1]["train"] < result.epochs[0]["train"]
    assert result.quantized_difference < 60
    assert result.best_held == min(e["held"] for e in result.epochs)
