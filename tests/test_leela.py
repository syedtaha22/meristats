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


import gzip
import io
import tarfile

import chess
import numpy as np
import pytest

from meristats import leela

LISTING = """<html><body><pre><a href="../">../</a>
<a href="README.txt">README.txt</a>                         20-Jul-2026 16:14        3781
<a href="training-r-0000.tar">training-r-0000.tar</a>       20-Jul-2026 17:15        100
<a href="training-r-0001.tar">training-r-0001.tar</a>       20-Jul-2026 17:18        200
<a href="training-r-0002.tar">training-r-0002.tar</a>       20-Jul-2026 17:20        300
<a href="training-r-0003.tar">training-r-0003.tar</a>       20-Jul-2026 17:21        400
</pre></body></html>"""


def record(board, root_q=0.2, best_q=0.4, result_q=1.0, halfmove=0):
    """
    One position as Lc0 stores it: the pieces of the side to move first, with the board flipped
    when Black is to move.
    """
    black = board.turn == chess.BLACK
    rng = np.random.default_rng(len(board.move_stack))
    data = np.zeros(leela.RECORD, dtype=np.uint8)
    data[8 : leela.PLANES_AT] = rng.integers(0, 256, leela.PLANES_AT - 8, dtype=np.uint8)  # policy
    planes = np.zeros(12, dtype="<u8")

    for square, piece in board.piece_map().items():
        place = square ^ 56 if black else square
        ours = piece.color == board.turn
        planes[(0 if ours else 6) + piece.piece_type - 1] |= np.uint64(1) << np.uint64(place)

    data[leela.PLANES_AT : leela.PLANES_AT + 96] = np.frombuffer(planes.tobytes(), dtype=np.uint8)
    us, them = board.turn, not board.turn
    data[leela.CASTLING_AT : leela.CASTLING_AT + 4] = [
        board.has_queenside_castling_rights(us),
        board.has_kingside_castling_rights(us),
        board.has_queenside_castling_rights(them),
        board.has_kingside_castling_rights(them),
    ]
    data[leela.SIDE_AT] = 1 if black else 0
    data[leela.HALFMOVE_AT] = halfmove
    values = np.zeros(15, dtype="<f4")
    values[0], values[1], values[7] = root_q, best_q, result_q
    data[leela.VALUES_AT : leela.VALUES_AT + 60] = np.frombuffer(values.tobytes(), dtype=np.uint8)
    return data.tobytes()


def game(moves, **values):
    """
    The boards of a game and its records.
    """
    board = chess.Board()
    boards = [board.copy()]

    for san in moves.split():
        board.push_san(san)
        boards.append(board.copy())

    return boards, b"".join(record(b, **values) for b in boards)


QUIET = "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7"


def fields(line):
    fen, result, number, score = line.rstrip("\n").split("\t")
    placement, side, castling, _, halfmove, _ = fen.split()
    return placement, side, castling, int(halfmove), float(result), int(number), int(score)


def test_the_positions_of_a_game_are_read_back_as_the_board_of_white():
    boards, raw = game(QUIET)
    lines, count = leela.decode_game(raw, 7, every=1, min_ply=0)

    assert count == len(boards) == 11 and len(lines) == 10

    for line, board in zip(lines, boards, strict=False):
        expected = board.fen().split()
        assert fields(line)[:3] == (expected[0], expected[1], expected[2])


def test_the_score_is_from_the_search_and_the_result_from_the_game():
    _, raw = game(QUIET, root_q=0.2, best_q=0.4, result_q=1.0)
    lines, _ = leela.decode_game(raw, 7, every=1, min_ply=0)
    white, black = fields(lines[0]), fields(lines[1])

    assert (white[1], white[4], white[5], white[6]) == ("w", 1.0, 7, 86)
    assert (black[1], black[4], black[5], black[6]) == ("b", 0.0, 7, -86)


def test_every_and_the_first_plies_thin_the_positions_out():
    _, raw = game(QUIET)
    lines, _ = leela.decode_game(raw, 1, every=2, min_ply=4)

    assert len(lines) == 3 and [fields(x)[1] for x in lines] == ["w", "w", "w"]


def test_a_position_before_a_capture_or_a_promotion_is_not_kept():
    boards, raw = game("e4 d5 exd5 Qxd5")
    lines, _ = leela.decode_game(raw, 1, every=1, min_ply=0)

    assert len(boards) == 5 and len(lines) == 2


def test_castling_rights_that_the_board_cannot_have_drop_the_position():
    board = chess.Board("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1")
    raw = record(board)
    broken = np.frombuffer(raw, dtype=np.uint8).copy()
    broken[leela.CASTLING_AT + 1] = 1  # a kingside right, with no rook on h1
    lines, _ = leela.decode_game(raw + raw, 1, every=1, min_ply=0)
    none, _ = leela.decode_game(broken.tobytes() * 2, 1, every=1, min_ply=0)

    assert len(lines) == 1 and fields(lines[0])[2] == "Q"
    assert none == []


def write_tar(path, games):
    with tarfile.open(path, "w") as tar:
        for index, raw in enumerate(games):
            data = gzip.compress(raw)
            member = tarfile.TarInfo(f"game_{index}.gz")
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))


def test_tars_are_converted_with_game_numbers_that_tell_the_tar_and_the_place(tmp_path):
    _, raw = game(QUIET)
    write_tar(tmp_path / "a.tar", [raw, raw])
    write_tar(tmp_path / "b.tar", [raw])
    done = []
    out = tmp_path / "out.txt"
    counts = leela.convert(
        out,
        [tmp_path / "a.tar", tmp_path / "b.tar"],
        every=1,
        min_ply=0,
        workers=2,
        progress=done.append,
    )
    numbers = sorted({fields(line)[5] for line in out.read_text().splitlines()})

    assert counts == {"tars": 2, "games": 3, "read": 33, "kept": 30}
    assert numbers == [0, 1, 100000] and len(done) == 2


def test_a_tar_that_ends_in_the_middle_gives_the_games_before_it(tmp_path):
    _, raw = game(QUIET)
    write_tar(tmp_path / "a.tar", [raw] * 6)
    data = (tmp_path / "a.tar").read_bytes()
    (tmp_path / "cut.tar").write_bytes(data[: len(data) * 3 // 5])
    lines, positions, games = leela.convert_tar((str(tmp_path / "cut.tar"), 0, 1, 0, 0))

    assert 1 <= games < 6 and positions == 11 * games and len(lines) == 10 * games


def test_the_index_lists_the_tars_with_their_sizes():
    opener = lambda request, timeout: io.BytesIO(LISTING.encode())  # noqa: E731

    assert leela.list_tars("r", opener) == [
        ("training-r-0000.tar", 100),
        ("training-r-0001.tar", 200),
        ("training-r-0002.tar", 300),
        ("training-r-0003.tar", 400),
    ]


def test_a_sample_is_spread_over_the_whole_run():
    tars = [(f"t{i}", i) for i in range(100)]

    assert [name for name, _ in leela.pick(tars, 4)] == ["t12", "t37", "t62", "t87"]
    assert leela.pick(tars, 200) == tars


class Server:
    def __init__(self, sizes):
        self.sizes = sizes
        self.asked = []

    def __call__(self, request, timeout):
        self.asked.append(request.full_url)

        if request.full_url.endswith("/"):
            return io.BytesIO(LISTING.encode())

        name = request.full_url.rsplit("/", 1)[1]
        return io.BytesIO(b"x" * self.sizes[name])


def test_a_fetch_downloads_the_sample_and_skips_what_is_there(tmp_path):
    sizes = {
        "training-r-0000.tar": 100,
        "training-r-0001.tar": 200,
        "training-r-0002.tar": 300,
        "training-r-0003.tar": 400,
    }
    first = Server(sizes)
    paths = leela.fetch("r", tmp_path, 2, opener=first)

    assert [p.name for p in paths] == ["training-r-0001.tar", "training-r-0003.tar"]
    assert [p.stat().st_size for p in paths] == [200, 400] and not list(tmp_path.glob("*.part"))
    second = Server(sizes)
    leela.fetch("r", tmp_path, 2, opener=second)

    assert len(second.asked) == 1  # the index only


def test_a_download_with_the_wrong_size_is_refused_and_left_as_a_part_file(tmp_path):
    server = Server(
        {
            "training-r-0000.tar": 1,
            "training-r-0001.tar": 5,
            "training-r-0002.tar": 3,
            "training-r-0003.tar": 4,
        }
    )

    with pytest.raises(leela.LeelaError, match="has 5 bytes, and the index says 200"):
        leela.fetch("r", tmp_path, 2, opener=server)

    assert not (tmp_path / "training-r-0001.tar").exists()
    assert (tmp_path / "training-r-0001.tar.part").exists()


def test_a_folder_without_tars_is_an_error():
    opener = lambda request, timeout: io.BytesIO(b"<html>nothing</html>")  # noqa: E731

    with pytest.raises(leela.LeelaError, match="has no tars"):
        leela.list_tars("nothing", opener)
