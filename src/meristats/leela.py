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
The training games of Leela Chess Zero as positions for the trainer.

The games are published as tars of the V7 format, one gzip file for each game and 8396 bytes for
each position, in folders of https://storage.lczero.org/files/training_data. `fetch` downloads
an evenly spread sample of the tars of a folder, and `convert` turns tars into lines
`FEN<TAB>result<TAB>game<TAB>score`, the format of the `positions` protocol, where the result is the
points of White and the score is in centipawns for the side to move.

A record holds twelve bitboards, the pieces of the side to move (P N B R Q K) and then those of
the other side, with a1 as bit 0 and the side to move at the bottom, so that the board of a
record of a black move is flipped. The score is the mean of the root value and the best value of
the search (side to move, -1 to 1), turned into centipawns on the curve that the trainer uses
(base 10 over 400 / 1.25). A position is dropped when

- it is among the first `min_ply` plies, or is not one of every `every` plies
- a capture or a promotion follows, since the played move is not quiet
- it has castling rights without the king and the rook on their squares (a game of Fischer
  random chess), which a standard position cannot hold
- its score is beyond 2500 centipawns

The data is published under the Open Database License (ODbL).
"""

import gzip
import math
import multiprocessing
import re
import tarfile
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np

INDEX = "https://storage.lczero.org/files/training_data/"
RECORD = 8396  # bytes of one position
PLANES_AT = 8 + 1858 * 4  # the twelve bitboards
CASTLING_AT = 8272  # four bytes: us queenside, us kingside, them queenside, them kingside
SIDE_AT = 8276  # 1 when Black is to move
HALFMOVE_AT = 8277
VALUES_AT = 8280  # 15 floats, the root value, the best value and the result among them
PIECES = "PNBRQK"
SHIFTS = np.arange(64, dtype=np.uint64)
MAX_SCORE = 2500

# Castling flag: the squares of the king and of the rook, with their letters.
CASTLING = {
    "K": (4, "K", 7, "R"),
    "Q": (4, "K", 0, "R"),
    "k": (60, "k", 63, "r"),
    "q": (60, "k", 56, "r"),
}


class LeelaError(RuntimeError):
    pass


def board_letters(row: np.ndarray, black: bool) -> list[str]:
    """
    The piece letter of each square of a board, as seen by White.

    `row` holds the plane of the piece on each square, or -1. A plane below 6 is a piece of the
    side to move, which is Black when `black` is set.
    """
    letters = []

    for plane in row:
        if plane < 0:
            letters.append("")
            continue

        white = (plane < 6) != black
        letter = PIECES[plane % 6]
        letters.append(letter if white else letter.lower())

    return letters


def castling_field(record: np.ndarray, letters: list[str], black: bool) -> str | None:
    """
    The castling field of a FEN, or None when a right does not match the board.
    """
    us_queen, us_king, them_queen, them_king = record[CASTLING_AT : CASTLING_AT + 4]
    rights = {
        "K": them_king if black else us_king,
        "Q": them_queen if black else us_queen,
        "k": us_king if black else them_king,
        "q": us_queen if black else them_queen,
    }
    field = ""

    for flag in "KQkq":
        if not rights[flag]:
            continue

        king_square, king, rook_square, rook = CASTLING[flag]

        if letters[king_square] != king or letters[rook_square] != rook:
            return None

        field += flag

    return field or "-"


def placement(letters: list[str]) -> str:
    """
    The piece placement field of a FEN.
    """
    ranks = []

    for rank in range(7, -1, -1):
        text = ""
        empty = 0

        for file in range(8):
            letter = letters[rank * 8 + file]

            if letter:
                text += (str(empty) if empty else "") + letter
                empty = 0
            else:
                empty += 1

        ranks.append(text + (str(empty) if empty else ""))

    return "/".join(ranks)


def decode_game(raw: bytes, game_id: int, every: int, min_ply: int) -> tuple[list[str], int]:
    """
    The lines of one game and the number of its positions.
    """
    records = np.frombuffer(raw, dtype=np.uint8).reshape(-1, RECORD)
    count = len(records)

    if count < 2:
        return [], 0

    planes = np.ascontiguousarray(records[:, PLANES_AT : PLANES_AT + 96])
    planes = planes.view("<u8").reshape(count, 12)
    bits = ((planes[:, :, None] >> SHIFTS) & np.uint64(1)).astype(bool)  # (count, 12, 64)
    occupied = bits.any(axis=1)
    kind = np.where(occupied, bits.argmax(axis=1), -1).astype(np.int8)  # plane of each square
    side = records[:, SIDE_AT]

    # The board as seen by White: the ranks of a record where Black is to move are flipped.
    kind = kind.reshape(count, 8, 8)
    flip = side == 1
    kind[flip] = kind[flip][:, ::-1, :]
    kind = kind.reshape(count, 64)

    pieces = occupied.sum(axis=1)
    pawns = bits[:, 0, :].sum(axis=1) + bits[:, 6, :].sum(axis=1)

    values = np.ascontiguousarray(records[:, VALUES_AT : VALUES_AT + 60])
    values = values.view("<f4").reshape(count, 15)
    root_q, best_q, result_q = values[:, 0], values[:, 1], values[:, 7]
    lines = []

    for i in range(count - 1):
        if i < min_ply or i % every:
            continue

        if pieces[i + 1] < pieces[i] or pawns[i + 1] < pawns[i]:
            continue  # a capture or a promotion follows

        black = bool(side[i])
        letters = board_letters(kind[i], black)
        castling = castling_field(records[i], letters, black)

        if castling is None:
            continue

        q = 0.5 * (float(root_q[i]) + float(best_q[i]))
        p = min(max((q + 1.0) / 2.0, 0.0005), 0.9995)
        side_cp = 320.0 * math.log10(p / (1.0 - p))

        if abs(side_cp) > MAX_SCORE:
            continue

        points = (float(result_q[i]) + 1.0) / 2.0
        white_result = (1.0 - points) if black else points
        white_cp = -side_cp if black else side_cp
        fen = "%s %s %s - %d 1" % (
            placement(letters),
            "b" if black else "w",
            castling,
            int(records[i, HALFMOVE_AT]),
        )
        lines.append("%s\t%g\t%d\t%d\n" % (fen, white_result, game_id, round(white_cp)))

    return lines, count


def convert_tar(job: tuple[str, int, int, int, int]) -> tuple[list[str], int, int]:
    """
    The lines, the number of positions and the number of games of one tar. A tar that ends in the
    middle of a game, as a download that is not finished does, gives the games before it.
    """
    path, tar_index, every, min_ply, limit_games = job
    lines: list[str] = []
    positions = 0
    games = 0

    try:
        with tarfile.open(path, "r|") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".gz"):
                    continue

                try:
                    raw = gzip.decompress(tar.extractfile(member).read())
                except (EOFError, OSError):
                    break

                if len(raw) % RECORD:
                    continue

                # The game number is the index of the tar and the place in it, so that a fifth of
                # the games can be held out by number, whatever the order of the workers.
                game_id = tar_index * 100000 + games
                game_lines, count = decode_game(raw, game_id, every, min_ply)
                lines.extend(game_lines)
                positions += count
                games += 1

                if limit_games and games >= limit_games:
                    break
    except (tarfile.ReadError, EOFError):
        pass

    return lines, positions, games


def convert(
    out: Path,
    tars: list[Path],
    *,
    every: int = 2,
    min_ply: int = 6,
    limit_games: int = 0,
    workers: int = 1,
    progress: Callable[[str], None] = lambda text: None,
) -> dict:
    """
    Writes the positions of the tars to `out` and returns the counts. The tars are read by
    `workers` processes, which start fresh (a fork would copy the threads of numpy).
    """
    jobs = [(str(tar), index, every, min_ply, limit_games) for index, tar in enumerate(tars)]
    kept = total = games = 0
    context = multiprocessing.get_context("forkserver")

    with context.Pool(max(1, min(len(jobs), workers))) as pool, open(out, "w") as handle:
        for done, (lines, positions, count) in enumerate(pool.imap_unordered(convert_tar, jobs), 1):
            handle.writelines(lines)
            kept += len(lines)
            total += positions
            games += count
            progress(f"{done} of {len(jobs)} tars, {kept} positions kept")

    return {"tars": len(tars), "games": games, "read": total, "kept": kept}


ROW = re.compile(r'<a href="([^"]+\.tar)">[^<]*</a>\s+\S+ \S+\s+(\d+)')


def list_tars(run: str, opener: Callable = urllib.request.urlopen) -> list[tuple[str, int]]:
    """
    The tars of a folder of the training data, as (name, bytes), in the order of the index.
    """
    request = urllib.request.Request(f"{INDEX}{run}/", headers={"User-Agent": "meristats"})

    try:
        with opener(request, timeout=60) as reply:
            page = reply.read().decode("utf-8", "replace")
    except OSError as error:
        raise LeelaError(f"cannot read the index of {run}: {error}") from error

    tars = [(name, int(size)) for name, size in ROW.findall(page)]

    if not tars:
        raise LeelaError(f"{run} has no tars: is it a folder of {INDEX}?")

    return tars


def pick(tars: list[tuple[str, int]], count: int) -> list[tuple[str, int]]:
    """
    `count` tars spread evenly over the list, so that a sample holds games of the whole run.
    """
    if count >= len(tars):
        return list(tars)

    return [tars[(2 * i + 1) * len(tars) // (2 * count)] for i in range(count)]


def fetch(
    run: str,
    directory: Path,
    count: int,
    *,
    opener: Callable = urllib.request.urlopen,
    progress: Callable[[str], None] = lambda text: None,
) -> list[Path]:
    """
    Downloads `count` tars of a folder, spread evenly, into `directory`, and returns their paths.
    A tar that is there with the size of the index is not downloaded again, and a download writes
    to a `.part` file that becomes the tar only when its size is right.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []

    for number, (name, size) in enumerate(pick(list_tars(run, opener), count), 1):
        path = directory / name
        paths.append(path)

        if path.exists() and path.stat().st_size == size:
            progress(f"{number} of {count}: {name} is there")
            continue

        part = path.with_name(name + ".part")
        request = urllib.request.Request(
            f"{INDEX}{run}/{name}", headers={"User-Agent": "meristats"}
        )

        try:
            with opener(request, timeout=120) as reply, open(part, "wb") as handle:
                while block := reply.read(1 << 20):
                    handle.write(block)
        except OSError as error:
            raise LeelaError(f"cannot download {name}: {error}") from error

        if part.stat().st_size != size:
            raise LeelaError(f"{name} has {part.stat().st_size} bytes, and the index says {size}")

        part.replace(path)
        progress(f"{number} of {count}: {name}, {size / 1e6:.0f} MB")

    return paths
