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
The experience file of the engine: positions that were analysed deeper than a game search goes,
each with the best move, the score and the depth of its analysis.

The engine reads the file with its option ExperienceFile and plays the move of an entry when the
analysis is deeper than its own search. The file has the text `MRDEXP01`, the number of entries
as a 64-bit integer, and the entries in order of their keys, each one 14 bytes: the position
hash, the move, the score and the depth. Numbers are little-endian.
"""

import os
import struct
from pathlib import Path

MAGIC = b"MRDEXP01"
HEADER = struct.Struct("<8sQ")
ENTRY = struct.Struct("<QHhH")
MATE_SCORE = 30000
MAX_DEPTH = 65535


class ExperienceError(ValueError):
    pass


def read(path: Path) -> dict[int, tuple[int, int, int]]:
    """
    The entries of a file as `{key: (move, score, depth)}`.
    """
    data = Path(path).read_bytes()

    if len(data) < HEADER.size:
        raise ExperienceError(f"{path} is too short")

    magic, count = HEADER.unpack_from(data)

    if magic != MAGIC:
        raise ExperienceError(f"{path} is not an experience file")

    if len(data) != HEADER.size + count * ENTRY.size:
        raise ExperienceError(f"{path} has {len(data)} bytes and {count} entries")

    entries = {}

    for offset in range(HEADER.size, len(data), ENTRY.size):
        key, move, score, depth = ENTRY.unpack_from(data, offset)
        entries[key] = (move, score, depth)

    return entries


def write(path: Path, entries: dict[int, tuple[int, int, int]]) -> None:
    """
    Writes entries, in the order of their keys. The file appears whole or not at all, because it
    is written beside the path and then renamed.
    """
    path = Path(path)
    temporary = path.with_name(path.name + ".part")

    with open(temporary, "wb") as handle:
        handle.write(HEADER.pack(MAGIC, len(entries)))

        for key in sorted(entries):
            move, score, depth = entries[key]
            handle.write(ENTRY.pack(key, move, score, min(depth, MAX_DEPTH)))

    os.replace(temporary, path)


def merge(
    old: dict[int, tuple[int, int, int]], new: dict[int, tuple[int, int, int]]
) -> dict[int, tuple[int, int, int]]:
    """
    The entries of both. A position that is in both keeps the deeper analysis, and the newer one
    when the depths are equal.
    """
    merged = dict(old)

    for key, entry in new.items():
        if key not in merged or entry[2] >= merged[key][2]:
            merged[key] = entry

    return merged


def engine_score(kind: str, value: int) -> int:
    """
    A UCI score as the engine holds it: centipawns, or a mate near 30000 that is further from it
    the more plies the mate takes. `value` of a mate counts moves, as UCI does, and is positive
    when the side to move gives the mate.
    """
    if kind == "cp":
        return max(-MATE_SCORE + 1000, min(MATE_SCORE - 1000, value))

    plies = 2 * abs(value) - (1 if value > 0 else 0)
    return (MATE_SCORE - plies) if value > 0 else -(MATE_SCORE - plies)
