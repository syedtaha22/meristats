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
The magic numbers of the slider attacks of a bitboard engine.

A magic number is a multiplier that maps every occupancy of the squares that can block a rook or a
bishop to a slot of a table of 2^bits entries, without two occupancies with different attacks
sharing a slot. The search tries sparse numbers from a fixed sequence, square by square (the rook
of each square, then its bishop), so that every run gives the same numbers.
"""

import numpy as np

MASK64 = (1 << 64) - 1
ROOK = ((1, 0), (-1, 0), (0, 1), (0, -1))
BISHOP = ((1, 1), (1, -1), (-1, 1), (-1, -1))
BATCH = 512


class Random:
    """
    A deterministic generator, so that the numbers found do not change between runs.
    """

    def __init__(self) -> None:
        self.state = 0x9E3779B97F4A7C15

    def next(self) -> int:
        s = self.state
        s ^= s >> 12
        s ^= (s << 25) & MASK64
        s ^= s >> 27
        self.state = s
        return (s * 0x2545F4914F6CDD1D) & MASK64

    def sparse(self) -> int:
        """
        A number with few set bits, which makes a better multiplier.
        """
        return self.next() & self.next() & self.next()


def walk(square: int, occupied: int, directions) -> int:
    """
    The squares that a slider attacks from `square`, along each ray until the edge or the first
    blocker.
    """
    attacks = 0

    for df, dr in directions:
        file, rank = square % 8 + df, square // 8 + dr

        while 0 <= file < 8 and 0 <= rank < 8:
            bit = 1 << (rank * 8 + file)
            attacks |= bit

            if occupied & bit:
                break

            file, rank = file + df, rank + dr

    return attacks


def relevant_mask(square: int, directions) -> int:
    """
    The squares whose occupancy can change the attacks: the rays without their last square.
    """
    mask = 0

    for df, dr in directions:
        file, rank = square % 8 + df, square // 8 + dr

        while 0 <= file + df < 8 and 0 <= rank + dr < 8:
            mask |= 1 << (rank * 8 + file)
            file, rank = file + df, rank + dr

    return mask


def find_magic(square: int, directions, random: Random) -> int:
    """
    The first multiplier of the sequence that maps every subset of the relevant squares to a slot
    of a table of 2^bits entries, without two subsets with different attacks sharing a slot.
    """
    mask = relevant_mask(square, directions)
    bits = mask.bit_count()
    size = 1 << bits
    occupancies = np.zeros(size, np.uint64)
    references = np.zeros(size, np.uint64)
    subset = 0

    for i in range(size):
        occupancies[i] = subset
        references[i] = walk(square, subset, directions)
        subset = (subset - mask) & mask  # the next subset, by the carry-rippler

    shift = np.uint64(64 - bits)

    while True:
        # A batch of candidates in the order of the sequence, each with the state after it, so
        # that the generator stops at the winner as a one-by-one search does.
        candidates, states = [], []

        while len(candidates) < BATCH:
            magic = random.sparse()

            if (((mask * magic) & MASK64) >> 56).bit_count() >= 6:
                candidates.append(magic)
                states.append(random.state)

        magics = np.array(candidates, np.uint64)
        with np.errstate(over="ignore"):
            slots = ((occupancies[None, :] * magics[:, None]) >> shift).astype(np.intp)

        table = np.zeros((len(candidates), size), np.uint64)
        rows = np.arange(len(candidates))[:, None]
        table[rows, slots] = references[None, :]
        valid = (table[rows, slots] == references[None, :]).all(axis=1)

        if valid.any():
            winner = int(np.argmax(valid))
            random.state = states[winner]
            return candidates[winner]


def find_all() -> tuple[list[int], list[int]]:
    """
    The magic numbers of the rook and of the bishop on each of the 64 squares.
    """
    random = Random()
    rook, bishop = [], []

    for square in range(64):
        rook.append(find_magic(square, ROOK, random))
        bishop.append(find_magic(square, BISHOP, random))

    return rook, bishop


def source_text(rook: list[int], bishop: list[int]) -> str:
    """
    The numbers as C++ source for the engine's `src/magic_numbers.inc`.
    """

    def array(name: str, magics: list[int]) -> str:
        text = f"constexpr uint64_t {name}[kSquares] = {{\n"

        for i, magic in enumerate(magics):
            text += (
                ("    " if i % 4 == 0 else " ")
                + f"0x{magic:016x}ULL,"
                + ("\n" if i % 4 == 3 else "")
            )

        return text + "};\n"

    return (
        "// Magic numbers of the rook and the bishop on each square, written by meristats.\n\n"
        + array("kRookMagics", rook)
        + "\n"
        + array("kBishopMagics", bishop)
    )
