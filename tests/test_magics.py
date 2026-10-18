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
import re
from pathlib import Path

import numpy as np
import pytest

from meristats import magics
from meristats.magics import BISHOP, ROOK, Random


def test_the_relevant_squares_leave_out_the_edge_of_each_ray():
    assert magics.relevant_mask(0, ROOK).bit_count() == 12
    assert magics.relevant_mask(27, ROOK).bit_count() == 10
    assert magics.relevant_mask(0, BISHOP).bit_count() == 6
    assert magics.relevant_mask(27, BISHOP).bit_count() == 9
    assert magics.relevant_mask(0, ROOK) & (1 << 7) == 0  # the end of the first rank
    assert magics.relevant_mask(0, ROOK) & (1 << 1)


def test_a_slider_attacks_along_each_ray_up_to_and_including_the_first_blocker():
    assert magics.walk(0, 0, ROOK).bit_count() == 14
    assert magics.walk(27, 0, BISHOP).bit_count() == 13
    blocked = magics.walk(0, 1 << 3, ROOK)

    assert blocked & (1 << 3) and not blocked & (1 << 4)
    assert magics.walk(0, 1 << 24, ROOK) & (1 << 24)


def test_the_generator_gives_the_same_sequence_every_time():
    first, second = Random(), Random()

    assert [first.next() for _ in range(5)] == [second.next() for _ in range(5)]
    assert all(Random().sparse().bit_count() < 40 for _ in range(10))


@pytest.mark.parametrize("square,directions", [(0, ROOK), (27, BISHOP), (63, BISHOP)])
def test_a_magic_number_maps_every_occupancy_to_a_slot_without_a_clash(square, directions):
    magic = magics.find_magic(square, directions, Random())
    mask = magics.relevant_mask(square, directions)
    bits = mask.bit_count()
    table = {}
    subset = 0

    for _ in range(1 << bits):
        slot = ((subset * magic) & magics.MASK64) >> (64 - bits)
        attacks = magics.walk(square, subset, directions)

        assert table.setdefault(slot, attacks) == attacks
        subset = (subset - mask) & mask


def test_the_numbers_are_written_as_the_source_of_the_engine():
    rook, bishop = list(range(64)), [0xABCDEF0123456789] * 64
    text = magics.source_text(rook, bishop)

    assert "constexpr uint64_t kRookMagics[kSquares] = {" in text
    assert "    0x0000000000000000ULL, 0x0000000000000001ULL, 0x0000000000000002ULL," in text
    assert text.count("0xabcdef0123456789ULL,") == 64
    assert text.endswith("};\n")
    assert all(len(line) < 100 for line in text.splitlines())


@pytest.mark.slow
def test_the_numbers_are_those_of_the_engine():
    repo = os.environ.get("MERISTATS_ENGINE_REPO")

    if not repo or not (Path(repo) / "src" / "magic_numbers.inc").exists():
        pytest.skip("MERISTATS_ENGINE_REPO does not name an engine repository")

    text = (Path(repo) / "src" / "magic_numbers.inc").read_text()
    expected = [int(x, 16) for x in re.findall(r"0x([0-9a-fA-F]{16})ULL", text)]
    rook, bishop = magics.find_all()

    assert rook + bishop == expected
    assert np.array(rook + bishop, np.uint64).size == 128
