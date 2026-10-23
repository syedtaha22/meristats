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


import pytest

from meristats import experience


def test_a_file_is_written_in_the_order_of_the_keys_and_read_back(tmp_path):
    entries = {30: (5, -12, 20), 10: (7, 40, 18), 20: (9, 29990, 30)}
    path = tmp_path / "e.bin"
    experience.write(path, entries)
    data = path.read_bytes()
    assert data[:8] == b"MRDEXP01"
    assert len(data) == 16 + 14 * 3
    assert [int.from_bytes(data[16 + 14 * i : 24 + 14 * i], "little") for i in range(3)] == [
        10,
        20,
        30,
    ]
    assert experience.read(path) == entries
    assert not list(tmp_path.glob("*.part"))


def test_the_deeper_analysis_of_a_position_is_kept_in_a_merge():
    old = {1: (1, 10, 20), 2: (2, 20, 20)}
    new = {1: (9, 99, 25), 2: (8, 88, 15), 3: (3, 30, 12)}
    assert experience.merge(old, new) == {1: (9, 99, 25), 2: (2, 20, 20), 3: (3, 30, 12)}
    assert experience.merge({1: (1, 10, 20)}, {1: (4, 40, 20)}) == {1: (4, 40, 20)}


def test_a_file_that_is_not_an_experience_file_is_refused(tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(b"short")

    with pytest.raises(experience.ExperienceError):
        experience.read(path)

    path.write_bytes(b"NOTMAGIC" + bytes(8))

    with pytest.raises(experience.ExperienceError):
        experience.read(path)

    experience.write(tmp_path / "ok.bin", {1: (1, 1, 1)})
    (tmp_path / "cut.bin").write_bytes((tmp_path / "ok.bin").read_bytes()[:-1])

    with pytest.raises(experience.ExperienceError):
        experience.read(tmp_path / "cut.bin")


def test_a_uci_score_becomes_the_score_of_the_engine():
    assert experience.engine_score("cp", 35) == 35
    assert experience.engine_score("cp", 99999) == 29000
    assert experience.engine_score("mate", 1) == 29999
    assert experience.engine_score("mate", 3) == 29995
    assert experience.engine_score("mate", -1) == -29998
    assert experience.engine_score("mate", -2) == -29996
