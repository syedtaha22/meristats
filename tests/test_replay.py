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

import pytest

from meristats import replay

FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def lines(count, tag, games=5):
    return "".join(f"{FEN}\t1\t{i % games}\t{tag}{i}\n" for i in range(count))


def test_a_night_is_stored_once_and_read_back(tmp_path):
    buffer = replay.Buffer(tmp_path)
    path = buffer.add("selfplay", "2026-10-23", lines(7, "s"))
    assert path.name == "2026-10-23.txt.zst"
    assert len(list(replay.read_lines(path))) == 7

    with pytest.raises(replay.ReplayError):
        buffer.add("selfplay", "2026-10-23", "other")

    with pytest.raises(replay.ReplayError):
        buffer.add("unknown", "2026-10-23", "x")


def test_the_files_of_the_nights_before_are_listed_oldest_first(tmp_path):
    buffer = replay.Buffer(tmp_path)

    for night in ("2026-10-22", "2026-10-21", "2026-10-23"):
        buffer.add("selfplay", night, lines(2, night))

    names = [p.name for p in buffer.files(["selfplay"], before="2026-10-23")]
    assert names == ["2026-10-21.txt.zst", "2026-10-22.txt.zst"]


def test_a_sample_is_uniform_and_has_the_size_that_was_asked_for():
    rng = random.Random(3)
    found = replay.sample((str(i) for i in range(10000)), 100, rng)
    assert len(found) == 100 and len(set(found)) == 100
    assert 3000 < sum(int(x) for x in found) / 100 < 7000
    assert replay.sample(iter(["a", "b"]), 5, rng) == ["a", "b"]


def test_the_training_file_has_the_night_the_history_the_mistakes_and_the_anchor(tmp_path):
    buffer = replay.Buffer(tmp_path / "buffer")
    buffer.add("selfplay", "2026-10-22", lines(40, "old"))
    buffer.add("selfplay", "2026-10-23", lines(10, "new"))
    buffer.add("lichess", "2026-10-23", lines(5, "game"))
    buffer.add("mistakes", "2026-10-22", lines(2, "mistake"))
    anchor = tmp_path / "anchor.txt"
    anchor.write_text(lines(30, "anchor", games=7))
    out = tmp_path / "train.txt"
    counts = replay.assemble(
        out,
        buffer,
        "2026-10-23",
        anchor=anchor,
        history_lines=20,
        anchor_lines=12,
        mistake_lines=10,
        mistake_repeat=3,
    )
    assert counts == {"night": 15, "history": 20, "mistakes": 6, "anchor": 12}
    rows = [line.split("\t") for line in out.read_text().splitlines()]
    assert len(rows) == 53
    assert sum(1 for r in rows if r[3].startswith("mistake")) == 6
    assert sum(1 for r in rows if r[3].startswith("new")) == 10
    assert sum(1 for r in rows if r[3].startswith("anchor")) == 12
    assert all(r[3].startswith(("old", "new", "game", "mistake", "anchor")) for r in rows)


def test_a_night_without_an_anchor_or_a_history_still_gives_a_file(tmp_path):
    buffer = replay.Buffer(tmp_path / "buffer")
    buffer.add("selfplay", "2026-10-23", lines(4, "new"))
    counts = replay.assemble(tmp_path / "t.txt", buffer, "2026-10-23", anchor=None)
    assert counts == {"night": 4, "history": 0, "mistakes": 0}
