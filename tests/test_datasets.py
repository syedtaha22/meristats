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

import hashlib
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from meristats import datasets
from meristats.datasets import DatasetError, Puzzle

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def write_parquet(path, rows):
    """
    rows: (id, fen, moves, rating, themes)
    """
    table = pa.table(
        {
            "PuzzleId": [r[0] for r in rows],
            "FEN": [r[1] for r in rows],
            "Moves": [r[2] for r in rows],
            "Rating": [r[3] for r in rows],
            "Themes": [r[4] for r in rows],
        }
    )
    pq.write_table(table, path)


def make_rows(count, prefix="p", mate=None, rating=1500):
    themes = [f"mateIn{mate}", "short"] if mate else ["advantage"]
    return [(f"{prefix}{i:04d}", START, "e2e4 e7e5 g1f3", rating + i, themes) for i in range(count)]


def test_mate_length():
    assert datasets.mate_length(["short", "mateIn3", "endgame"]) == 3
    assert datasets.mate_length(["advantage"]) == 0
    assert datasets.mate_length([]) == 0


def test_groups_by_mate_length_and_band(tmp_path):
    path = tmp_path / "a.parquet"
    write_parquet(path, make_rows(5, "a", rating=1500) + make_rows(3, "b", mate=2, rating=1000))
    groups = datasets.group_puzzles([path])

    assert sorted(groups) == [(0, 7), (2, 5)]
    assert len(groups[(0, 7)]) == 5 and len(groups[(2, 5)]) == 3


def test_allocate_takes_all_of_the_listed_mates_and_the_rest_in_proportion():
    groups = {
        (3, 5): list("abcdef"),
        (0, 7): [str(i) for i in range(60)],
        (0, 8): [str(i) for i in range(40)],
    }
    counts = datasets.allocate(groups, 56, {3})

    assert counts[(3, 5)] == 6
    assert counts[(0, 7)] + counts[(0, 8)] == 50
    assert counts[(0, 7)] == 30 and counts[(0, 8)] == 20
    assert sum(counts.values()) == 56


def test_allocate_by_largest_remainder_sums_to_the_total():
    groups = {(0, k): [f"{k}-{i}" for i in range(size)] for k, size in enumerate((7, 11, 13, 19))}
    counts = datasets.allocate(groups, 20, set())

    assert sum(counts.values()) == 20
    assert all(counts[key] <= len(groups[key]) for key in groups)


def test_allocate_refuses_a_total_below_the_mates():
    groups = {(3, 5): list("abcdef"), (0, 7): ["x"]}

    with pytest.raises(DatasetError):
        datasets.allocate(groups, 5, {3})

    with pytest.raises(DatasetError):
        datasets.allocate({(3, 5): ["a"]}, 5, {3})


def test_draw_is_deterministic_and_weights_cover_the_groups():
    groups = {(0, 7): [f"x{i:03d}" for i in range(100)], (0, 8): [f"y{i:03d}" for i in range(40)]}
    counts = {(0, 7): 10, (0, 8): 40}
    first = datasets.draw(groups, counts, seed=1)
    second = datasets.draw(groups, counts, seed=1)
    other = datasets.draw(groups, counts, seed=2)

    assert first == second and first != other
    assert len(first) == 50
    assert sum(first.values()) == pytest.approx(140)
    assert all(first[key] == 10.0 for key in first if key.startswith("x"))
    assert all(first[key] == 1.0 for key in first if key.startswith("y"))


def test_draw_does_not_depend_on_the_order_of_the_ids():
    ids = [f"x{i:03d}" for i in range(50)]
    a = datasets.draw({(0, 1): list(ids)}, {(0, 1): 7}, seed=3)
    b = datasets.draw({(0, 1): list(reversed(ids))}, {(0, 1): 7}, seed=3)

    assert a == b


def test_write_sample_applies_the_first_move_and_formats_the_line(tmp_path):
    path = tmp_path / "a.parquet"
    write_parquet(path, make_rows(3, "a", mate=2))
    out = tmp_path / "s.tsv"
    rows = datasets.write_sample([path], {"a0000": 2.5, "a0002": 1.0}, out)
    lines = out.read_text().splitlines()

    assert rows == 2 and [line.split("\t")[0] for line in lines] == ["a0000", "a0002"]
    fields = lines[0].split("\t")
    assert fields[1] == "1500" and fields[2] == "2" and fields[3] == "2.500000"
    assert fields[4] == "mateIn2,short"
    assert fields[5] == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    assert fields[6] == "e7e5 g1f3"


def test_the_whole_sample_pipeline_and_the_loader(tmp_path):
    path = tmp_path / "a.parquet"
    write_parquet(path, make_rows(20, "n") + make_rows(4, "m", mate=3, rating=2100))
    out = tmp_path / "s.tsv"
    description = datasets.build_puzzle_sample([path], out, total=14, take_all=(3,), seed=5)

    assert description["rows"] == 14 and description["database_puzzles"] == 24
    assert description["sources"] == {"a.parquet": datasets.file_sha256(path)}

    puzzles = datasets.load_puzzles(out)
    assert len(puzzles) == 14
    assert sum(1 for p in puzzles if p.mate == 3) == 4
    assert sum(p.weight for p in puzzles) == pytest.approx(24)
    assert [p.id for p in puzzles] == sorted(p.id for p in puzzles)
    assert len(datasets.load_puzzles(out, limit=5)) == 5

    again = tmp_path / "again.tsv"
    datasets.build_puzzle_sample([path], again, total=14, take_all=(3,), seed=5)
    assert again.read_bytes() == out.read_bytes()


def test_puzzle_solver_moves():
    puzzle = Puzzle("x", 1500, 0, 1.0, ("a",), START, ("e7e5", "g1f3", "b8c6", "f1c4", "g8f6"))
    assert puzzle.solver_moves == 3
    assert Puzzle("y", 1, 0, 1.0, (), START, ("e7e5",)).solver_moves == 1
    assert Puzzle("y", 1, 0, 1.0, (), START, ("e7e5", "a")).solver_moves == 1


def test_load_puzzles_rejects_a_malformed_line(tmp_path):
    bad = tmp_path / "bad.tsv"
    bad.write_text("id\t1500\tnot enough\n")

    with pytest.raises(DatasetError, match="line 1"):
        datasets.load_puzzles(bad)


def test_fetch_reuses_files_by_hard_link_and_checks_the_hash(tmp_path):
    source = tmp_path / "old"
    source.mkdir()
    hashes = {}

    for name in datasets.PUZZLE_FILES:
        (source / name).write_bytes(name.encode() * 10)
        hashes[name] = hashlib.sha256(name.encode() * 10).hexdigest()

    target = tmp_path / "cache"
    paths = datasets.fetch_puzzle_files(target, hashes, reuse_from=(source,))

    assert [p.name for p in paths] == datasets.PUZZLE_FILES
    assert os.stat(paths[0]).st_ino == os.stat(source / paths[0].name).st_ino

    again = datasets.fetch_puzzle_files(target, hashes, reuse_from=())
    assert again == paths


def test_fetch_refuses_a_file_with_the_wrong_hash(tmp_path):
    source = tmp_path / "old"
    source.mkdir()

    for name in datasets.PUZZLE_FILES:
        (source / name).write_bytes(b"wrong")

    hashes = {name: hashlib.sha256(b"right").hexdigest() for name in datasets.PUZZLE_FILES}
    monkey_urls = datasets.urllib.request

    def fail(url, path):
        raise OSError("no network in tests")

    original = monkey_urls.urlretrieve
    monkey_urls.urlretrieve = fail

    try:
        with pytest.raises(OSError, match="no network"):
            datasets.fetch_puzzle_files(tmp_path / "cache", hashes, reuse_from=(source,))
    finally:
        monkey_urls.urlretrieve = original


def test_register_file_stores_a_restorable_copy(store, tmp_path):
    path = tmp_path / "positions.txt"
    path.write_text("a position\n" * 50)
    dataset_id, digest = datasets.register_file(store, "positions", "1", path, {"rows": 50})

    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert store.get_blob(digest) == path.read_bytes()
    assert datasets.register_file(store, "positions", "1", path, {"rows": 50})[0] == dataset_id
    assert datasets.register_file(store, "p2", "1", path, {}, keep_copy=False)[1] == digest


def test_read_suite_skips_comments_and_blank_lines(tmp_path):
    path = tmp_path / "suite.txt"
    path.write_text("# header\n\nopening\tope-1\t8/8/8/8/8/8/8/K6k b\nendgame\tend-1\tfen2\n")
    positions = datasets.read_suite(path)

    assert [(p.region, p.name, p.fen) for p in positions] == [
        ("opening", "ope-1", "8/8/8/8/8/8/8/K6k b"),
        ("endgame", "end-1", "fen2"),
    ]


def test_read_perft_parses_the_counts(tmp_path):
    path = tmp_path / "perft.txt"
    path.write_text("# header\ninitial\tfen\t20,400,8902\n")

    assert datasets.read_perft(path) == [datasets.PerftPosition("initial", "fen", (20, 400, 8902))]


@pytest.mark.parametrize(
    "reader,first",
    [(datasets.read_suite, "fine\tfine\tfine"), (datasets.read_perft, "fine\tfen\t1,2,3")],
)
def test_a_malformed_line_names_its_number(tmp_path, reader, first):
    path = tmp_path / "bad.txt"
    path.write_text(first + "\nonly\ttwo\n")

    with pytest.raises(DatasetError, match="line 2"):
        reader(path)


def test_a_perft_count_that_is_not_a_number_is_refused(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("name\tfen\t20,x,8902\n")

    with pytest.raises(DatasetError, match="whole number"):
        datasets.read_perft(path)


def test_the_real_suite_and_perft_files_parse():
    from pathlib import Path

    here = Path(__file__).resolve().parents[1] / "datasets"
    suite = datasets.read_suite(here / "suite.txt")
    perft = datasets.read_perft(here / "perft.txt")

    assert len(suite) == 15 and {p.region for p in suite} == {"opening", "middlegame", "endgame"}
    assert len(perft) == 6 and perft[0].counts[:3] == (20, 400, 8902)


def test_hashes_on_server_reads_the_lfs_oids(monkeypatch):
    import io
    import json

    listing = [
        {"path": f"data/{name}", "lfs": {"oid": "ab" * 32}} for name in datasets.PUZZLE_FILES
    ]
    monkeypatch.setattr(
        datasets.urllib.request, "urlopen", lambda url: io.BytesIO(json.dumps(listing).encode())
    )

    assert datasets.hashes_on_server() == {name: "ab" * 32 for name in datasets.PUZZLE_FILES}


def test_a_file_on_another_device_is_copied_when_a_link_fails(tmp_path, monkeypatch):
    source = tmp_path / "old"
    source.mkdir()
    hashes = {}

    for name in datasets.PUZZLE_FILES:
        (source / name).write_bytes(name.encode())
        hashes[name] = hashlib.sha256(name.encode()).hexdigest()

    def refuse(src, dst):
        raise OSError("cross-device link")

    monkeypatch.setattr(datasets.os, "link", refuse)
    paths = datasets.fetch_puzzle_files(tmp_path / "cache", hashes, reuse_from=(source,))

    assert paths[0].read_bytes() == datasets.PUZZLE_FILES[0].encode()
    assert os.stat(paths[0]).st_ino != os.stat(source / paths[0].name).st_ino


def test_a_cached_file_with_the_wrong_hash_is_replaced_from_the_reuse_folder(tmp_path):
    source = tmp_path / "old"
    source.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    hashes = {}

    for name in datasets.PUZZLE_FILES:
        (source / name).write_bytes(name.encode())
        (cache / name).write_bytes(b"damaged")
        hashes[name] = hashlib.sha256(name.encode()).hexdigest()

    paths = datasets.fetch_puzzle_files(cache, hashes, reuse_from=(source,))

    assert all(p.read_bytes() == p.name.encode() for p in paths)
