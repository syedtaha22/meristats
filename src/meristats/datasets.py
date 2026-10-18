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
import json
import os
import random
import shutil
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import chess
import pyarrow.parquet as pq

from meristats.store import Store

PUZZLE_URL = "https://huggingface.co/datasets/Lichess/chess-puzzles/resolve/main/data/"
PUZZLE_API = "https://huggingface.co/api/datasets/Lichess/chess-puzzles/tree/main/data"
PUZZLE_FILES = [f"train-0000{i}-of-00003.parquet" for i in range(3)]
BAND = 200
SAMPLE_NAME = "lichess-puzzles-sample"


class DatasetError(Exception):
    """
    A dataset file is missing, does not match its hash, or is malformed.
    """


@dataclass(frozen=True)
class Puzzle:
    id: str
    rating: int
    mate: int
    weight: float
    themes: tuple[str, ...]
    fen: str
    moves: tuple[str, ...]

    @property
    def solver_moves(self) -> int:
        """
        The number of moves of the solver: the moves at even places of the solution.
        """
        return (len(self.moves) + 1) // 2


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


def hashes_on_server() -> dict[str, str]:
    """
    The SHA-256 hash that the server gives for each file of the puzzle database.
    """
    with urllib.request.urlopen(PUZZLE_API) as response:
        entries = json.load(response)

    return {entry["path"].split("/")[-1]: entry["lfs"]["oid"] for entry in entries}


def fetch_puzzle_files(
    directory: Path, hashes: dict[str, str] | None = None, reuse_from: tuple[Path, ...] = ()
) -> list[Path]:
    """
    Returns the files of the puzzle database in `directory`, each with the SHA-256 hash that the
    server gives. A file that is missing or has another hash is taken from a folder of
    `reuse_from` when one holds a file with the right hash (a hard link, so that nothing is
    copied), and is downloaded otherwise.
    """
    directory.mkdir(parents=True, exist_ok=True)
    hashes = hashes or hashes_on_server()
    paths = []

    for name in PUZZLE_FILES:
        path = directory / name
        wanted = hashes[name]

        if path.exists() and file_sha256(path) == wanted:
            paths.append(path)
            continue

        if path.exists():
            path.unlink()

        for folder in reuse_from:
            candidate = Path(folder) / name

            if candidate.exists() and file_sha256(candidate) == wanted:
                try:
                    os.link(candidate, path)
                except OSError:
                    shutil.copyfile(candidate, path)

                break
        else:
            urllib.request.urlretrieve(PUZZLE_URL + name, path)

        if file_sha256(path) != wanted:
            path.unlink()
            raise DatasetError(f"{name} does not have the SHA-256 hash {wanted}")

        paths.append(path)

    return paths


def mate_length(themes: list[str]) -> int:
    """
    The number N of the theme `mateInN`, or 0 for a puzzle without one.
    """
    for theme in themes:
        if theme.startswith("mateIn"):
            return int(theme[6:])

    return 0


def group_puzzles(paths: list[Path]) -> dict[tuple[int, int], list[str]]:
    """
    The ids of the database, grouped by kind of puzzle (the length of the mate, or 0) and band of
    rating.
    """
    groups = defaultdict(list)

    for path in paths:
        table = pq.read_table(path, columns=["PuzzleId", "Rating", "Themes"])

        for puzzle_id, rating, themes in zip(
            table.column("PuzzleId").to_pylist(),
            table.column("Rating").to_pylist(),
            table.column("Themes").to_pylist(),
        ):
            groups[(mate_length(themes), rating // BAND)].append(puzzle_id)

    return groups


def allocate(groups: dict, total: int, take_all: set[int]) -> dict:
    """
    The number of puzzles to draw from each group: all of the groups of mates in `take_all`, and
    the rest of `total` in proportion to the size of the other groups, by the largest remainder.
    """
    full = {key for key in groups if key[0] in take_all}
    rest = sum(len(ids) for key, ids in groups.items() if key not in full)
    budget = total - sum(len(groups[key]) for key in full)

    if budget < 0 or rest == 0:
        raise DatasetError("total is smaller than the puzzles of take_all")

    counts = {key: len(groups[key]) for key in full}
    shares = {}

    for key, ids in groups.items():
        if key not in full:
            shares[key] = budget * len(ids) / rest
            counts[key] = min(len(ids), int(shares[key]))

    missing = total - sum(counts.values())

    for key in sorted(shares, key=lambda k: (shares[k] - int(shares[k]), k), reverse=True):
        if missing <= 0:
            break

        if counts[key] < len(groups[key]):
            counts[key] += 1
            missing -= 1

    return counts


def draw(groups: dict, counts: dict, seed: int) -> dict[str, float]:
    """
    The sample, as the weight of each chosen id. The weight is the size of the group over the
    number drawn from it.
    """
    weights = {}

    for key in sorted(groups):
        ids = sorted(groups[key])
        count = counts[key]

        if count == 0:
            continue

        chosen = ids if count >= len(ids) else random.Random(f"{seed} {key}").sample(ids, count)

        for puzzle_id in chosen:
            weights[puzzle_id] = len(ids) / count

    return weights


def write_sample(paths: list[Path], weights: dict[str, float], out: Path) -> int:
    """
    Writes the sample to `out`, sorted by id, with one puzzle for each line and the tab separated
    fields id, rating, mate, weight, themes, FEN and moves. The FEN is the position after the
    first move of the puzzle, which is a move of the opponent. Returns the number of puzzles.
    """
    rows = []
    names = ("PuzzleId", "FEN", "Moves", "Rating", "Themes")

    for path in paths:
        table = pq.read_table(path, columns=list(names))
        columns = [table.column(name).to_pylist() for name in names]

        for puzzle_id, fen, moves, rating, themes in zip(*columns):
            if puzzle_id not in weights:
                continue

            moves = moves.split()
            board = chess.Board(fen)
            board.push_uci(moves[0])
            rows.append(
                (
                    puzzle_id,
                    rating,
                    mate_length(themes),
                    weights[puzzle_id],
                    ",".join(themes),
                    board.fen(),
                    " ".join(moves[1:]),
                )
            )

    rows.sort()
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as file:
        for puzzle_id, rating, mate, weight, themes, fen, moves in rows:
            file.write(f"{puzzle_id}\t{rating}\t{mate}\t{weight:.6f}\t{themes}\t{fen}\t{moves}\n")

    return len(rows)


def build_puzzle_sample(
    paths: list[Path],
    out: Path,
    total: int = 1_000_000,
    take_all: tuple[int, ...] = (3, 4, 5),
    seed: int = 1,
) -> dict:
    """
    Builds the sample file from the parquet files, and returns its description.
    """
    groups = group_puzzles(paths)
    weights = draw(groups, allocate(groups, total, set(take_all)), seed)
    rows = write_sample(paths, weights, out)
    return {
        "total": total,
        "take_all": list(take_all),
        "seed": seed,
        "band": BAND,
        "rows": rows,
        "database_puzzles": sum(len(ids) for ids in groups.values()),
        "sources": {path.name: file_sha256(path) for path in paths},
    }


def load_puzzles(path: Path, limit: int = 0) -> list[Puzzle]:
    """
    Reads the first `limit` puzzles of a sample file, or all of them when `limit` is 0. The
    position of a puzzle in the list is its item number.
    """
    puzzles = []

    with open(path, encoding="utf-8") as file:
        for number, line in enumerate(file, 1):
            if limit and len(puzzles) >= limit:
                break

            fields = line.rstrip("\n").split("\t")

            if len(fields) != 7:
                raise DatasetError(f"{path.name} line {number} has {len(fields)} fields, not 7")

            puzzles.append(
                Puzzle(
                    fields[0],
                    int(fields[1]),
                    int(fields[2]),
                    float(fields[3]),
                    tuple(fields[4].split(",")),
                    fields[5],
                    tuple(fields[6].split()),
                )
            )

    return puzzles


def register_file(
    store: Store, name: str, version: str, path: Path, description: dict, keep_copy: bool = True
) -> tuple[int, str]:
    """
    Records a dataset file by name and hash, and stores a compressed copy of it as a blob, so that
    the data of a run can be restored from the data folder alone. Returns the dataset id and the
    hash.
    """
    data = Path(path).read_bytes()

    if keep_copy:
        blob = store.put_blob(data, "dataset", Path(path).name)
        digest = blob.sha256
    else:
        digest = hashlib.sha256(data).hexdigest()

    return store.add_dataset(name, version, digest, description), digest


@dataclass(frozen=True)
class SuitePosition:
    region: str
    name: str
    fen: str


@dataclass(frozen=True)
class PerftPosition:
    name: str
    fen: str
    counts: tuple[int, ...]


def read_suite(path: Path) -> list[SuitePosition]:
    """
    Reads a suite file: one position for each line, with the tab separated fields region, name
    and FEN. Empty lines and lines that start with `#` are skipped.
    """
    positions = []

    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line or line.startswith("#"):
            continue

        fields = line.split("\t")

        if len(fields) != 3:
            raise DatasetError(f"{Path(path).name} line {number} has {len(fields)} fields, not 3")

        positions.append(SuitePosition(*fields))

    return positions


def read_perft(path: Path) -> list[PerftPosition]:
    """
    Reads a perft file: one position for each line, with the tab separated fields name, FEN and
    the node counts for depth 1, 2, 3 and so on, separated by commas.
    """
    positions = []

    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line or line.startswith("#"):
            continue

        fields = line.split("\t")

        if len(fields) != 3:
            raise DatasetError(f"{Path(path).name} line {number} has {len(fields)} fields, not 3")

        try:
            counts = tuple(int(n) for n in fields[2].split(","))
        except ValueError as error:
            raise DatasetError(
                f"{Path(path).name} line {number} has a count that is not a whole number"
            ) from error

        positions.append(PerftPosition(fields[0], fields[1], counts))

    return positions
