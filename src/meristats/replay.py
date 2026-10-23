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
The replay buffer: the labelled positions of every night, kept so that each night's training can
see the days before it, and the assembly of a training file from the buffer and the anchor data.

The buffer is a folder of compressed files, `<kind>/<night>.txt.zst`, in the format of the
`positions` protocol. The kinds are `selfplay` (games of the engine against itself), `lichess`
(games of the account, analysed again) and `mistakes` (positions where the account gave away
material). Files are written once.
"""

import io
import random
from collections.abc import Iterable, Iterator
from pathlib import Path

import zstandard

KINDS = ("selfplay", "lichess", "mistakes")


class ReplayError(RuntimeError):
    pass


def read_lines(path: Path) -> Iterator[str]:
    """
    The lines of a data file, plain or compressed with zstd.
    """
    path = Path(path)

    if path.suffix == ".zst":
        with open(path, "rb") as handle:
            reader = zstandard.ZstdDecompressor().stream_reader(handle)

            for line in io.TextIOWrapper(reader, encoding="utf-8"):
                if line.strip():
                    yield line.rstrip("\n")

        return

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield line.rstrip("\n")


def sample(lines: Iterable[str], count: int, rng: random.Random) -> list[str]:
    """
    A uniform sample of up to `count` lines, in one pass (reservoir sampling).
    """
    kept: list[str] = []

    for index, line in enumerate(lines):
        if len(kept) < count:
            kept.append(line)
        else:
            slot = rng.randint(0, index)

            if slot < count:
                kept[slot] = line

    return kept


class Buffer:
    """
    The folder of the replay buffer.
    """

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def path(self, kind: str, night: str) -> Path:
        return self.directory / kind / f"{night}.txt.zst"

    def add(self, kind: str, night: str, text: str) -> Path:
        """
        Stores the lines of one night. A night that is in the buffer is left as it is, and
        refused, because a stored file never changes.
        """
        if kind not in KINDS:
            raise ReplayError(f"not a kind of the buffer: {kind}")

        path = self.path(kind, night)

        if path.exists():
            raise ReplayError(f"{path} is in the buffer already")

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".part")
        temporary.write_bytes(zstandard.ZstdCompressor(level=6).compress(text.encode("utf-8")))
        temporary.replace(path)
        return path

    def files(self, kinds: Iterable[str], *, before: str | None = None) -> list[Path]:
        """
        The files of some kinds, oldest first. With `before`, only the nights that sort earlier.
        """
        found = []

        for kind in kinds:
            for path in sorted((self.directory / kind).glob("*.txt.zst")):
                night = path.name.removesuffix(".txt.zst")

                if before is None or night < before:
                    found.append(path)

        return found


def assemble(
    out: Path,
    buffer: Buffer,
    night: str,
    *,
    anchor: Path | None,
    seed: int = 1,
    history_lines: int = 1_500_000,
    anchor_lines: int = 1_000_000,
    mistake_lines: int = 50_000,
    mistake_repeat: int = 3,
) -> dict:
    """
    Writes the training file of a night to `out`. It holds all of the night's positions, a sample
    of the nights before it, a sample of the anchor data, and the mistakes (of the night and of
    the nights before) each `mistake_repeat` times, which gives them more weight. The lines of a
    game keep one game number, so that the games that the trainer holds out are whole games.
    """
    rng = random.Random(seed)
    sources: list[tuple[str, list[str]]] = []
    tonight = [
        line
        for kind in ("selfplay", "lichess")
        if buffer.path(kind, night).exists()
        for line in read_lines(buffer.path(kind, night))
    ]
    sources.append(("night", tonight))
    past = buffer.files(("selfplay", "lichess"), before=night)
    sources.append(
        (
            "history",
            sample((line for path in past for line in read_lines(path)), history_lines, rng),
        )
    )
    past_mistakes = buffer.files(("mistakes",))
    mistakes = sample(
        (line for path in past_mistakes for line in read_lines(path)), mistake_lines, rng
    )
    sources.append(("mistakes", mistakes * mistake_repeat))

    if anchor is not None and Path(anchor).exists():
        sources.append(("anchor", sample(read_lines(Path(anchor)), anchor_lines, rng)))

    numbers: dict[tuple[str, str], int] = {}
    counts = {}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as handle:
        for name, lines in sources:
            written = 0

            for line in lines:
                fields = line.split("\t")

                if len(fields) != 4:
                    continue

                key = (name if name != "mistakes" else "night", fields[2])
                fields[2] = str(numbers.setdefault(key, len(numbers) + 1))
                handle.write("\t".join(fields) + "\n")
                written += 1

            counts[name] = written

    return counts
