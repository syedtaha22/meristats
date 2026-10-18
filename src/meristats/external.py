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
import platform
import tarfile
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from meristats.store import Store
from meristats.uci import UciEngine


class ExternalError(Exception):
    """
    A download does not match its recorded size or hash, or no build exists for this machine.
    """


@dataclass(frozen=True)
class Asset:
    url: str
    size: int
    sha256: str
    member: str


STOCKFISH_RELEASE = "sf_19"
STOCKFISH_URL = "https://github.com/official-stockfish/Stockfish/releases/download/sf_19/"

# The size and the SHA-256 hash are the values that GitHub lists for the assets of the release.
STOCKFISH = {
    "x86_64": Asset(
        STOCKFISH_URL + "stockfish-linux-x86-64-universal.tar.gz",
        81388977,
        "9defc0d4e55d49c65a6d042f3e571a39fcea499ade6dbe741b53b8c65e03611f",
        "stockfish/stockfish-linux-x86-64-universal",
    ),
    "aarch64": Asset(
        STOCKFISH_URL + "stockfish-linux-arm64-universal.tar.gz",
        80181250,
        "fe26cfd1d9db4c8af3d21e24d9ff34cacb31c1f940085a7583da11796f2bac01",
        "stockfish/stockfish-linux-arm64-universal",
    ),
    "riscv64": Asset(
        STOCKFISH_URL + "stockfish-linux-riscv64-universal.tar.gz",
        80241608,
        "d4b4d118c003bc669a939d9fb208e3ef6b367730fd79a70dbc278b18475df3d1",
        "stockfish/stockfish-linux-riscv64-universal",
    ),
}

BOOK = Asset(
    "https://github.com/official-stockfish/books/raw/master/8moves_v3.pgn.zip",
    807558,
    "7e1e9dd118b4bb97d8a8b5b8a790c86e21f8509d59a27d2883767d94477be02e",
    "8moves_v3.pgn",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


def download(
    asset: Asset,
    destination: Path,
    retrieve: Callable[[str, Path], object] = urllib.request.urlretrieve,
) -> Path:
    """
    Downloads an asset unless a file with its size and hash is already there, and checks the
    result. A file that does not match is removed.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    if (
        destination.exists()
        and destination.stat().st_size == asset.size
        and _sha256(destination) == asset.sha256
    ):
        return destination

    retrieve(asset.url, destination)

    if destination.stat().st_size != asset.size or _sha256(destination) != asset.sha256:
        found = f"{destination.stat().st_size} bytes, SHA-256 {_sha256(destination)}"
        destination.unlink()
        raise ExternalError(
            f"{asset.url} does not match: {found}, expected {asset.size} bytes, "
            f"SHA-256 {asset.sha256}"
        )

    return destination


def fetch_stockfish(
    cache_dir: Path,
    machine: str | None = None,
    table: dict[str, Asset] | None = None,
    retrieve: Callable[[str, Path], object] = urllib.request.urlretrieve,
) -> Path:
    """
    Downloads the Stockfish release for this machine, checks it against the recorded hash, and
    returns the path of the program. There is no build for a machine that is not in the table.
    """
    machine = machine or platform.machine()
    assets = table if table is not None else STOCKFISH

    if machine not in assets:
        raise ExternalError(f"no Stockfish {STOCKFISH_RELEASE} build for {machine}")

    asset = assets[machine]
    root = cache_dir / "downloads"
    archive = download(asset, root / Path(asset.url).name, retrieve)
    target = root / "stockfish" / Path(asset.member).name

    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)

        with tarfile.open(archive) as tar:
            member = tar.getmember(asset.member)
            extracted = tar.extractfile(member)
            target.write_bytes(extracted.read())

        target.chmod(0o755)

    return target


def fetch_book(
    cache_dir: Path,
    asset: Asset = BOOK,
    retrieve: Callable[[str, Path], object] = urllib.request.urlretrieve,
) -> Path:
    """
    Downloads the opening book (a zip), checks it against the recorded hash, unzips it, and
    returns the path of the PGN file.
    """
    root = cache_dir / "downloads"
    archive = download(asset, root / Path(asset.url).name, retrieve)
    target = root / asset.member

    if not target.exists():
        with zipfile.ZipFile(archive) as zipped:
            target.write_bytes(zipped.read(asset.member))

    return target


def register_external(store: Store, name: str, binary: Path) -> tuple[int, str]:
    """
    Records an engine program that is not built from this project, by the hash of its file, and
    returns the engine id and the name that the program gives in `uci`.
    """
    with UciEngine(binary) as engine:
        version = engine.name

    digest = _sha256(binary)
    return store.add_engine(
        "external", name, None, digest, {"path": str(binary)}, None, version
    ), version
