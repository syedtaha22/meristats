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
import io
import stat
import tarfile
import zipfile

import pytest

from meristats import external
from meristats.external import Asset, ExternalError


def local_retrieve(source):
    def retrieve(url, destination):
        destination.write_bytes(source.read_bytes())

    return retrieve


def asset_for(path, member="x"):
    data = path.read_bytes()
    return Asset(
        "https://example.org/" + path.name, len(data), hashlib.sha256(data).hexdigest(), member
    )


def test_download_checks_size_and_hash_and_caches(tmp_path):
    source = tmp_path / "f.bin"
    source.write_bytes(b"content")
    asset = asset_for(source)
    calls = []

    def retrieve(url, destination):
        calls.append(url)
        destination.write_bytes(source.read_bytes())

    target = tmp_path / "cache" / "f.bin"
    assert external.download(asset, target, retrieve) == target
    assert external.download(asset, target, retrieve) == target
    assert len(calls) == 1


def test_a_download_with_the_wrong_hash_is_removed_and_refused(tmp_path):
    source = tmp_path / "f.bin"
    source.write_bytes(b"content")
    asset = asset_for(source)
    bad = Asset(asset.url, asset.size, "0" * 64, "x")
    target = tmp_path / "cache" / "f.bin"

    with pytest.raises(ExternalError, match="does not match"):
        external.download(bad, target, local_retrieve(source))

    assert not target.exists()


def test_a_download_with_the_wrong_size_is_refused(tmp_path):
    source = tmp_path / "f.bin"
    source.write_bytes(b"content")
    asset = asset_for(source)
    wrong = Asset(asset.url, asset.size + 1, asset.sha256, "x")

    with pytest.raises(ExternalError):
        external.download(wrong, tmp_path / "f2.bin", local_retrieve(source))


def test_a_cached_file_with_the_wrong_hash_is_downloaded_again(tmp_path):
    source = tmp_path / "f.bin"
    source.write_bytes(b"content")
    asset = asset_for(source)
    target = tmp_path / "cache" / "f.bin"
    target.parent.mkdir()
    target.write_bytes(b"conteXt")

    external.download(asset, target, local_retrieve(source))
    assert target.read_bytes() == b"content"


def make_tar(path, member, data):
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo(member)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))


def test_fetch_stockfish_extracts_an_executable(tmp_path):
    archive = tmp_path / "sf.tar.gz"
    make_tar(archive, "stockfish/sf-fake", b"#!/bin/sh\necho hi\n")
    asset = asset_for(archive, "stockfish/sf-fake")
    path = external.fetch_stockfish(
        tmp_path / "cache", "x86_64", {"x86_64": asset}, local_retrieve(archive)
    )

    assert path.name == "sf-fake" and path.read_bytes() == b"#!/bin/sh\necho hi\n"
    assert path.stat().st_mode & stat.S_IXUSR
    assert (
        external.fetch_stockfish(
            tmp_path / "cache", "x86_64", {"x86_64": asset}, local_retrieve(archive)
        )
        == path
    )


def test_fetch_stockfish_refuses_a_machine_without_a_build(tmp_path):
    with pytest.raises(ExternalError, match="no Stockfish"):
        external.fetch_stockfish(tmp_path, "mips", {})


def test_the_recorded_assets_have_hashes_and_sizes_for_three_machines():
    assert set(external.STOCKFISH) == {"x86_64", "aarch64", "riscv64"}
    assert external.STOCKFISH["x86_64"].sha256 == (
        "9defc0d4e55d49c65a6d042f3e571a39fcea499ade6dbe741b53b8c65e03611f"
    )
    assert external.STOCKFISH["x86_64"].size == 81388977
    assert (
        external.BOOK.sha256 == "7e1e9dd118b4bb97d8a8b5b8a790c86e21f8509d59a27d2883767d94477be02e"
    )
    assert all(len(a.sha256) == 64 for a in external.STOCKFISH.values())


def test_fetch_book_unzips_the_pgn(tmp_path):
    archive = tmp_path / "book.zip"

    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("8moves_v3.pgn", '[Event "?"]\n\n1. e4 *\n')

    asset = asset_for(archive, "8moves_v3.pgn")
    path = external.fetch_book(tmp_path / "cache", asset, local_retrieve(archive))

    assert path.name == "8moves_v3.pgn" and path.read_text().startswith("[Event")


def test_register_external_records_the_program_by_hash(store, fake_engine):
    program, _ = fake_engine()
    engine_id, version = external.register_external(store, "fake", program)
    row = store.db.execute("SELECT * FROM engines WHERE engine_id = ?", (engine_id,)).fetchone()

    assert version == "Fake 1.0" and row["kind"] == "external" and row["commit_sha"] is None
    assert row["binary_sha256"] == hashlib.sha256(program.read_bytes()).hexdigest()
    assert external.register_external(store, "fake", program)[0] == engine_id
