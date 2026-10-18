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
import re
import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path

import zstandard

SCHEMA_VERSION = 2
LOCK_TIMEOUT = 60.0
ZSTD_LEVEL = 10


def utc_now() -> str:
    """
    The current time in UTC, as ISO 8601 text with a trailing Z.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_hex(data: bytes) -> str:
    """
    The SHA-256 hash of `data`, as hexadecimal text.
    """
    return hashlib.sha256(data).hexdigest()


def canonical_json(value) -> str:
    """
    JSON with sorted keys and no spaces, so that equal values give equal text.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class StoreError(Exception):
    """
    The data folder is damaged or a stored file does not match its hash.
    """


@dataclass(frozen=True)
class Blob:
    sha256: str
    size_bytes: int
    stored_bytes: int


class Store:
    """
    The data folder: one SQLite database, `meristats.db`, and `blobs/`, the files that the
    database names, compressed with zstd and stored under the SHA-256 hash of their bytes.

    Rows of the protected tables are never changed or removed, which triggers of the database
    enforce. A blob is written once.
    """

    def __init__(self, data_dir: Path):
        self.root = Path(data_dir)
        self.blob_dir = self.root / "blobs"
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        # A writer waits for the lock for up to a minute, so that a run is not lost at its end
        # because another process is writing at that moment.
        self.db = sqlite3.connect(
            self.root / "meristats.db", isolation_level=None, timeout=LOCK_TIMEOUT
        )
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("PRAGMA journal_mode = WAL")
        self._migrate()

    def close(self) -> None:
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ schema

    def _migrations(self) -> list[tuple[int, str]]:
        found = []

        for entry in files("meristats").joinpath("migrations").iterdir():
            match = re.fullmatch(r"(\d+)_.*\.sql", entry.name)

            if match:
                found.append((int(match.group(1)), entry.read_text(encoding="utf-8")))

        return sorted(found)

    def schema_version(self) -> int:
        """
        The version of the schema of the database, 0 for a new database.
        """
        has_meta = self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta'"
        ).fetchone()

        if not has_meta:
            return 0

        row = self.db.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        return int(row["value"]) if row else 0

    def _migrate(self) -> None:
        current = self.schema_version()
        newest = max((number for number, _ in self._migrations()), default=0)

        if current > newest:
            raise StoreError(
                f"the database has schema {current}, newer than this program ({newest})"
            )

        for number, script in self._migrations():
            if number <= current:
                continue

            self.db.executescript("BEGIN;\n" + script + "\nCOMMIT;")
            self.db.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(number),),
            )

    # ------------------------------------------------------------------- blobs

    def _blob_path(self, sha: str) -> Path:
        return self.blob_dir / sha[:2] / f"{sha}.zst"

    def put_blob(self, data: bytes, kind: str, original_name: str | None = None) -> Blob:
        """
        Stores `data` under the hash of its bytes and returns its record. Storing the same bytes
        again changes nothing.
        """
        sha = sha256_hex(data)
        path = self._blob_path(sha)
        known = self.db.execute("SELECT * FROM blobs WHERE sha256 = ?", (sha,)).fetchone()

        if known and path.exists():
            return Blob(sha, known["size_bytes"], known["stored_bytes"])

        packed = zstandard.ZstdCompressor(level=ZSTD_LEVEL).compress(data)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Each writer has its own temporary file, so that two processes storing the same bytes
        # at once do not rename each other's file.
        partial = path.parent / f".{sha}.{os.getpid()}.{uuid.uuid4().hex}.part"

        try:
            partial.write_bytes(packed)
            os.replace(partial, path)
        finally:
            partial.unlink(missing_ok=True)

        if not known:
            self.db.execute(
                "INSERT OR IGNORE INTO blobs (sha256, size_bytes, stored_bytes, kind, "
                "original_name) VALUES (?, ?, ?, ?, ?)",
                (sha, len(data), len(packed), kind, original_name),
            )

        return Blob(sha, len(data), len(packed))

    def get_blob(self, sha: str) -> bytes:
        """
        Returns the original bytes of a blob, after checking them against the hash.
        """
        path = self._blob_path(sha)

        if not path.exists():
            raise StoreError(f"blob {sha} is not stored")

        data = zstandard.ZstdDecompressor().decompress(path.read_bytes())

        if sha256_hex(data) != sha:
            raise StoreError(f"blob {sha} does not match its hash")

        return data

    def verify_blobs(self) -> list[str]:
        """
        Checks every blob of the database and returns the hashes that are missing or damaged.
        """
        bad = []

        for row in self.db.execute("SELECT sha256 FROM blobs"):
            try:
                self.get_blob(row["sha256"])
            except (StoreError, zstandard.ZstdError):
                bad.append(row["sha256"])

        return bad

    # -------------------------------------------------------------------- rows

    def _insert_unique(self, table: str, key: dict, values: dict, id_column: str) -> int:
        """
        Inserts a row unless one with the same `key` exists, and returns the id of the row.
        """
        where = " AND ".join(f"{name} IS ?" for name in key)
        row = self.db.execute(
            f"SELECT {id_column} FROM {table} WHERE {where}", tuple(key.values())
        ).fetchone()

        if row:
            return row[id_column]

        columns = {**key, **values}
        names = ", ".join(columns)
        marks = ", ".join("?" for _ in columns)

        try:
            cursor = self.db.execute(
                f"INSERT INTO {table} ({names}) VALUES ({marks})", tuple(columns.values())
            )
        except sqlite3.IntegrityError:
            # Another process stored the same row between the lookup and the insert.
            row = self.db.execute(
                f"SELECT {id_column} FROM {table} WHERE {where}", tuple(key.values())
            ).fetchone()

            if row is None:
                raise

            return row[id_column]

        return cursor.lastrowid

    def add_machine(self, info: dict) -> int:
        """
        Records a machine, by the fingerprint of its description, and returns its id.
        """
        fingerprint = sha256_hex(canonical_json(info).encode())
        return self._insert_unique(
            "machines",
            {"fingerprint": fingerprint},
            {
                "cpu_model": info.get("cpu_model", "unavailable"),
                "logical_cpus": info.get("logical_cpus", 0),
                "kernel": info.get("kernel", "unavailable"),
                "os": info.get("os", "unavailable"),
                "json": canonical_json(info),
            },
            "machine_id",
        )

    def add_engine(
        self,
        kind: str,
        name: str,
        commit_sha: str | None,
        binary_sha256: str,
        build: dict,
        bench_nodes: int | None,
        version_text: str,
    ) -> int:
        """
        Records an engine build, by commit and binary, and returns its id.
        """
        return self._insert_unique(
            "engines",
            {"kind": kind, "commit_sha": commit_sha, "binary_sha256": binary_sha256},
            {
                "name": name,
                "build_json": canonical_json(build),
                "bench_nodes": bench_nodes,
                "version_text": version_text,
            },
            "engine_id",
        )

    def find_engines(self, commit_sha: str) -> list[sqlite3.Row]:
        """
        The engine rows of a commit.
        """
        return self.db.execute(
            "SELECT * FROM engines WHERE commit_sha = ?", (commit_sha,)
        ).fetchall()

    def add_dataset(self, name: str, version: str, sha256: str, description: dict) -> int:
        """
        Records a dataset, by name and hash, and returns its id.
        """
        return self._insert_unique(
            "datasets",
            {"name": name, "sha256": sha256},
            {"version": version, "description_json": canonical_json(description)},
            "dataset_id",
        )

    def add_run(
        self,
        *,
        protocol: str,
        protocol_version: int,
        grade: str,
        status: str,
        started_utc: str,
        finished_utc: str,
        engine_id: int | None,
        config: str | None,
        limits: dict,
        dataset_id: int | None,
        seed: int | None,
        machine_id: int | None,
        conditions: dict,
        code_commit: str | None,
        code_dirty: bool | None,
        summary: dict,
        items_blob: str | None = None,
        supersedes: str | None = None,
        notes: str | None = None,
        blobs: tuple[tuple[str, str], ...] = (),
    ) -> str:
        """
        Writes one finished run, with the roles of its blobs, and returns its id. The id is the
        start time, the protocol and the first 8 characters of the hash of the provenance.
        """
        provenance = {
            "protocol": protocol,
            "protocol_version": protocol_version,
            "engine_id": engine_id,
            "config": config,
            "limits": limits,
            "dataset_id": dataset_id,
            "seed": seed,
            "machine_id": machine_id,
            "code_commit": code_commit,
            "started_utc": started_utc,
        }
        blobs = tuple(dict.fromkeys(blobs))
        stamp = started_utc.replace("-", "").replace(":", "")
        base = f"{stamp}-{protocol}-{sha256_hex(canonical_json(provenance).encode())[:8]}"

        # Two runs with the same provenance in the same second are both real runs, so the second
        # gets a numeric suffix. A run is written when it ends, and must not be lost to a clash.
        for attempt in range(1, 100):
            run_id = base if attempt == 1 else f"{base}-{attempt}"
            self.db.execute("BEGIN")

            try:
                self.db.execute(
                    "INSERT INTO runs (run_id, protocol, protocol_version, grade, status, "
                    "started_utc, finished_utc, engine_id, config, limits_json, dataset_id, seed, "
                    "machine_id, conditions_json, code_commit, code_dirty, schema_version, "
                    "supersedes, summary_json, items_blob, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, "
                    "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        protocol,
                        protocol_version,
                        grade,
                        status,
                        started_utc,
                        finished_utc,
                        engine_id,
                        config,
                        canonical_json(limits),
                        dataset_id,
                        seed,
                        machine_id,
                        canonical_json(conditions),
                        code_commit,
                        None if code_dirty is None else int(code_dirty),
                        SCHEMA_VERSION,
                        supersedes,
                        canonical_json(summary),
                        items_blob,
                        notes,
                    ),
                )

                for sha, role in blobs:
                    self.db.execute(
                        "INSERT INTO run_blobs (run_id, sha256, role) VALUES (?, ?, ?)",
                        (run_id, sha, role),
                    )

                self.db.execute("COMMIT")
                return run_id
            except sqlite3.IntegrityError as error:
                self.db.execute("ROLLBACK")

                if "runs.run_id" not in str(error):
                    raise
            except Exception:
                self.db.execute("ROLLBACK")
                raise

        raise StoreError(f"more than 99 runs with the id {base}")

    def get_run(self, run_id: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()

    def runs(self, protocol: str | None = None) -> list[sqlite3.Row]:
        """
        The runs, oldest first, of one protocol or of all.
        """
        if protocol:
            return self.db.execute(
                "SELECT * FROM runs WHERE protocol = ? ORDER BY started_utc", (protocol,)
            ).fetchall()

        return self.db.execute("SELECT * FROM runs ORDER BY started_utc").fetchall()

    # -------------------------------------------------------------------- pins

    def pin(self, page: str, label: str, run_id: str | None) -> None:
        """
        Puts a run on a page of the site under a label, or with `run_id` None takes the label off
        the page. A pin of the same page and label replaces the earlier one.
        """
        if run_id is not None and self.get_run(run_id) is None:
            raise StoreError(f"no run {run_id}")

        self.db.execute(
            "INSERT INTO pins (page, label, run_id, pinned_utc) VALUES (?, ?, ?, ?)",
            (page, label, run_id, utc_now()),
        )

    def pins(self) -> list[sqlite3.Row]:
        """
        The pins that count: the newest of each page and label, without the removed ones, in the
        order of the page and the label.
        """
        return self.db.execute(
            "SELECT p.page, p.label, p.run_id, p.pinned_utc FROM pins p WHERE p.run_id IS NOT NULL "
            "AND p.pin_id = (SELECT MAX(q.pin_id) FROM pins q WHERE q.page = p.page "
            "AND q.label = p.label) ORDER BY p.page, p.label"
        ).fetchall()

    # ------------------------------------------------------------------ backup

    def backup(self, destination: Path) -> None:
        """
        Copies the database, with SQLite's backup function, and the blobs to `destination`,
        which must not exist or must be empty.
        """
        destination = Path(destination)

        if destination.exists() and any(destination.iterdir()):
            raise StoreError(f"{destination} is not empty")

        destination.mkdir(parents=True, exist_ok=True)

        with closing(sqlite3.connect(destination / "meristats.db")) as target:
            self.db.backup(target)

        for source in self.blob_dir.rglob("*.zst"):
            target_path = destination / "blobs" / source.relative_to(self.blob_dir)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(source.read_bytes())
