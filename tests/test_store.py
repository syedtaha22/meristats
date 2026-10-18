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

import sqlite3

import pytest

from meristats.store import SCHEMA_VERSION, Store, StoreError, sha256_hex, utc_now

RUN = dict(
    protocol="puzzles",
    protocol_version=2,
    grade="verified",
    status="complete",
    started_utc="2026-10-18T05:00:00Z",
    finished_utc="2026-10-18T05:10:00Z",
    engine_id=None,
    config="network",
    limits={"nodes": 20000},
    dataset_id=None,
    seed=1,
    machine_id=None,
    conditions={"power_source": "ac"},
    code_commit="abc",
    code_dirty=False,
    summary={"solved": 0.93},
)


def test_a_new_database_has_the_current_schema(store):
    assert store.schema_version() == SCHEMA_VERSION


def test_reopening_does_not_migrate_again(tmp_path):
    with Store(tmp_path / "d") as first:
        first.put_blob(b"keep", "t")

    with Store(tmp_path / "d") as second:
        assert second.schema_version() == SCHEMA_VERSION
        assert second.db.execute("SELECT COUNT(*) FROM blobs").fetchone()[0] == 1


def test_a_newer_schema_is_refused(tmp_path):
    with Store(tmp_path / "d") as first:
        first.db.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")

    with pytest.raises(StoreError, match="newer"):
        Store(tmp_path / "d")


def test_blob_round_trip_and_hash(store):
    data = bytes(range(256)) * 100
    blob = store.put_blob(data, "items", "x.bin")

    assert blob.sha256 == sha256_hex(data)
    assert blob.size_bytes == len(data) and blob.stored_bytes < len(data)
    assert store.get_blob(blob.sha256) == data


def test_the_same_bytes_are_stored_once(store):
    first = store.put_blob(b"same", "a")
    second = store.put_blob(b"same", "b")

    assert first == second
    assert store.db.execute("SELECT COUNT(*) FROM blobs").fetchone()[0] == 1


def test_an_empty_blob_works(store):
    assert store.get_blob(store.put_blob(b"", "empty").sha256) == b""


def test_a_missing_blob_is_reported(store):
    with pytest.raises(StoreError, match="not stored"):
        store.get_blob("0" * 64)


def test_a_damaged_blob_is_detected(store):
    blob = store.put_blob(b"original", "t")
    path = store._blob_path(blob.sha256)
    path.write_bytes(path.read_bytes()[:-3] + b"xyz")

    assert store.verify_blobs() == [blob.sha256]

    other = store.put_blob(b"a different one", "t")
    store._blob_path(other.sha256).write_bytes(
        __import__("zstandard").ZstdCompressor().compress(b"not the original")
    )

    assert set(store.verify_blobs()) == {blob.sha256, other.sha256}


def test_a_blob_that_was_deleted_is_stored_again(store):
    blob = store.put_blob(b"again", "t")
    store._blob_path(blob.sha256).unlink()
    store.put_blob(b"again", "t")

    assert store.get_blob(blob.sha256) == b"again"
    assert store.verify_blobs() == []


@pytest.mark.parametrize(
    "table,column",
    [
        ("machines", "cpu_model"),
        ("engines", "name"),
        ("datasets", "version"),
        ("blobs", "kind"),
        ("runs", "notes"),
        ("run_blobs", "role"),
    ],
)
def test_rows_cannot_be_changed_or_removed(store, table, column):
    blob = store.put_blob(b"x", "t")
    machine = store.add_machine({"cpu_model": "cpu", "logical_cpus": 1, "kernel": "k", "os": "o"})
    engine = store.add_engine("meridian", "m", "a" * 40, "b" * 64, {}, 5, "v")
    dataset = store.add_dataset("d", "1", "c" * 64, {})
    store.add_run(
        **{**RUN, "engine_id": engine, "dataset_id": dataset, "machine_id": machine},
        items_blob=blob.sha256,
        blobs=((blob.sha256, "pgn"),),
    )
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        store.db.execute(f"UPDATE {table} SET {column} = 'changed'")

    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        store.db.execute(f"DELETE FROM {table}")


def test_machines_engines_and_datasets_are_deduplicated(store):
    info = {"cpu_model": "cpu", "logical_cpus": 4, "kernel": "k", "os": "o"}

    assert store.add_machine(info) == store.add_machine(dict(info))
    assert store.add_machine(info) != store.add_machine({**info, "logical_cpus": 8})
    assert store.add_engine("meridian", "m", "a" * 40, "b" * 64, {}, 1, "v") == store.add_engine(
        "meridian", "m", "a" * 40, "b" * 64, {}, 1, "v"
    )
    assert store.add_engine("meridian", "m", "a" * 40, "c" * 64, {}, 1, "v") != store.add_engine(
        "meridian", "m", "a" * 40, "b" * 64, {}, 1, "v"
    )
    assert store.add_dataset("d", "1", "c" * 64, {}) == store.add_dataset("d", "1", "c" * 64, {})


def test_an_external_engine_without_a_commit_is_deduplicated(store):
    first = store.add_engine("external", "stockfish", None, "d" * 64, {}, None, "SF 19")
    second = store.add_engine("external", "stockfish", None, "d" * 64, {}, None, "SF 19")

    assert first == second


def test_a_run_gets_an_id_from_its_start_time_and_protocol(store):
    run_id = store.add_run(**RUN)

    assert run_id.startswith("20261018T050000Z-puzzles-")
    assert len(run_id.rsplit("-", 1)[1]) == 8
    row = store.get_run(run_id)
    assert row["status"] == "complete" and row["grade"] == "verified"
    assert row["code_dirty"] == 0 and row["schema_version"] == SCHEMA_VERSION


def test_the_same_run_twice_gets_a_numeric_suffix_instead_of_failing(store):
    first = store.add_run(**RUN)
    second = store.add_run(**RUN)
    third = store.add_run(**RUN)

    assert second == first + "-2" and third == first + "-3"
    assert [r["run_id"] for r in store.runs("puzzles")] == [first, second, third]


def test_a_foreign_key_error_is_not_mistaken_for_a_clash(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.add_run(**RUN, blobs=(("f" * 64, "pgn"),))

    assert store.runs() == []


def test_a_run_with_a_missing_blob_is_rolled_back(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.add_run(**RUN, blobs=(("f" * 64, "pgn"),))

    assert store.runs() == []


def test_invalid_grade_and_status_are_refused(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.add_run(**{**RUN, "grade": "great"})

    with pytest.raises(sqlite3.IntegrityError):
        store.add_run(**{**RUN, "status": "maybe"})


def test_runs_are_listed_oldest_first_and_by_protocol(store):
    store.add_run(**{**RUN, "started_utc": "2026-10-18T06:00:00Z"})
    store.add_run(**{**RUN, "started_utc": "2026-10-18T05:00:00Z"})
    store.add_run(**{**RUN, "protocol": "rules"})

    puzzles = store.runs("puzzles")
    assert [row["started_utc"] for row in puzzles] == [
        "2026-10-18T05:00:00Z",
        "2026-10-18T06:00:00Z",
    ]
    assert len(store.runs()) == 3


def test_backup_copies_the_database_and_the_blobs(tmp_path, store):
    blob = store.put_blob(b"payload " * 100, "t")
    run = store.add_run(**RUN)
    store.backup(tmp_path / "copy")

    with Store(tmp_path / "copy") as copy:
        assert copy.get_run(run) is not None
        assert copy.get_blob(blob.sha256) == b"payload " * 100


def test_backup_into_a_folder_with_files_is_refused(tmp_path, store):
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "x").write_text("x")

    with pytest.raises(StoreError, match="not empty"):
        store.backup(tmp_path / "full")


def test_utc_now_is_iso_text_with_a_z():
    assert utc_now().endswith("Z") and len(utc_now()) == 20


def test_the_same_blob_listed_twice_with_one_role_is_stored_once(store):
    blob = store.put_blob(b"same", "t")
    run = store.add_run(
        **RUN, blobs=((blob.sha256, "pgn"), (blob.sha256, "pgn"), (blob.sha256, "items"))
    )

    assert sorted(
        r["role"] for r in store.db.execute("SELECT role FROM run_blobs WHERE run_id = ?", (run,))
    ) == ["items", "pgn"]


def test_two_processes_storing_the_same_blob_and_dataset_do_not_clash(tmp_path):
    import multiprocessing

    def work(queue):
        try:
            with Store(tmp_path / "shared") as shared:
                ids = [shared.add_dataset("d", "1", "c" * 64, {}) for _ in range(20)]
                blobs = [shared.put_blob(b"same bytes" * 100, "t").sha256 for _ in range(20)]
                queue.put(("ok", set(ids), set(blobs)))
        except Exception as error:
            queue.put(("error", repr(error), None))

    Store(tmp_path / "shared").close()
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    workers = [context.Process(target=work, args=(queue,)) for _ in range(4)]

    for process in workers:
        process.start()

    results = [queue.get(timeout=300) for _ in workers]

    for process in workers:
        process.join(timeout=60)

    assert [r[0] for r in results] == ["ok"] * 4, results
    assert len({frozenset(r[1]) for r in results}) == 1
    assert len({frozenset(r[2]) for r in results}) == 1


def test_the_insert_race_for_a_unique_row_returns_the_existing_id(store, monkeypatch):
    first = store.add_dataset("d", "1", "c" * 64, {})
    calls = {"n": 0}

    class Wrapper:
        def __init__(self, db):
            self._db = db

        def execute(self, sql, params=()):
            if sql.startswith("SELECT dataset_id") and calls["n"] == 0:
                calls["n"] += 1
                return self._db.execute("SELECT dataset_id FROM datasets WHERE 0")

            return self._db.execute(sql, params)

        def __getattr__(self, name):
            return getattr(self._db, name)

    store.db = Wrapper(store.db)

    assert store.add_dataset("d", "1", "c" * 64, {}) == first
