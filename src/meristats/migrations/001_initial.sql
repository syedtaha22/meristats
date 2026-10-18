-- meristats, a testing and results framework for the Meridian chess engine
-- Copyright (C) 2026 Syed Taha
--
-- This program is free software: you can redistribute it and/or modify
-- it under the terms of the GNU General Public License as published by
-- the Free Software Foundation, either version 3 of the License, or
-- (at your option) any later version.
--
-- This program is distributed in the hope that it will be useful,
-- but WITHOUT ANY WARRANTY; without even the implied warranty of
-- MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
-- GNU General Public License for more details.
--
-- You should have received a copy of the GNU General Public License
-- along with this program.  If not, see <https://www.gnu.org/licenses/>.

CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE machines (
    machine_id    INTEGER PRIMARY KEY,
    fingerprint   TEXT NOT NULL UNIQUE,
    cpu_model     TEXT NOT NULL,
    logical_cpus  INTEGER NOT NULL,
    kernel        TEXT NOT NULL,
    os            TEXT NOT NULL,
    json          TEXT NOT NULL
);

CREATE TABLE engines (
    engine_id     INTEGER PRIMARY KEY,
    kind          TEXT NOT NULL CHECK (kind IN ('meridian', 'external')),
    name          TEXT NOT NULL,
    commit_sha    TEXT,
    binary_sha256 TEXT NOT NULL,
    build_json    TEXT NOT NULL,
    bench_nodes   INTEGER,
    version_text  TEXT NOT NULL,
    UNIQUE (kind, commit_sha, binary_sha256)
);

CREATE TABLE datasets (
    dataset_id       INTEGER PRIMARY KEY,
    name             TEXT NOT NULL,
    version          TEXT NOT NULL,
    sha256           TEXT NOT NULL,
    description_json TEXT NOT NULL,
    UNIQUE (name, sha256)
);

CREATE TABLE blobs (
    sha256        TEXT PRIMARY KEY,
    size_bytes    INTEGER NOT NULL,
    stored_bytes  INTEGER NOT NULL,
    kind          TEXT NOT NULL,
    original_name TEXT
);

CREATE TABLE runs (
    run_id           TEXT PRIMARY KEY,
    protocol         TEXT NOT NULL,
    protocol_version INTEGER NOT NULL,
    grade            TEXT NOT NULL CHECK (grade IN ('verified', 'unverified')),
    status           TEXT NOT NULL CHECK (status IN ('complete', 'failed', 'aborted')),
    started_utc      TEXT NOT NULL,
    finished_utc     TEXT NOT NULL,
    engine_id        INTEGER REFERENCES engines (engine_id),
    config           TEXT,
    limits_json      TEXT NOT NULL,
    dataset_id       INTEGER REFERENCES datasets (dataset_id),
    seed             INTEGER,
    machine_id       INTEGER REFERENCES machines (machine_id),
    conditions_json  TEXT NOT NULL,
    code_commit      TEXT,
    code_dirty       INTEGER,
    schema_version   INTEGER NOT NULL,
    supersedes       TEXT REFERENCES runs (run_id),
    summary_json     TEXT NOT NULL,
    items_blob       TEXT REFERENCES blobs (sha256),
    notes            TEXT
);

CREATE TABLE run_blobs (
    run_id TEXT NOT NULL REFERENCES runs (run_id),
    sha256 TEXT NOT NULL REFERENCES blobs (sha256),
    role   TEXT NOT NULL,
    PRIMARY KEY (run_id, sha256, role)
);

-- Rows are never changed or removed.
CREATE TRIGGER machines_no_update BEFORE UPDATE ON machines
BEGIN SELECT RAISE(ABORT, 'machines rows are immutable'); END;
CREATE TRIGGER machines_no_delete BEFORE DELETE ON machines
BEGIN SELECT RAISE(ABORT, 'machines rows are immutable'); END;
CREATE TRIGGER engines_no_update BEFORE UPDATE ON engines
BEGIN SELECT RAISE(ABORT, 'engines rows are immutable'); END;
CREATE TRIGGER engines_no_delete BEFORE DELETE ON engines
BEGIN SELECT RAISE(ABORT, 'engines rows are immutable'); END;
CREATE TRIGGER datasets_no_update BEFORE UPDATE ON datasets
BEGIN SELECT RAISE(ABORT, 'datasets rows are immutable'); END;
CREATE TRIGGER datasets_no_delete BEFORE DELETE ON datasets
BEGIN SELECT RAISE(ABORT, 'datasets rows are immutable'); END;
CREATE TRIGGER blobs_no_update BEFORE UPDATE ON blobs
BEGIN SELECT RAISE(ABORT, 'blobs rows are immutable'); END;
CREATE TRIGGER blobs_no_delete BEFORE DELETE ON blobs
BEGIN SELECT RAISE(ABORT, 'blobs rows are immutable'); END;
CREATE TRIGGER runs_no_update BEFORE UPDATE ON runs
BEGIN SELECT RAISE(ABORT, 'runs rows are immutable'); END;
CREATE TRIGGER runs_no_delete BEFORE DELETE ON runs
BEGIN SELECT RAISE(ABORT, 'runs rows are immutable'); END;
CREATE TRIGGER run_blobs_no_update BEFORE UPDATE ON run_blobs
BEGIN SELECT RAISE(ABORT, 'run_blobs rows are immutable'); END;
CREATE TRIGGER run_blobs_no_delete BEFORE DELETE ON run_blobs
BEGIN SELECT RAISE(ABORT, 'run_blobs rows are immutable'); END;
