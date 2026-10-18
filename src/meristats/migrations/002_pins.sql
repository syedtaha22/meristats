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

-- A pin puts a run on a page of the site under a label. The newest pin for a page and a label is
-- the one that counts, and a pin without a run takes the label off the page. Rows are never
-- changed or removed.
CREATE TABLE pins (
    pin_id      INTEGER PRIMARY KEY,
    page        TEXT NOT NULL,
    label       TEXT NOT NULL,
    run_id      TEXT REFERENCES runs (run_id),
    pinned_utc  TEXT NOT NULL
);

CREATE TRIGGER pins_no_update BEFORE UPDATE ON pins
BEGIN SELECT RAISE(ABORT, 'pins rows are immutable'); END;
CREATE TRIGGER pins_no_delete BEFORE DELETE ON pins
BEGIN SELECT RAISE(ABORT, 'pins rows are immutable'); END;
