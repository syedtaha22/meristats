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

import pytest

from meristats import config as settings
from meristats.config import ConfigError, load

GOOD = """
engine_repo = "~/engine"
data_dir = "/data/m"
cache_dir = "/cache/m"
workers = 3
fastchess = "/bin/fastchess"
"""


def test_load_reads_all_keys_and_expands_home(tmp_path):
    file = tmp_path / "m.toml"
    file.write_text(GOOD)
    loaded = load(file)

    assert loaded.workers == 3
    assert str(loaded.engine_repo).endswith("/engine") and "~" not in str(loaded.engine_repo)
    assert str(loaded.fastchess) == "/bin/fastchess"
    assert loaded.stockfish is None


@pytest.mark.parametrize("missing", ["engine_repo", "data_dir", "cache_dir", "workers"])
def test_missing_required_key_is_named(tmp_path, missing):
    file = tmp_path / "m.toml"
    file.write_text("\n".join(line for line in GOOD.splitlines() if not line.startswith(missing)))

    with pytest.raises(ConfigError, match=missing):
        load(file)


@pytest.mark.parametrize("value", ["0", "-2", "1.5", "true", '"4"'])
def test_workers_must_be_a_positive_whole_number(tmp_path, value):
    file = tmp_path / "m.toml"
    file.write_text(GOOD.replace("workers = 3", f"workers = {value}"))

    with pytest.raises(ConfigError, match="workers"):
        load(file)


def test_broken_toml_is_a_config_error(tmp_path):
    file = tmp_path / "m.toml"
    file.write_text("this is = = not toml")

    with pytest.raises(ConfigError, match="cannot read"):
        load(file)


def test_find_uses_the_environment_when_no_path_is_given(tmp_path, monkeypatch):
    file = tmp_path / "env.toml"
    file.write_text(GOOD)
    monkeypatch.setenv("MERISTATS_CONFIG", str(file))
    monkeypatch.chdir(tmp_path)

    assert settings.find() == file


def test_find_without_any_file_explains_the_options(tmp_path, monkeypatch):
    monkeypatch.delenv("MERISTATS_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConfigError, match="MERISTATS_CONFIG"):
        settings.find()
