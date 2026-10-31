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

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

REQUIRED = ("engine_repo", "data_dir", "cache_dir", "workers")


class ConfigError(Exception):
    """
    A settings file is missing, unreadable or incomplete.
    """


@dataclass(frozen=True)
class Config:
    """
    The settings of one machine: where the engine repository and the data live, and how many
    worker processes to use.
    """

    engine_repo: Path
    data_dir: Path
    cache_dir: Path
    workers: int
    fastchess: Path | None = None
    stockfish: Path | None = None
    trainer_python: Path | None = None
    networks_branch: str = "nightly-networks"
    anchor_positions: Path | None = None
    leela_positions: Path | None = None
    engine_repository: str | None = None
    site_repository: str | None = None
    site_worktree: Path | None = None
    app_id: int | None = None
    app_key: Path | None = None
    commit_name: str | None = None
    commit_email: str | None = None
    mail_settings: Path | None = None


def _path(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser()


def find(explicit: Path | None = None) -> Path:
    """
    Returns the settings file: the given path, then `$MERISTATS_CONFIG`, then `meristats.toml` in
    the current folder.
    """
    for candidate in (explicit, os.environ.get("MERISTATS_CONFIG"), "meristats.toml"):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise ConfigError(
        "no settings file: pass --config, set MERISTATS_CONFIG, or create "
        "meristats.toml (see meristats.toml.example)"
    )


def load(path: Path | None = None) -> Config:
    """
    Reads the settings file. `workers` has no default.
    """
    file = find(path)

    try:
        raw = tomllib.loads(file.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"cannot read {file}: {error}") from error

    missing = [key for key in REQUIRED if key not in raw]

    if missing:
        raise ConfigError(f"{file} lacks: {', '.join(missing)}")

    workers = raw["workers"]

    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ConfigError(f"{file}: workers must be a whole number of at least 1")

    return Config(
        engine_repo=_path(raw["engine_repo"]),
        data_dir=_path(raw["data_dir"]),
        cache_dir=_path(raw["cache_dir"]),
        workers=workers,
        fastchess=_path(raw["fastchess"]) if raw.get("fastchess") else None,
        stockfish=_path(raw["stockfish"]) if raw.get("stockfish") else None,
        trainer_python=_path(raw["trainer_python"]) if raw.get("trainer_python") else None,
        networks_branch=raw.get("networks_branch", "nightly-networks"),
        anchor_positions=_path(raw["anchor_positions"]) if raw.get("anchor_positions") else None,
        leela_positions=_path(raw["leela_positions"]) if raw.get("leela_positions") else None,
        engine_repository=raw.get("engine_repository") or None,
        site_repository=raw.get("site_repository") or None,
        site_worktree=_path(raw["site_worktree"]) if raw.get("site_worktree") else None,
        app_id=raw.get("app_id") or None,
        app_key=_path(raw["app_key"]) if raw.get("app_key") else None,
        commit_name=raw.get("commit_name") or None,
        commit_email=raw.get("commit_email") or None,
        mail_settings=_path(raw["mail_settings"]) if raw.get("mail_settings") else None,
    )
