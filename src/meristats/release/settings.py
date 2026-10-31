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

"""
Where the release steps work, taken from the settings of meristats (see config.py).
"""

from dataclasses import dataclass
from pathlib import Path

from meristats.config import Config
from meristats.release.common import ReleaseError


@dataclass(frozen=True)
class Target:
    """
    The engine clone and its repository on GitHub: where a network is committed and pushed.
    """

    engine: Path
    repository: str
    ledger: Path
    branch: str
    base: str
    build: Path
    author_name: str
    author_email: str


@dataclass(frozen=True)
class Site:
    """
    The worktree of the branch `site` of meristats and its repository on GitHub.
    """

    directory: Path
    repository: str
    cycles: Path
    author_name: str
    author_email: str
    remote: str | None = None


def missing(config: Config, *names: str) -> list[str]:
    return [name for name in names if getattr(config, name) in (None, "")]


def require(config: Config, *names: str) -> None:
    left = missing(config, *names)

    if left:
        raise ReleaseError("meristats.toml lacks: " + ", ".join(left))


def target_of(config: Config) -> Target:
    require(config, "engine_repository", "commit_name", "commit_email")
    return Target(
        engine=Path(config.engine_repo),
        repository=config.engine_repository,
        ledger=Path(config.data_dir) / "networks-ledger.json",
        branch=config.networks_branch,
        base="main",
        build=Path(config.cache_dir) / "promote-build",
        author_name=config.commit_name,
        author_email=config.commit_email,
    )


def site_of(config: Config) -> Site:
    require(config, "site_repository", "site_worktree", "commit_name", "commit_email")
    return Site(
        directory=Path(config.site_worktree),
        repository=config.site_repository,
        cycles=Path(config.data_dir) / "cycles",
        author_name=config.commit_name,
        author_email=config.commit_email,
    )
