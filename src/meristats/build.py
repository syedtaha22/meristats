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
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path

from meristats.config import Config
from meristats.provenance import git
from meristats.store import Store
from meristats.uci import UciEngine

BENCH_DEPTH = 10
DEFAULT_HIDDEN = 128
NODES = re.compile(r"Nodes searched\s*:\s*(\d+)")


class BuildError(Exception):
    """
    The engine could not be built, or the build does not match an earlier build of the commit.
    """


@dataclass(frozen=True)
class BuiltEngine:
    engine_id: int
    commit_sha: str
    binary: Path
    binary_sha256: str
    bench_nodes: int | None
    version_text: str
    build: dict


def resolve_commit(repo: Path, ref: str) -> str:
    """
    The full SHA of a commit of `repo`. A branch or a tag is resolved, and the SHA is what is
    recorded.
    """
    sha = git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")

    if sha is None:
        raise BuildError(f"{ref} is not a commit of {repo}")

    return sha


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


def export_source(repo: Path, sha: str, destination: Path) -> None:
    """
    Writes the files of commit `sha` into `destination`, without the working tree of `repo`.
    """
    if destination.exists():
        shutil.rmtree(destination)

    destination.mkdir(parents=True)
    archive = subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", sha], capture_output=True, check=False
    )

    if archive.returncode != 0:
        raise BuildError(f"git archive {sha} failed: {archive.stderr.decode()[-300:]}")

    import io

    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(destination, filter="data")


def run_bench(binary: Path, depth: int = BENCH_DEPTH, timeout: float = 600.0):
    """
    Runs `binary bench <depth>` and returns the nodes it reports, None when they are missing from
    the output (an engine without `bench`).
    """
    try:
        done = subprocess.run(
            [str(binary), "bench", str(depth)], capture_output=True, text=True, timeout=timeout
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    nodes = NODES.search(done.stdout)
    return int(nodes.group(1)) if nodes else None


def _cache_value(cache_file: Path, key: str) -> str:
    try:
        for line in cache_file.read_text().splitlines():
            if line.startswith(key + ":"):
                return line.split("=", 1)[1]
    except OSError:
        pass

    return "unavailable"


def _first_line(command: list[str]) -> str:
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=30)
        return (done.stdout or done.stderr).splitlines()[0].strip()
    except (OSError, subprocess.TimeoutExpired, IndexError):
        return "unavailable"


def define_flags(defines: dict[str, str] | None) -> list[str]:
    """
    The compiler flags for macros of the engine, such as `CHESS_NNUE_HIDDEN=256`, in a fixed order.
    """
    return [f"-D{name}={value}" for name, value in sorted((defines or {}).items())]


def width_defines(hidden: int | None) -> dict[str, str] | None:
    """
    The macros that build the engine for a network of `hidden` values for each side, or None for
    the width that the engine has by default.
    """
    return None if hidden in (None, DEFAULT_HIDDEN) else {"CHESS_NNUE_HIDDEN": str(hidden)}


def build_name(sha: str, defines: dict[str, str] | None) -> str:
    """
    The folder name of a build: the commit, and the macros when there are some.
    """
    flags = "-".join(f"{name}-{value}" for name, value in sorted((defines or {}).items()))
    return f"{sha}-{flags}" if flags else sha


def build_engine(
    config: Config,
    store: Store,
    ref: str,
    *,
    bench_depth: int = BENCH_DEPTH,
    jobs: int | None = None,
    target: str = "meridian",
    defines: dict[str, str] | None = None,
) -> BuiltEngine:
    """
    Builds the engine at a commit of the engine repository and records the build.

    The source is exported with `git archive`, so changes in the working tree never enter the
    build. The build is a Release build, and its flags are recorded. The binary is hashed, and
    `meridian bench` gives the node count, the fingerprint of the behaviour of the commit. A
    commit that was built before must give the same node count again. `defines` sets macros of the
    engine for this build, such as the width of the network, and a build with other macros is
    another build.
    """
    sha = resolve_commit(config.engine_repo, ref)
    source = config.cache_dir / "src" / sha
    build_dir = config.cache_dir / "build" / build_name(sha, defines)
    binary = build_dir / target

    if not binary.exists():
        export_source(config.engine_repo, sha, source)
        build_dir.mkdir(parents=True, exist_ok=True)
        steps = [
            [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(build_dir),
                "-DCMAKE_BUILD_TYPE=Release",
                *([f"-DCMAKE_CXX_FLAGS={' '.join(define_flags(defines))}"] if defines else []),
            ],
            [
                "cmake",
                "--build",
                str(build_dir),
                "-j",
                str(jobs or os.cpu_count() or 1),
                "--target",
                target,
            ],
        ]

        for step in steps:
            done = subprocess.run(step, capture_output=True, text=True)

            if done.returncode != 0:
                tail = "\n".join((done.stdout + done.stderr).splitlines()[-15:])
                raise BuildError(f"{' '.join(step[:2])} failed for {sha[:8]}:\n{tail}")

        if not binary.exists():
            raise BuildError(f"the build of {sha[:8]} produced no {target}")

    cache = build_dir / "CMakeCache.txt"
    compiler = _cache_value(cache, "CMAKE_CXX_COMPILER")
    build = {
        "build_type": _cache_value(cache, "CMAKE_BUILD_TYPE"),
        "cxx_flags_release": _cache_value(cache, "CMAKE_CXX_FLAGS_RELEASE"),
        "compiler": compiler,
        "compiler_version": _first_line([compiler, "--version"])
        if compiler != "unavailable"
        else "unavailable",
        "cmake_version": _first_line(["cmake", "--version"]),
        "defines": dict(sorted((defines or {}).items())),
    }

    nodes = run_bench(binary, bench_depth)

    with UciEngine(binary) as engine:
        version_text = engine.name

    digest = file_sha256(binary)

    for row in store.find_engines(sha):
        if json.loads(row["build_json"]).get("defines", {}) != build["defines"]:
            continue

        if row["bench_nodes"] is not None and nodes is not None and row["bench_nodes"] != nodes:
            raise BuildError(
                f"commit {sha[:8]} gave {nodes} bench nodes, and an earlier build gave "
                f"{row['bench_nodes']}: the same source must give the same count"
            )

    engine_id = store.add_engine("meridian", "meridian", sha, digest, build, nodes, version_text)
    return BuiltEngine(engine_id, sha, binary, digest, nodes, version_text, build)
