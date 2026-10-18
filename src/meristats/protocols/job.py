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
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from meristats import runner
from meristats.build import BuiltEngine, BuildError
from meristats.config import Config
from meristats.store import Store

PROTOCOL = "job"
VERSION = 1
TOOLS = ("train_policy", "build_book")
REPOSITORY_TOOLS = Path(__file__).resolve().parents[3] / "tools"
CMAKE_INCLUDE = Path(__file__).resolve().parents[1] / "cmake" / "meristats_tools.cmake"
ROLES = {".nnue": "network", ".params": "params"}


class JobError(Exception):
    """
    A tool could not be built, or did not write the files that it was asked to.
    """


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


def build_tool(
    config: Config, engine: BuiltEngine, tool: str, tools_dir: Path = REPOSITORY_TOOLS
) -> Path:
    """
    Compiles a C++ tool against the engine at the commit of `engine`. The source of the engine is
    the export that `build_engine` made, and the tool is built with the flags of the engine and
    linked to its library `chess_core`. Returns the program.
    """
    if tool not in TOOLS:
        raise JobError(f"{tool} is not a tool: {', '.join(TOOLS)}")

    source = config.cache_dir / "src" / engine.commit_sha
    build = config.cache_dir / "build-tools" / engine.commit_sha
    program = build / tool
    sources = tools_dir / f"{tool}.cpp"

    if not sources.exists():
        raise JobError(f"no source {sources}")

    steps = [
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_PROJECT_INCLUDE={CMAKE_INCLUDE}",
            f"-DMERISTATS_TOOLS={tool}",
            f"-DMERISTATS_TOOLS_DIR={tools_dir}",
        ],
        ["cmake", "--build", str(build), "-j", str(os.cpu_count() or 1), "--target", tool],
    ]

    for step in steps:
        done = subprocess.run(step, capture_output=True, text=True)

        if done.returncode != 0:
            tail = "\n".join((done.stdout + done.stderr).splitlines()[-15:])
            raise BuildError(
                f"{tool} at {engine.commit_sha[:8]}: {' '.join(step[:2])} failed:\n{tail}"
            )

    return program


def _role(name: str) -> str:
    return ROLES.get(Path(name).suffix, "output")


def run(
    config: Config,
    store: Store,
    engine: BuiltEngine,
    tool: str,
    args: list[str],
    *,
    inputs: list[Path] = (),
    outputs: list[str] = (),
    tools_dir: Path = REPOSITORY_TOOLS,
    timeout: float | None = None,
    allow_dirty: bool = False,
    framework=...,
    progress=lambda text: print(text, file=sys.stderr, flush=True),
) -> tuple[str, dict]:
    """
    Builds a tool at the commit of an engine, runs it, and stores the run.

    The run holds the command line, the hash of the program and of its source, the engine commit,
    the framework commit (the version of the tool), the hash and size of every input file (the
    inputs themselves are not copied), every output file as a blob, the exit status and the time.
    The tool runs in a fresh folder, and `outputs` are the names of the files, relative to that
    folder, that it must write. A tool that exits with an error, or that does not write an output,
    is stored as a failed run.
    """
    context = runner.begin(config, store, allow_dirty=allow_dirty, framework=framework)
    binary = build_tool(config, engine, tool, tools_dir)
    program = [str(binary)]
    binary_hash = _sha256(binary)
    source_hash = _sha256(tools_dir / f"{tool}.cpp")

    for path in inputs:
        if not Path(path).is_file():
            raise JobError(f"input {path} is not a file")

    context.checks += runner.engine_checks(engine)
    context.sample("before")
    jobs = config.cache_dir / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    work = Path(
        tempfile.mkdtemp(prefix=f"{context.started_utc.replace(':', '')}-{tool}-", dir=jobs)
    )
    started = time.monotonic()
    done = subprocess.run(
        [*program, *args], cwd=work, capture_output=True, text=True, timeout=timeout
    )
    seconds = time.monotonic() - started
    context.sample("end")
    log = (done.stdout + done.stderr).encode()
    log_blob = store.put_blob(log, "tool-log", f"{tool}-output.txt")
    blobs = [(log_blob.sha256, "tool-log")]
    produced, missing = [], []

    for name in outputs:
        path = work / name

        if not path.is_file():
            missing.append(name)
            continue

        blob = store.put_blob(path.read_bytes(), "output", name)
        blobs.append((blob.sha256, _role(name)))
        produced.append(
            {"name": name, "sha256": blob.sha256, "bytes": blob.size_bytes, "role": _role(name)}
        )

    status = "complete" if done.returncode == 0 and not missing else "failed"
    summary = {
        "tool": tool,
        "args": args,
        "exit_code": done.returncode,
        "seconds": seconds,
        "tool_sha256": binary_hash,
        "tool_source_sha256": source_hash,
        "engine_commit": engine.commit_sha,
        "inputs": [
            {"path": str(p), "sha256": _sha256(Path(p)), "bytes": Path(p).stat().st_size}
            for p in inputs
        ],
        "outputs": produced,
        "missing": missing,
    }
    run_id = runner.finish(
        context,
        protocol=PROTOCOL,
        version=VERSION,
        engine=engine,
        config=None,
        limits={"tool": tool, "args": args, "timeout": timeout},
        dataset_id=None,
        summary=summary,
        items_blob=None,
        blobs=tuple(blobs),
        status=status,
        notes=None
        if status == "complete"
        else f"exit {done.returncode}" + (f", missing {', '.join(missing)}" if missing else ""),
    )
    progress(f"{tool}: exit {done.returncode} in {seconds:.1f}s, {len(produced)} outputs")
    return run_id, summary
