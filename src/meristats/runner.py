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

from dataclasses import dataclass, field
from pathlib import Path

from meristats import preflight, provenance
from meristats.build import BuiltEngine
from meristats.config import Config
from meristats.preflight import Check, require
from meristats.store import Store, utc_now
from meristats.uci import UciEngine


@dataclass
class RunContext:
    """
    What every run records before it starts: when, on which machine, from which state of the
    framework, and the conditions at the start.
    """

    config: Config
    store: Store
    started_utc: str
    machine_id: int
    grade: str
    code_commit: str | None
    code_dirty: bool | None
    checks: list[Check]
    samples: list[dict] = field(default_factory=list)

    def sample(self, label: str) -> None:
        """
        Records the conditions of this moment under a label.
        """
        self.samples.append({"at": label, **provenance.conditions()})

    def conditions(self) -> dict:
        return {
            "samples": self.samples,
            "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in self.checks],
        }


def begin(
    config: Config, store: Store, *, allow_dirty: bool = False, framework: Path | None = ...
) -> RunContext:
    """
    Starts a run: checks that the framework is clean (a dirty tree needs `allow_dirty` and gives
    the grade `unverified`), records the machine and samples the conditions.
    """
    root = provenance.framework_root() if framework is ... else framework
    check, commit, dirty, grade = preflight.framework_clean(root, allow_dirty)
    require([check])
    machine_id = store.add_machine(provenance.machine_info())
    context = RunContext(config, store, utc_now(), machine_id, grade, commit, dirty, [check])
    context.sample("start")
    return context


def engine_checks(engine: BuiltEngine, depth: int = 10) -> list[Check]:
    """
    The checks of an engine that every run needs: it comes from a commit, and `bench` still gives
    the recorded node count.
    """
    return [preflight.engine_from_sha(engine), preflight.fingerprint(engine, depth)]


def apply_options(uci: UciEngine, options: dict[str, str | int], config: str | None) -> list[str]:
    """
    Sends the options to an engine, `Config` last, and returns the replies that the engine gave.
    An option that the engine does not have is skipped when it is not `Config`. A `Config` for an
    engine without that option is an error, because the run would not measure what was asked.
    """
    replies = []

    for name, value in options.items():
        if name in uci.options:
            replies += uci.set_option(name, value)

    if config is not None:
        if "Config" not in uci.options:
            raise preflight.PreflightError(f"{uci.name} has no Config option for '{config}'")

        replies += uci.set_option("Config", config)

    return replies


def finish(
    context: RunContext,
    *,
    protocol: str,
    version: int,
    engine: BuiltEngine | None,
    config: str | None,
    limits: dict,
    dataset_id: int | None,
    summary: dict,
    items_blob: str | None,
    blobs: tuple[tuple[str, str], ...] = (),
    status: str = "complete",
    notes: str | None = None,
    seed: int | None = None,
) -> str:
    """
    Writes the run, with the conditions sampled during it, and returns its id.
    """
    return context.store.add_run(
        protocol=protocol,
        protocol_version=version,
        grade=context.grade,
        status=status,
        started_utc=context.started_utc,
        finished_utc=utc_now(),
        engine_id=engine.engine_id if engine else None,
        config=config,
        limits=limits,
        dataset_id=dataset_id,
        seed=seed,
        machine_id=context.machine_id,
        conditions=context.conditions(),
        code_commit=context.code_commit,
        code_dirty=context.code_dirty,
        summary=summary,
        items_blob=items_blob,
        blobs=blobs,
        notes=notes,
    )
