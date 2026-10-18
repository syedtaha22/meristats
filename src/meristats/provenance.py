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
import platform
import subprocess
from pathlib import Path

UNAVAILABLE = "unavailable"
CPUINFO = Path("/proc/cpuinfo")
POWER_SUPPLY = Path("/sys/class/power_supply")
CPU_DIR = Path("/sys/devices/system/cpu")


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def machine_info() -> dict:
    """
    Describes the machine: CPU model, logical CPUs, kernel and operating system. A value that the
    system does not give is the text `unavailable`.
    """
    model = UNAVAILABLE

    try:
        for line in CPUINFO.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        model = platform.processor() or UNAVAILABLE

    return {
        "cpu_model": model,
        "logical_cpus": os.cpu_count() or 0,
        "architecture": platform.machine() or UNAVAILABLE,
        "kernel": platform.release() or UNAVAILABLE,
        "os": platform.platform() or UNAVAILABLE,
    }


def power_source() -> str:
    """
    `ac` when a mains supply is online, `battery` when a battery is discharging, and
    `unavailable` when the system exposes neither.
    """
    try:
        supplies = sorted(POWER_SUPPLY.iterdir())
    except OSError:
        return UNAVAILABLE

    on_battery = False

    for supply in supplies:
        kind = _read(supply / "type")

        if kind == "Mains" and _read(supply / "online") == "1":
            return "ac"

        if kind == "Battery" and _read(supply / "status") == "Discharging":
            on_battery = True

    return "battery" if on_battery else UNAVAILABLE


def cpu_frequency() -> dict:
    """
    The frequency of the logical CPUs in kHz, as the minimum, the mean and the maximum, or
    `unavailable`.
    """
    values = []

    try:
        for cpu in CPU_DIR.glob("cpu[0-9]*"):
            text = _read(cpu / "cpufreq" / "scaling_cur_freq")

            if text and text.isdigit():
                values.append(int(text))
    except OSError:
        pass

    if not values:
        return {"khz": UNAVAILABLE}

    return {
        "khz_min": min(values),
        "khz_mean": round(sum(values) / len(values)),
        "khz_max": max(values),
    }


def conditions() -> dict:
    """
    The conditions of a moment: power source, governor, CPU frequency and the 1-minute load
    average. Nothing here is enforced.
    """
    try:
        load = os.getloadavg()[0]
    except OSError:
        load = UNAVAILABLE

    return {
        "power_source": power_source(),
        "governor": _read(CPU_DIR / "cpu0" / "cpufreq" / "scaling_governor") or UNAVAILABLE,
        "frequency": cpu_frequency(),
        "load_1min": load,
    }


def git(repo: Path, *args: str) -> str | None:
    """
    The output of a git command in `repo`, or None when it fails.
    """
    try:
        done = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    return done.stdout.strip()


def git_state(repo: Path) -> tuple[str | None, bool | None]:
    """
    The commit of `HEAD` and whether the working tree has changes, for a git repository, and
    (None, None) otherwise. Ignored files do not count as changes.
    """
    commit = git(repo, "rev-parse", "HEAD")

    if commit is None:
        return None, None

    status = git(repo, "status", "--porcelain")
    return commit, bool(status)


def framework_root() -> Path | None:
    """
    The folder of the meristats repository, when this code runs from a checkout of it.
    """
    root = Path(__file__).resolve().parents[2]
    return root if (root / ".git").exists() else None
