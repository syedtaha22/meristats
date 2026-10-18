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

from meristats import provenance
from meristats.provenance import UNAVAILABLE, git_state


def make_supply(root, name, kind, **files):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "type").write_text(kind + "\n")

    for key, value in files.items():
        (folder / key).write_text(value + "\n")


def test_machine_info_has_the_fields_of_the_machines_table():
    info = provenance.machine_info()

    assert {"cpu_model", "logical_cpus", "architecture", "kernel", "os"} <= set(info)
    assert info["logical_cpus"] >= 1 and info["cpu_model"]


def test_power_source_is_ac_when_a_mains_supply_is_online(tmp_path, monkeypatch):
    make_supply(tmp_path, "BAT0", "Battery", status="Charging")
    make_supply(tmp_path, "AC", "Mains", online="1")
    monkeypatch.setattr(provenance, "POWER_SUPPLY", tmp_path)

    assert provenance.power_source() == "ac"


def test_power_source_is_battery_when_discharging(tmp_path, monkeypatch):
    make_supply(tmp_path, "BAT0", "Battery", status="Discharging")
    make_supply(tmp_path, "AC", "Mains", online="0")
    monkeypatch.setattr(provenance, "POWER_SUPPLY", tmp_path)

    assert provenance.power_source() == "battery"


def test_power_source_is_unavailable_without_supplies(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "POWER_SUPPLY", tmp_path / "missing")
    assert provenance.power_source() == UNAVAILABLE

    monkeypatch.setattr(provenance, "POWER_SUPPLY", tmp_path)
    assert provenance.power_source() == UNAVAILABLE


def test_cpu_frequency_summarises_the_cores(tmp_path, monkeypatch):
    for number, khz in enumerate((1000, 3000, 2000)):
        folder = tmp_path / f"cpu{number}" / "cpufreq"
        folder.mkdir(parents=True)
        (folder / "scaling_cur_freq").write_text(f"{khz}\n")

    monkeypatch.setattr(provenance, "CPU_DIR", tmp_path)

    assert provenance.cpu_frequency() == {"khz_min": 1000, "khz_mean": 2000, "khz_max": 3000}


def test_cpu_frequency_is_unavailable_without_cpufreq(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "CPU_DIR", tmp_path)

    assert provenance.cpu_frequency() == {"khz": UNAVAILABLE}
    assert provenance.conditions()["governor"] == UNAVAILABLE


def test_conditions_has_every_field_and_enforces_nothing():
    conditions = provenance.conditions()

    assert set(conditions) == {"power_source", "governor", "frequency", "load_1min"}


def test_git_state_outside_a_repository_is_none(tmp_path):
    assert git_state(tmp_path) == (None, None)
    assert provenance.git(tmp_path, "rev-parse", "HEAD") is None


def test_machine_info_without_cpuinfo_uses_the_platform(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "CPUINFO", tmp_path / "missing")
    monkeypatch.setattr(provenance.platform, "processor", lambda: "")

    assert provenance.machine_info()["cpu_model"] == UNAVAILABLE

    monkeypatch.setattr(provenance.platform, "processor", lambda: "x86 test")
    assert provenance.machine_info()["cpu_model"] == "x86 test"


def test_machine_info_reads_the_model_name(tmp_path, monkeypatch):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor\t: 0\nmodel name\t: Test CPU 9000\nmodel name\t: second\n")
    monkeypatch.setattr(provenance, "CPUINFO", cpuinfo)

    assert provenance.machine_info()["cpu_model"] == "Test CPU 9000"


def test_conditions_without_a_load_average(monkeypatch):
    def broken():
        raise OSError("no loadavg")

    monkeypatch.setattr(provenance.os, "getloadavg", broken)

    assert provenance.conditions()["load_1min"] == UNAVAILABLE


def test_framework_root_is_none_outside_a_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "__file__", str(tmp_path / "a" / "b" / "provenance.py"))

    assert provenance.framework_root() is None

    (tmp_path / ".git").mkdir()
    assert provenance.framework_root() == tmp_path


def test_a_frequency_file_that_is_not_a_number_is_ignored(tmp_path, monkeypatch):
    folder = tmp_path / "cpu0" / "cpufreq"
    folder.mkdir(parents=True)
    (folder / "scaling_cur_freq").write_text("<unsupported>\n")
    monkeypatch.setattr(provenance, "CPU_DIR", tmp_path)

    assert provenance.cpu_frequency() == {"khz": UNAVAILABLE}
