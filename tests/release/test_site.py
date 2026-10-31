"""
Tests of the publication of the cycles on the site.
"""

import json
import subprocess

import pytest

from meristats.release import site
from meristats.release.settings import Site

RUN = dict(
    protocol="match",
    protocol_version=1,
    grade="verified",
    status="complete",
    started_utc="2026-10-18T05:00:00Z",
    finished_utc="2026-10-18T05:10:00Z",
    engine_id=None,
    config="network",
    limits={"nodes": 50000},
    dataset_id=None,
    seed=1,
    machine_id=None,
    conditions={},
    code_commit="abc",
    code_dirty=False,
    summary={"games": 10},
)


def run(*command, cwd=None):
    done = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def write_cycle(folder, name, run_id=None):
    path = folder / name
    path.mkdir(parents=True)
    stages = {"match": {"run": run_id}} if run_id else {}
    (path / "summary.json").write_text(json.dumps({"cycle": name, "stages": stages}))


@pytest.fixture
def setup(tmp_path, monkeypatch, store):
    remote = tmp_path / "remote.git"
    run("git", "init", "--quiet", "--bare", str(remote))
    seed = tmp_path / "seed"
    run("git", "clone", "--quiet", str(remote), str(seed))
    run("git", "checkout", "--quiet", "-b", "site", cwd=seed)
    (seed / "index.html").write_text("")
    run("git", "-c", "user.name=t", "-c", "user.email=t@e", "add", "-A", cwd=seed)
    run("git", "-c", "user.name=t", "-c", "user.email=t@e", "commit", "--quiet", "-m", "x", cwd=seed)
    run("git", "push", "--quiet", "origin", "site", cwd=seed)
    clone = tmp_path / "meristats"
    clone.mkdir()
    run("git", "init", "--quiet", cwd=clone)

    def write(directory, the_store):
        pins = "\n".join(f"{pin['label']} {pin['run_id']}" for pin in the_store.pins())
        (directory / "index.html").write_text(pins)
        return [directory / "index.html"]

    monkeypatch.setattr(site, "write_site", write)
    target = Site(
        directory=clone / "site-build",
        repository="owner/meristats",
        cycles=tmp_path / "cycles",
        author_name="bot[bot]",
        author_email="1+bot[bot]@users.noreply.github.com",
        remote=str(remote),
    )
    return target, remote, store


def head(remote):
    return run("git", "--git-dir", str(remote), "rev-parse", "site")


def test_a_label_is_the_time_of_the_cycle():
    assert site.label_of("20261031T031500Z") == "Cycle 2026-10-31 03:15 UTC"
    assert site.label_of("2026-10-31") == "Cycle 2026-10-31"


def test_a_new_cycle_is_pinned_published_and_pushed_by_the_app(setup):
    target, remote, store = setup
    run_id = store.add_run(**RUN)
    write_cycle(target.cycles, "20261031T031500Z", run_id)
    commit = site.publish(target, store)

    assert head(remote) == commit
    assert run("git", "--git-dir", str(remote), "log", "-1", "--format=%an|%ae|%s", "site") == (
        "bot[bot]|1+bot[bot]@users.noreply.github.com|"
        "Publish the match of cycle 2026-10-31 03:15 utc"
    )
    assert [pin["label"] for pin in store.pins()] == ["Cycle 2026-10-31 03:15 UTC"]


def test_a_cycle_that_is_pinned_is_not_pinned_again_and_nothing_is_committed(setup):
    target, remote, store = setup
    write_cycle(target.cycles, "20261031T031500Z", store.add_run(**RUN))
    site.publish(target, store)

    assert site.publish(target, store) is None
    assert len(store.pins()) == 1


def test_a_cycle_without_a_match_is_left_out(setup):
    target, remote, store = setup
    write_cycle(target.cycles, "20261030T031500Z")
    write_cycle(target.cycles, "20261031T031500Z", store.add_run(**RUN))
    site.publish(target, store)

    assert [pin["label"] for pin in store.pins()] == ["Cycle 2026-10-31 03:15 UTC"]


def test_several_new_cycles_make_one_commit(setup):
    target, remote, store = setup
    write_cycle(target.cycles, "20261030T031500Z", store.add_run(**RUN))
    write_cycle(target.cycles, "20261031T031500Z", store.add_run(**{**RUN, "started_utc": "2026-10-19T05:00:00Z"}))
    site.publish(target, store)

    assert run("git", "--git-dir", str(remote), "log", "-1", "--format=%s", "site") == (
        "Publish the matches of 2 cycles"
    )


def test_the_site_is_set_back_to_the_remote_before_publishing(setup):
    target, remote, store = setup
    write_cycle(target.cycles, "20261031T031500Z", store.add_run(**RUN))
    site.publish(target, store)
    (target.directory / "stray.html").write_text("x")
    write_cycle(target.cycles, "20261101T031500Z", store.add_run(**{**RUN, "started_utc": "2026-10-19T05:00:00Z"}))
    site.publish(target, store)

    assert "stray.html" not in run("git", "--git-dir", str(remote), "ls-tree", "--name-only", "site")
