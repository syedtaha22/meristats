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

import shutil

import pytest

from meristats import cli
from meristats.cli import main

needs_build = pytest.mark.skipif(
    not (shutil.which("cmake") and shutil.which("g++")), reason="needs cmake and g++"
)


def run_main(config_file, *args):
    return main(["--config", str(config_file), *args])


def test_help_and_version_exit_cleanly(capsys):
    with pytest.raises(SystemExit) as version:
        main(["--version"])

    assert version.value.code == 0 and "meristats" in capsys.readouterr().out

    with pytest.raises(SystemExit) as nothing:
        main([])

    assert nothing.value.code == 2


def test_a_missing_settings_file_is_a_clear_error(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MERISTATS_CONFIG", raising=False)

    assert main(["verify"]) == 1
    assert "no settings file" in capsys.readouterr().err


def test_doctor_reports_each_check_and_the_database(config_file, capsys):
    code = run_main(config_file, "doctor")
    out = capsys.readouterr().out

    assert "ok   python" in out and "ok   package chess" in out and "ok   git" in out
    assert "ok   settings: 2 workers" in out and "ok   database: schema 2" in out
    assert "warn stockfish: not set" in out and "warn fastchess: not set" in out
    assert code == 0


def test_doctor_without_settings_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MERISTATS_CONFIG", raising=False)

    assert main(["doctor"]) == 1
    assert "FAIL settings" in capsys.readouterr().out


@needs_build
def test_build_prints_the_commit_hash_and_bench(config_file, tiny_repo, capsys):
    repo, commit = tiny_repo
    sha = commit(321)

    assert run_main(config_file, "build", "HEAD") == 0
    out = capsys.readouterr().out

    assert sha in out and "bench   321 nodes" in out and "sha256" in out


@needs_build
def test_building_a_commit_that_does_not_exist_is_an_error(config_file, tiny_repo, capsys):
    repo, commit = tiny_repo
    commit(1)

    assert run_main(config_file, "build", "nope") == 1
    assert "not a commit" in capsys.readouterr().err


def test_backup_and_verify(config_file, tmp_path, capsys):
    from meristats.config import load
    from meristats.store import Store

    with Store(load(config_file).data_dir) as store:
        store.put_blob(b"data", "t")

    assert run_main(config_file, "verify") == 0
    assert "0 damaged or missing blobs" in capsys.readouterr().out
    assert run_main(config_file, "backup", str(tmp_path / "copy")) == 0
    assert (tmp_path / "copy" / "meristats.db").exists()
    assert run_main(config_file, "backup", str(tmp_path / "copy")) == 1


def test_verify_exits_with_an_error_for_a_damaged_blob(config_file, capsys):
    from meristats.config import load
    from meristats.store import Store

    with Store(load(config_file).data_dir) as store:
        blob = store.put_blob(b"data", "t")
        store._blob_path(blob.sha256).write_bytes(b"damaged")

    assert run_main(config_file, "verify") == 1
    assert blob.sha256 in capsys.readouterr().out


class Recorder:
    def __init__(self, result):
        self.calls = []
        self.result = result

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


@pytest.fixture
def stubbed(monkeypatch, config_file, tiny_repo):
    """
    Replaces `build_engine` and the protocols with recorders, so the command line can be tested
    without running an engine.
    """
    from meristats.build import BuiltEngine

    repo, commit = tiny_repo
    commit(5)
    engine = BuiltEngine(1, "a" * 40, repo / "meridian", "b" * 64, 5, "Tiny", {})
    monkeypatch.setattr(cli, "build_engine", lambda config, store, ref: engine)
    return monkeypatch


def test_run_puzzles_passes_the_limit_and_options(stubbed, config_file, capsys):
    summary = {
        "whole": {
            "p": 0.9,
            "low": 0.89,
            "high": 0.91,
            "first_move": 0.95,
            "puzzle_rating": 2400.0,
            "mean_nodes": 1000.0,
        },
        "errors": {"count": 0},
    }
    recorder = Recorder(("run-1", summary))
    stubbed.setattr(cli.puzzles_protocol, "run", recorder)

    assert (
        run_main(
            config_file,
            "run",
            "puzzles",
            "--engine",
            "x",
            "--depth",
            "4",
            "--items",
            "100",
            "--engine-config",
            "network+see=0",
            "--hash",
            "32",
            "--allow-dirty",
        )
        == 0
    )
    ((args, kwargs),) = recorder.calls
    out = capsys.readouterr().out

    assert args[4] == cli.puzzles_protocol.Limit("depth", 4)
    assert kwargs["engine_config"] == "network+see=0" and kwargs["items"] == 100
    assert kwargs["hash_mb"] == 32 and kwargs["allow_dirty"] is True
    assert "run-1" in out and "solved 90.00%" in out and "puzzle rating 2400" in out


@pytest.mark.parametrize(
    "flags,expected",
    [
        ([], ("nodes", 20000)),
        (["--nodes", "5000"], ("nodes", 5000)),
        (["--movetime", "50"], ("movetime", 50)),
        (["--depth", "6"], ("depth", 6)),
    ],
)
def test_puzzle_limits(stubbed, config_file, flags, expected):
    recorder = Recorder(
        (
            "r",
            {
                "whole": {
                    "p": 1,
                    "low": 1,
                    "high": 1,
                    "first_move": 1,
                    "puzzle_rating": 1,
                    "mean_nodes": 1,
                },
                "errors": {"count": 0},
            },
        )
    )
    stubbed.setattr(cli.puzzles_protocol, "run", recorder)
    run_main(config_file, "run", "puzzles", "--engine", "x", *flags)

    limit = recorder.calls[0][0][4]
    assert (limit.kind, limit.value) == expected


def test_run_search_nodes_parses_configs_and_depths(stubbed, config_file, capsys):
    summary = {
        "network": {"6": {"mean_nodes": 4030.0, "mean_ms": 4.9, "nodes_per_second": 823006.0}}
    }
    recorder = Recorder(("run-2", summary))
    stubbed.setattr(cli.search_nodes_protocol, "run", recorder)

    assert (
        run_main(
            config_file,
            "run",
            "search-nodes",
            "--engine",
            "x",
            "--configs",
            "network,baseline",
            "--depths",
            "4,6",
            "--rounds",
            "2",
            "--cpu",
            "3",
        )
        == 0
    )
    ((args, kwargs),) = recorder.calls

    assert args[4] == ["network", "baseline"] and args[5] == [4, 6]
    assert kwargs["rounds"] == 2 and kwargs["cpu"] == 3
    assert "run-2" in capsys.readouterr().out

    recorder.calls.clear()
    run_main(config_file, "run", "search-nodes", "--engine", "x")
    assert recorder.calls[0][0][4] == [None] and recorder.calls[0][0][5] == [1, 2, 3, 4, 5, 6]


def test_run_rules_exit_status_follows_the_counts(stubbed, config_file, capsys):
    good = {
        "all_correct": True,
        "counts": 26,
        "wrong": [],
        "speed": {"initial": {"depth": 5, "nodes_per_second": 63e6}},
    }
    bad = {"all_correct": False, "counts": 26, "wrong": ["x depth 2: 401 not 400"], "speed": {}}
    stubbed.setattr(cli.rules_protocol, "run", Recorder(("r", good)))
    assert run_main(config_file, "run", "rules", "--engine", "x") == 0
    assert "63.0 Mnps" in capsys.readouterr().out

    stubbed.setattr(cli.rules_protocol, "run", Recorder(("r", bad)))
    assert run_main(config_file, "run", "rules", "--engine", "x") == 1
    assert "WRONG x depth 2" in capsys.readouterr().out


def test_run_match_builds_the_sprt_and_limit(stubbed, config_file, tmp_path, capsys):
    book = tmp_path / "book.pgn"
    book.write_text("x")
    summary = {
        "games": 10,
        "wins": 6,
        "draws": 2,
        "losses": 2,
        "score": 0.7,
        "elo": 150.0,
        "sprt": {
            "elo0": 0.0,
            "elo1": 10.0,
            "llr": 1.5,
            "bounds": [-2.9, 2.9],
            "verdict": "continue",
            "fastchess_verdict": "continue",
        },
    }
    recorder = Recorder(("run-3", summary))
    stubbed.setattr(cli.match_protocol, "run", recorder)

    assert (
        run_main(
            config_file,
            "run",
            "match",
            "--a",
            "x",
            "--b",
            "y",
            "--a-config",
            "network",
            "--b-config",
            "baseline",
            "--depth",
            "4",
            "--rounds",
            "50",
            "--elo1",
            "20",
            "--seed",
            "9",
            "--book",
            str(book),
            "--no-compliance",
        )
        == 0
    )
    ((args, kwargs),) = recorder.calls

    assert args[5] == cli.match_protocol.Limit("depth", 4)
    assert kwargs["sprt"] == cli.match_protocol.Sprt(0.0, 20.0)
    assert kwargs["a_config"] == "network" and kwargs["b_config"] == "baseline"
    assert kwargs["rounds"] == 50 and kwargs["seed"] == 9 and kwargs["check_compliance"] is False
    assert "SPRT [0, 10]: LLR 1.50" in capsys.readouterr().out

    recorder.calls.clear()
    run_main(config_file, "run", "match", "--a", "x", "--b", "y", "--no-sprt", "--book", str(book))
    assert recorder.calls[0][1]["sprt"] is None
    assert recorder.calls[0][0][5] == cli.match_protocol.Limit("st", 0.05)


def test_run_ladder_fetches_stockfish_when_the_settings_have_none(
    stubbed, config_file, tmp_path, capsys
):
    book = tmp_path / "book.pgn"
    book.write_text("x")
    program = tmp_path / "stockfish"
    stubbed.setattr(cli.external, "fetch_stockfish", lambda cache: program)
    stubbed.setattr(cli.external, "register_external", lambda store, name, path: (7, "SF 19"))
    summary = {
        "games": 100,
        "levels": {"2000": {"wins": 40, "draws": 5, "losses": 5}},
        "rating": {"value": 2930.0, "low": 2876.0, "high": 2986.0},
    }
    recorder = Recorder(("run-4", summary))
    stubbed.setattr(cli.ladder_protocol, "run", recorder)

    assert (
        run_main(
            config_file,
            "run",
            "ladder",
            "--engine",
            "x",
            "--levels",
            "2000,2200",
            "--rounds",
            "5",
            "--st",
            "0.1",
            "--book",
            str(book),
        )
        == 0
    )
    ((args, kwargs),) = recorder.calls
    out = capsys.readouterr().out

    assert args[3] == program and args[4] == 7 and args[7] == [2000, 2200]
    assert args[6] == cli.match_protocol.Limit("st", 0.1) and kwargs["rounds"] == 5
    assert "rating 2930 (2876 to 2986) in 100 games" in out and "stockfish-2000: +40 =5 -5" in out


def test_run_job_passes_the_arguments_after_the_double_dash(stubbed, config_file, tmp_path, capsys):
    summary = {
        "exit_code": 0,
        "seconds": 1.0,
        "tool_sha256": "a" * 64,
        "missing": [],
        "outputs": [{"name": "n.nnue", "bytes": 10, "sha256": "b" * 64}],
    }
    recorder = Recorder(("run-5", summary))
    stubbed.setattr(cli.job_protocol, "run", recorder)
    data = tmp_path / "d.txt"
    data.write_text("x")

    assert (
        run_main(
            config_file,
            "run",
            "job",
            "--engine",
            "x",
            "--input",
            str(data),
            "--output",
            "n.nnue",
            "--timeout",
            "5",
            "build_book",
            "--",
            "--epochs",
            "1",
        )
        == 0
    )
    ((args, kwargs),) = recorder.calls

    assert args[3] == "build_book" and args[4] == ["--epochs", "1"]
    assert kwargs["inputs"] == [data] and kwargs["outputs"] == ["n.nnue"]
    assert kwargs["timeout"] == 5.0
    assert "n.nnue  10 bytes  sha256 " + "b" * 64 in capsys.readouterr().out

    summary["missing"] = ["n.nnue"]
    assert run_main(config_file, "run", "job", "--engine", "x", "build_book") == 1


def test_fetch_prints_the_path(stubbed, config_file, tmp_path, capsys):
    stubbed.setattr(cli.external, "fetch_book", lambda cache: tmp_path / "book.pgn")
    stubbed.setattr(cli.external, "fetch_stockfish", lambda cache: tmp_path / "sf")

    assert run_main(config_file, "fetch", "book") == 0
    assert run_main(config_file, "fetch", "stockfish") == 0
    out = capsys.readouterr().out
    assert "book.pgn" in out and "sf" in out


def test_an_external_error_is_reported_without_a_traceback(stubbed, config_file, capsys):
    def fail(cache):
        raise cli.external.ExternalError("no Stockfish build for mips")

    stubbed.setattr(cli.external, "fetch_stockfish", fail)

    assert run_main(config_file, "fetch", "stockfish") == 1
    assert "no Stockfish build for mips" in capsys.readouterr().err


def test_doctor_reports_a_missing_package_and_a_missing_engine_repo(tmp_path, monkeypatch, capsys):
    import importlib

    real = importlib.import_module

    def without_pyarrow(name, *a, **k):
        if name == "pyarrow":
            raise ImportError("no pyarrow")

        return real(name, *a, **k)

    monkeypatch.setattr(cli.importlib, "import_module", without_pyarrow)
    file = tmp_path / "m.toml"
    file.write_text(
        f'engine_repo = "{tmp_path / "missing"}"\ndata_dir = "{tmp_path / "d"}"\n'
        f'cache_dir = "{tmp_path / "c"}"\nworkers = 1\n'
        f'fastchess = "{tmp_path / "no-fastchess"}"\n'
        f'stockfish = "{tmp_path / "no-stockfish"}"\n'
    )

    assert main(["--config", str(file), "doctor"]) == 1
    out = capsys.readouterr().out

    assert "FAIL package pyarrow: not installed" in out
    assert "FAIL engine repository" in out and "FAIL fastchess" in out and "FAIL stockfish" in out


def test_doctor_runs_the_fastchess_it_is_given(tmp_path, tiny_repo, capsys):
    repo, _ = tiny_repo
    program = tmp_path / "fastchess"
    program.write_text("#!/bin/sh\necho fastchess test 1.0\n")
    program.chmod(0o755)
    file = tmp_path / "m.toml"
    file.write_text(
        f'engine_repo = "{repo}"\ndata_dir = "{tmp_path / "d"}"\n'
        f'cache_dir = "{tmp_path / "c"}"\nworkers = 1\nfastchess = "{program}"\n'
    )

    assert main(["--config", str(file), "doctor"]) == 0
    assert "ok   fastchess: fastchess test 1.0" in capsys.readouterr().out


def test_selftest_without_a_tests_folder_is_an_error(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "a" / "b" / "cli.py"))

    assert main(["selftest"]) == 1
    assert "no tests folder" in capsys.readouterr().err


def test_a_dirty_framework_is_reported_without_a_traceback(
    stubbed, config_file, tmp_path, capsys, monkeypatch
):
    from meristats import provenance

    repo = tmp_path / "dirty"
    repo.mkdir()
    from conftest import run_git

    run_git(repo, "init", "-q", "-b", "main")
    (repo / "f").write_text("x")
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-q", "-m", "m")
    (repo / "f").write_text("changed")
    monkeypatch.setattr(provenance, "framework_root", lambda: repo)
    (tmp_path / "s.tsv").write_text("")

    assert run_main(config_file, "run", "rules", "--engine", "x") == 1
    assert "framework clean: the working tree has changes" in capsys.readouterr().err


def test_the_dataset_command_builds_and_registers_the_sample(
    config_file, tmp_path, monkeypatch, capsys
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from meristats import datasets
    from meristats.config import load
    from meristats.store import Store

    start = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    path = tmp_path / "p.parquet"
    pq.write_table(
        pa.table(
            {
                "PuzzleId": [f"p{i}" for i in range(30)],
                "FEN": [start] * 30,
                "Moves": ["e7e5 g1f3 b8c6"] * 30,
                "Rating": [1500 + i for i in range(30)],
                "Themes": [["advantage"]] * 30,
            }
        ),
        path,
    )
    monkeypatch.setattr(
        datasets, "fetch_puzzle_files", lambda directory, hashes=None, reuse_from=(): [path]
    )

    assert run_main(config_file, "dataset", "puzzles", "--total", "20", "--take-all", "") == 0
    out = capsys.readouterr().out

    assert "20 of 30 puzzles" in out and "sha256" in out
    with Store(load(config_file).data_dir) as store:
        assert store.db.execute("SELECT COUNT(*) FROM datasets").fetchone()[0] == 1


def test_a_failing_dataset_download_is_reported(config_file, monkeypatch, capsys):
    from meristats import datasets

    def fail(directory, hashes=None, reuse_from=()):
        raise datasets.DatasetError("hash mismatch")

    monkeypatch.setattr(datasets, "fetch_puzzle_files", fail)

    assert run_main(config_file, "dataset", "puzzles") == 1
    assert "hash mismatch" in capsys.readouterr().err


def test_run_experiment_passes_the_settings_and_prints_each_decision(
    stubbed, config_file, tmp_path, capsys
):
    book = tmp_path / "book.pgn"
    book.write_text("x")
    results = [
        ("run-a", {"new": "network+a", "keep": True, "reasons": ["match 56%", "nodes 0.9"]}),
        ("run-b", {"new": "network+a+b", "keep": False, "reasons": ["match 40%"]}),
    ]
    recorder = Recorder(("network+a", results))
    stubbed.setattr(cli.experiment_protocol, "run_changes", recorder)

    assert (
        run_main(
            config_file,
            "run",
            "experiment",
            "--engine",
            "x",
            "--base",
            "network",
            "--change",
            "a",
            "--change",
            "b",
            "--items",
            "5000",
            "--nodes",
            "1000",
            "--depths",
            "4,6",
            "--st",
            "0.1",
            "--rounds",
            "50",
            "--elo1",
            "20",
            "--seed",
            "9",
            "--cpu",
            "3",
            "--book",
            str(book),
            "--allow-dirty",
        )
        == 0
    )
    ((args, kwargs),) = recorder.calls
    out = capsys.readouterr().out
    settings = args[8]

    assert args[3:5] == ("network", ["a", "b"])
    assert settings.puzzle_items == 5000 and settings.puzzle_nodes == 1000
    assert settings.depths == (4, 6) and settings.st == 0.1 and settings.rounds == 50
    assert settings.sprt == cli.match_protocol.Sprt(0.0, 20.0) and settings.seed == 9
    assert settings.cpu == 3 and kwargs["allow_dirty"] is True
    assert "network+a: kept  run-a" in out and "network+a+b: dropped  run-b" in out
    assert "  match 56%" in out and "final base: network+a" in out


def test_run_experiment_needs_a_change(stubbed, config_file):
    with pytest.raises(SystemExit):
        run_main(config_file, "run", "experiment", "--engine", "x")


def test_run_spsa_passes_the_arguments_and_prints_the_tuned_preset(
    stubbed, config_file, tmp_path, capsys
):
    book = tmp_path / "book.pgn"
    book.write_text("x")
    recorder = Recorder(("run-s", {"tuned": "network+rfp_margin=118"}))
    stubbed.setattr(cli.spsa_protocol, "run", recorder)

    assert (
        run_main(
            config_file,
            "run",
            "spsa",
            "--engine",
            "x",
            "--base",
            "network",
            "--iterations",
            "30",
            "--pairs",
            "12",
            "--st",
            "0.03",
            "--book",
            str(book),
        )
        == 0
    )
    ((args, kwargs),) = recorder.calls

    assert args[3] == "network" and args[5] == cli.match_protocol.Limit("st", 0.03)
    assert kwargs["iterations"] == 30 and kwargs["pairs"] == 12
    assert "run-s" in capsys.readouterr().out


def test_run_puzzles_and_match_pass_the_network_files(stubbed, config_file, tmp_path):
    net = tmp_path / "n.nnue"
    book = tmp_path / "book.pgn"
    book.write_text("x")
    puzzle = Recorder(
        (
            "r",
            {
                "whole": dict(p=1, low=1, high=1, first_move=1, puzzle_rating=1, mean_nodes=1),
                "errors": {"count": 0},
            },
        )
    )
    match = Recorder(("r", {"games": 2, "wins": 1, "draws": 1, "losses": 0}))
    stubbed.setattr(cli.puzzles_protocol, "run", puzzle)
    stubbed.setattr(cli.match_protocol, "run", match)

    run_main(config_file, "run", "puzzles", "--engine", "x", "--eval-file", str(net))
    run_main(
        config_file,
        "run",
        "match",
        "--a",
        "x",
        "--b",
        "y",
        "--book",
        str(book),
        "--a-eval-file",
        str(net),
        "--b-eval-file",
        str(net),
    )

    assert puzzle.calls[0][1]["eval_file"] == net
    assert match.calls[0][1]["a_eval_file"] == net and match.calls[0][1]["b_eval_file"] == net


def test_run_positions_passes_the_settings_and_prints_the_counts(
    stubbed, config_file, tmp_path, capsys
):
    summary = {
        "games": 10,
        "white_wins": 4,
        "draws": 3,
        "black_wins": 3,
        "positions": 120,
        "output": {"sha256": "ab" * 32},
    }
    recorder = Recorder(("run-p", summary))
    stubbed.setattr(cli.positions_protocol, "run", recorder)

    assert (
        run_main(
            config_file,
            "run",
            "positions",
            "--engine",
            "x",
            "--out",
            str(tmp_path / "p.txt"),
            "--games",
            "10",
            "--depth",
            "5",
            "--skip",
            "2",
            "--seed",
            "4",
            "--workers",
            "3",
            "--engine-config",
            "full",
            "--random-plies",
            "6",
        )
        == 0
    )
    ((args, kwargs),) = recorder.calls
    out = capsys.readouterr().out

    assert args[3] == tmp_path / "p.txt"
    assert kwargs["games"] == 10 and kwargs["depth"] == 5 and kwargs["skip"] == 2
    assert kwargs["seed"] == 4 and kwargs["workers"] == 3 and kwargs["random_plies"] == 6
    assert kwargs["engine_config"] == "full"
    assert "run-p" in out and "120 positions" in out and "abababababababab" in out


def test_run_train_passes_the_settings_and_prints_the_result(
    stubbed, config_file, tmp_path, capsys
):
    summary = {
        "best_held": 0.0123,
        "best_epoch": 7,
        "device": "cuda",
        "seconds": {"loading": 12.0, "training": 90.0, "per_epoch": 9.0},
        "network": {"path": "/x/net.nnue", "sha256": "cd" * 32},
    }
    recorder = Recorder(("run-t", summary))
    stubbed.setattr(cli.train_protocol, "run", recorder)

    assert (
        run_main(
            config_file,
            "run",
            "train",
            "--engine",
            "x",
            "--data",
            str(tmp_path / "d.txt"),
            "--epochs",
            "5",
            "--batch",
            "512",
            "--lr",
            "0.01",
            "--lambda",
            "0.5",
            "--max-score",
            "1000",
            "--limit",
            "100",
            "--seed",
            "3",
            "--device",
            "cpu",
            "--init",
            str(tmp_path / "i.nnue"),
            "--source",
            str(tmp_path / "n.inc"),
        )
        == 0
    )
    ((args, kwargs),) = recorder.calls
    out = capsys.readouterr().out

    assert args[3] == tmp_path / "d.txt"
    assert kwargs["epochs"] == 5 and kwargs["batch"] == 512 and kwargs["lr"] == 0.01
    assert kwargs["lam"] == 0.5 and kwargs["max_score"] == 1000 and kwargs["limit"] == 100
    assert kwargs["seed"] == 3 and kwargs["device"] == "cpu"
    assert kwargs["init"] == tmp_path / "i.nnue" and kwargs["source"] == tmp_path / "n.inc"
    assert "run-t" in out and "0.01230 at epoch 7 on cuda" in out and "9.0s for each epoch" in out


def test_the_magics_command_writes_the_source(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.magics, "find_all", lambda: ([1] * 64, [2] * 64))

    assert cli.main(["magics", "--out", str(tmp_path / "m.inc")]) == 0
    assert "kRookMagics" in (tmp_path / "m.inc").read_text()
    assert cli.main(["magics"]) == 0
    assert "kBishopMagics" in capsys.readouterr().out


def test_a_network_is_written_as_source(config_file, tmp_path):
    import numpy as np

    from meristats import nnue

    rng = np.random.default_rng(2)
    network = nnue.Network(
        rng.integers(-9, 9, (nnue.INPUTS, nnue.HIDDEN), dtype=np.int16),
        rng.integers(-9, 9, nnue.HIDDEN, dtype=np.int16),
        rng.integers(-9, 9, (nnue.BUCKETS, 2, nnue.HIDDEN), dtype=np.int16),
        rng.integers(-9, 9, nnue.BUCKETS, dtype=np.int32),
    )
    file, source = tmp_path / "new.nnue", tmp_path / "nnue_net.inc"
    nnue.write_network(network, file)

    assert run_main(config_file, "network", "source", str(file), "--out", str(source)) == 0
    assert np.array_equal(nnue.read_source(source).feature_weights, network.feature_weights)
