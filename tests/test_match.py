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

import dataclasses
import json
import os
from pathlib import Path

import pytest

from meristats import stats
from meristats.build import BuiltEngine
from meristats.preflight import PreflightError
from meristats.protocols import match as protocol
from meristats.protocols.match import Game, Limit, MatchError, Player, Sprt

PGN = """[Event "Fastchess Tournament"]
[Round "1"]
[White "A"]
[Black "B"]
[Result "1-0"]
[PlyCount "29"]
[Termination "normal"]
[Opening "Sicilian Defense"]

1. e4 {book} c5 {book} 2. Nf3 d6 1-0

[Event "Fastchess Tournament"]
[Round "1"]
[White "B"]
[Black "A"]
[Result "1/2-1/2"]
[PlyCount "80"]
[Termination "normal"]
[Opening "Sicilian Defense"]

1. e4 c5 1/2-1/2

[Event "Fastchess Tournament"]
[Round "2"]
[White "A"]
[Black "B"]
[Result "0-1"]
[PlyCount "40"]
[Termination "time forfeit"]
[Opening "Ruy Lopez"]

1. e4 e5 0-1

[Event "Fastchess Tournament"]
[Round "2"]
[White "B"]
[Black "A"]
[Result "0-1"]
[PlyCount "41"]
[Termination "normal"]
[Opening "Ruy Lopez"]

1. e4 e5 0-1

[Event "Fastchess Tournament"]
[Round "3"]
[White "A"]
[Black "B"]
[Result "*"]
[PlyCount "2"]
[Termination "unterminated"]

1. e4 *
"""


def game(number, round_, white, black, result, termination="normal"):
    return Game(number, round_, "o", white, black, result, termination, 10)


def test_limit_arguments_and_validation():
    assert Limit("st", 0.05).argument() == "st=0.05"
    assert Limit("st", 1).argument() == "st=1"
    assert Limit("depth", 4).argument() == "depth=4"
    assert Limit("nodes", 20000.0).argument() == "nodes=20000"

    for bad in (("tc", 1), ("st", 0), ("depth", -2)):
        with pytest.raises(ValueError):
            Limit(*bad)


def test_parse_games_reads_headers():
    games = protocol.parse_games(PGN)

    assert len(games) == 5
    assert (games[0].round, games[0].white, games[0].black, games[0].result) == (1, "A", "B", "1-0")
    assert games[0].opening == "Sicilian Defense" and games[0].plies == 29
    assert games[2].termination == "time forfeit" and games[4].result == "*"
    assert [g.number for g in games] == [1, 2, 3, 4, 5]
    assert protocol.parse_games("") == []


def test_points_of_counts_half_points_from_the_side_of_the_player():
    assert protocol.points_of(game(1, 1, "A", "B", "1-0"), "A") == 2
    assert protocol.points_of(game(1, 1, "A", "B", "1-0"), "B") == 0
    assert protocol.points_of(game(1, 1, "B", "A", "1-0"), "A") == 0
    assert protocol.points_of(game(1, 1, "B", "A", "0-1"), "A") == 2
    assert protocol.points_of(game(1, 1, "A", "B", "1/2-1/2"), "B") == 1
    assert protocol.points_of(game(1, 1, "A", "B", "*"), "A") is None
    assert protocol.points_of(game(1, 1, "X", "Y", "1-0"), "A") is None


def test_pair_results_groups_by_round_and_drops_unfinished_rounds():
    games = protocol.parse_games(PGN)

    # round 1: A wins as white, draws as black = 2 + 1; round 2: A loses as white, wins as black
    assert protocol.pair_results(games) == [(2, 1), (0, 2)]
    assert stats.pentanomial(protocol.pair_results(games)) == [0, 0, 1, 1, 0]


def test_fastchess_verdict_parsing():
    assert (
        protocol.fastchess_verdict("x\nSPRT ([0.00, 10.00]) completed - H1 was accepted\n") == "H1"
    )
    assert protocol.fastchess_verdict("SPRT ([-2.50, 0.50]) completed - H0 was accepted") == "H0"
    assert protocol.fastchess_verdict("Finished match") is None


def test_build_command_has_the_options_of_the_protocol():
    a = Player("A", Path("/bin/a"), {"Threads": 1, "Config": "network+see=0"}, 1)
    b = Player("B", Path("/bin/b"), {"Hash": 16}, 2)
    command = protocol.build_command(
        Path("/bin/fastchess"),
        a,
        b,
        Limit("st", 0.05),
        Path("/books/b.pgn"),
        rounds=10,
        concurrency=4,
        seed=7,
        pgn=Path("/w/g.pgn"),
        log=Path("/w/l.log"),
        sprt=Sprt(),
    )
    text = " ".join(command)

    assert command[0] == "/bin/fastchess"
    assert "-engine cmd=/bin/a name=A option.Threads=1 option.Config=network+see=0" in text
    assert "-engine cmd=/bin/b name=B option.Hash=16" in text
    assert "-each st=0.05" in text and "-games 2 -repeat -rounds 10" in text
    assert "-concurrency 4 -recover" in text and "-srand 7" in text
    assert "-openings file=/books/b.pgn format=pgn order=random" in text
    assert "-pgnout file=/w/g.pgn nodes=true timeleft=true" in text
    assert "-sprt elo0=0 elo1=10 alpha=0.05 beta=0.05 model=logistic" in text
    without = protocol.build_command(
        Path("f"),
        a,
        b,
        Limit("depth", 4),
        Path("b"),
        rounds=1,
        concurrency=1,
        seed=1,
        pgn=Path("g"),
        log=Path("l"),
        sprt=None,
    )
    assert "-sprt" not in without and "-each depth=4" in " ".join(without)


def test_summarise_for_a_match_without_a_test():
    summary = protocol.summarise(protocol.parse_games(PGN), None, "")

    assert (summary["games"], summary["finished"]) == (5, 4)
    assert (summary["wins"], summary["draws"], summary["losses"]) == (2, 1, 1)
    assert summary["pentanomial"] == [0, 0, 1, 1, 0] and summary["pairs"] == 2
    assert summary["score"] == pytest.approx(0.625) and "sprt" not in summary
    assert summary["terminations"] == {"normal": 3, "time forfeit": 1, "unterminated": 1}
    assert summary["score_low"] < summary["score"] < summary["score_high"]


def test_summarise_with_a_test_stores_both_verdicts():
    games = []

    for number in range(60):
        games += [
            game(2 * number + 1, number + 1, "A", "B", "1-0"),
            game(2 * number + 2, number + 1, "B", "A", "0-1"),
        ]

    agree = protocol.summarise(games, Sprt(), "SPRT ([0.00, 10.00]) completed - H1 was accepted")
    assert agree["sprt"]["verdict"] == "H1" and agree["sprt"]["fastchess_verdict"] == "H1"
    assert agree["sprt"]["agree"] and agree["sprt"]["llr"] > agree["sprt"]["bounds"][1]

    differ = protocol.summarise(games, Sprt(), "Finished match")
    assert differ["sprt"]["fastchess_verdict"] == "continue" and not differ["sprt"]["agree"]


def test_games_table_round_trips():
    import io

    import pyarrow.parquet as pq

    games = protocol.parse_games(PGN)
    table = pq.read_table(io.BytesIO(protocol.games_table(games))).to_pydict()

    assert table["result"] == ["1-0", "1/2-1/2", "0-1", "0-1", "*"]
    assert table["round"] == [1, 1, 2, 2, 3] and table["plies"][0] == 29


def test_compliance_uses_the_fastchess_program(fake_fastchess, monkeypatch):
    program = fake_fastchess(["1-0"])

    assert protocol.compliance(program, Path("/bin/engine")).ok

    monkeypatch.setenv("FAKE_COMPLIANCE", "bad")
    assert not protocol.compliance(program, Path("/bin/engine")).ok
    assert not protocol.compliance(Path("/nonexistent/fastchess"), Path("/bin/engine")).ok


def engines(store, oracle):
    program = oracle({})
    a = store.add_engine("meridian", "oracle", "a" * 40, "1" * 64, {}, 7, "Oracle 1")
    b = store.add_engine("meridian", "oracle", "b" * 40, "2" * 64, {}, 7, "Oracle 1")
    return (
        BuiltEngine(a, "a" * 40, program, "1" * 64, 7, 1000, "Oracle 1", {}),
        BuiltEngine(b, "b" * 40, program, "2" * 64, 7, 1000, "Oracle 1", {}),
    )


@pytest.fixture
def book(tmp_path):
    path = tmp_path / "book.pgn"
    path.write_text('[Event "?"]\n\n1. e4 e5 *\n')
    return path


def match_config(config, fastchess):
    return dataclasses.replace(config, fastchess=fastchess)


def test_a_match_is_stored_with_games_logs_and_both_verdicts(
    config, store, oracle, book, fake_fastchess, framework_repo
):
    results = ["1-0", "0-1"] * 40
    fastchess = fake_fastchess(results, verdict="H1")
    a, b = engines(store, oracle)
    run_id, summary = protocol.run(
        match_config(config, fastchess),
        store,
        a,
        b,
        book,
        Limit("st", 0.05),
        a_config="network",
        b_config="baseline",
        rounds=40,
        seed=3,
        framework=framework_repo,
        progress=lambda t: None,
    )
    row = store.get_run(run_id)
    roles = {
        r["role"]
        for r in store.db.execute("SELECT role FROM run_blobs WHERE run_id = ?", (run_id,))
    }
    args = Path(os.environ["FAKE_ARGS_FILE"]).read_text()

    assert (summary["wins"], summary["losses"]) == (80, 0)
    assert summary["sprt"]["verdict"] == "H1" and summary["sprt"]["agree"]
    assert row["protocol"] == "match" and row["grade"] == "verified"
    assert row["config"] == "network vs baseline" and row["seed"] == 3 and row["notes"] is None
    assert roles == {"pgn", "tool-log", "items"}
    assert store.get_blob(
        store.db.execute("SELECT sha256 FROM run_blobs WHERE role = 'pgn'").fetchone()["sha256"]
    ).startswith(b"[Event")
    assert "option.Config=network" in args and "option.Config=baseline" in args
    assert "option.MoveOverhead=0" in args and "-each st=0.05" in args
    assert not Path("config.json").exists() and any(config.cache_dir.rglob("config.json"))
    limits = json.loads(row["limits_json"])
    assert limits["sprt"]["elo1"] == 10 and limits["rounds"] == 40


def test_a_disagreement_between_the_verdicts_is_flagged(
    config, store, oracle, book, fake_fastchess, framework_repo
):
    fastchess = fake_fastchess(["1-0", "0-1"] * 60, verdict=None)
    a, b = engines(store, oracle)
    run_id, summary = protocol.run(
        match_config(config, fastchess),
        store,
        a,
        b,
        book,
        Limit("st", 0.05),
        rounds=60,
        framework=framework_repo,
        progress=lambda t: None,
    )

    assert not summary["sprt"]["agree"]
    assert store.get_run(run_id)["notes"] == "sprt_disagree"


def test_a_depth_limit_sends_no_move_overhead(
    config, store, oracle, book, fake_fastchess, framework_repo
):
    fastchess = fake_fastchess(["1-0", "0-1"])
    a, b = engines(store, oracle)
    protocol.run(
        match_config(config, fastchess),
        store,
        a,
        b,
        book,
        Limit("depth", 4),
        sprt=None,
        rounds=1,
        framework=framework_repo,
        progress=lambda t: None,
    )

    args = Path(os.environ["FAKE_ARGS_FILE"]).read_text()
    assert "MoveOverhead" not in args and "-sprt" not in args and "-each depth=4" in args


def test_an_external_player_gets_its_own_options(
    config, store, oracle, book, fake_fastchess, framework_repo
):
    fastchess = fake_fastchess(["1-0", "0-1"])
    a, _ = engines(store, oracle)
    external = store.add_engine("external", "stockfish", None, "3" * 64, {}, None, "SF 19")
    player = Player(
        "B", Path("/bin/stockfish"), {"UCI_LimitStrength": "true", "UCI_Elo": 2000}, external
    )
    run_id, _ = protocol.run(
        match_config(config, fastchess),
        store,
        a,
        None,
        book,
        Limit("st", 0.05),
        b_player=player,
        sprt=None,
        rounds=1,
        framework=framework_repo,
        check_compliance=False,
        progress=lambda t: None,
    )
    args = Path(os.environ["FAKE_ARGS_FILE"]).read_text()

    assert (
        "-engine cmd=/bin/stockfish name=B option.UCI_LimitStrength=true option.UCI_Elo=2000"
        in args
    )
    assert json.loads(store.get_run(run_id)["limits_json"])["b_engine_id"] == external


def test_a_failing_compliance_check_stops_the_run(
    config, store, oracle, book, fake_fastchess, framework_repo
):
    fastchess = fake_fastchess(["1-0", "0-1"], FAKE_COMPLIANCE="bad")
    a, b = engines(store, oracle)

    with pytest.raises(PreflightError, match="compliance"):
        protocol.run(
            match_config(config, fastchess),
            store,
            a,
            b,
            book,
            Limit("st", 0.05),
            rounds=1,
            framework=framework_repo,
            progress=lambda t: None,
        )

    assert store.runs("match") == []


def test_a_failing_fastchess_is_an_error_and_stores_nothing(
    config, store, oracle, book, fake_fastchess, framework_repo
):
    fastchess = fake_fastchess(["1-0", "0-1"], FAKE_EXIT=2)
    a, b = engines(store, oracle)

    with pytest.raises(MatchError, match="fastchess failed"):
        protocol.run(
            match_config(config, fastchess),
            store,
            a,
            b,
            book,
            Limit("st", 0.05),
            rounds=1,
            framework=framework_repo,
            check_compliance=False,
            progress=lambda t: None,
        )

    assert store.runs("match") == []


def test_no_fastchess_in_the_settings_is_an_error(config, store, oracle, book, framework_repo):
    a, b = engines(store, oracle)

    with pytest.raises(MatchError, match="no fastchess"):
        protocol.run(
            config,
            store,
            a,
            b,
            book,
            Limit("st", 0.05),
            rounds=1,
            framework=framework_repo,
            check_compliance=False,
        )


def test_a_ladder_fits_a_rating_from_all_levels(
    config, store, oracle, book, fake_fastchess, framework_repo
):
    from meristats.protocols import ladder

    fastchess = fake_fastchess(
        (["1-0", "0-1"] * 14) + (["1/2-1/2", "1/2-1/2"] * 3) + (["0-1", "1-0"] * 3)
    )
    a, _ = engines(store, oracle)
    external_id = store.add_engine("external", "stockfish", None, "3" * 64, {}, None, "SF 19")
    run_id, summary = ladder.run(
        match_config(config, fastchess),
        store,
        a,
        Path("/bin/stockfish"),
        external_id,
        book,
        Limit("st", 0.05),
        [1800, 2200, 2600],
        rounds=20,
        framework=framework_repo,
        check_compliance=False,
        progress=lambda t: None,
    )
    row = store.get_run(run_id)

    assert set(summary["levels"]) == {"1800", "2200", "2600"}
    assert summary["games"] == 3 * 40 and row["protocol"] == "ladder"
    rating = summary["rating"]
    assert rating["low"] < rating["value"] < rating["high"]
    assert 2100 < rating["value"] < 2900
    roles = [
        r["role"]
        for r in store.db.execute("SELECT role FROM run_blobs WHERE run_id = ?", (run_id,))
    ]
    # the fake gives identical games at every level, so the blobs are shared
    assert set(roles) == {"pgn", "tool-log"}
    assert row["items_blob"] is None


def test_a_ladder_needs_levels(config, store, oracle, book, framework_repo):
    from meristats.protocols import ladder

    a, _ = engines(store, oracle)

    with pytest.raises(ValueError):
        ladder.run(
            config,
            store,
            a,
            Path("/bin/sf"),
            1,
            book,
            Limit("st", 0.05),
            [],
            framework=framework_repo,
        )


def test_a_ladder_with_a_depth_limit_and_a_config_sends_no_overhead(
    config, store, oracle, book, fake_fastchess, framework_repo
):
    from meristats.protocols import ladder

    fastchess = fake_fastchess(["1-0", "0-1"])
    a, _ = engines(store, oracle)
    external_id = store.add_engine("external", "stockfish", None, "3" * 64, {}, None, "SF 19")
    run_id, _ = ladder.run(
        match_config(config, fastchess),
        store,
        a,
        Path("/bin/stockfish"),
        external_id,
        book,
        Limit("depth", 4),
        [2000],
        a_config="network+see=0",
        rounds=1,
        framework=framework_repo,
        check_compliance=False,
        progress=lambda t: None,
    )
    args = Path(os.environ["FAKE_ARGS_FILE"]).read_text()

    assert "option.Config=network+see=0" in args and "MoveOverhead" not in args
    assert store.get_run(run_id)["config"] == "network+see=0"


def test_each_side_can_play_with_its_own_network_file(
    config, store, oracle, book, fake_fastchess, framework_repo, tmp_path
):
    import hashlib
    import json

    network = tmp_path / "net.nnue"
    network.write_bytes(b"network")
    fastchess = fake_fastchess(["1-0", "0-1"] * 4)
    a, b = engines(store, oracle)
    run_id, _ = protocol.run(
        match_config(config, fastchess),
        store,
        a,
        b,
        book,
        Limit("st", 0.05),
        a_eval_file=network,
        rounds=4,
        sprt=None,
        framework=framework_repo,
        progress=lambda t: None,
    )
    args = Path(os.environ["FAKE_ARGS_FILE"]).read_text()
    limits = json.loads(store.get_run(run_id)["limits_json"])

    assert args.count("option.EvalFile=") == 1
    assert f"option.EvalFile={network.resolve()}" in args
    assert limits["eval_files"] == {"a": hashlib.sha256(b"network").hexdigest()}
    assert "EvalFile" in limits["options_a"] and "EvalFile" not in limits["options_b"]
