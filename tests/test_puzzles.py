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

import chess
import pytest

from meristats import datasets
from meristats.build import BuiltEngine
from meristats.datasets import Puzzle
from meristats.protocols import puzzles as protocol
from meristats.protocols.puzzles import Item, Limit
from meristats.uci import UciEngine

START = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
AFTER_E5 = "e7e5 g1f3"


def position(fen, played=()):
    return f"position fen {fen}" + (" moves " + " ".join(played) if played else "")


def two_mate_position():
    """
    A king and queen against a king with two different mates in one, and the two moves.
    """
    for wk in range(64):
        for wq in range(64):
            board = chess.Board(None)
            board.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
            board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(wq, chess.Piece(chess.QUEEN, chess.WHITE))

            if wk == wq or not board.is_valid() or board.is_check():
                continue

            mates = []

            for move in board.legal_moves:
                after = board.copy()
                after.push(move)

                if after.is_checkmate():
                    mates.append(move.uci())

            if len(mates) >= 2:
                return board.fen(), mates

    raise AssertionError("no position with two mates")


def run_solve(oracle, answers, puzzle, limit=Limit("nodes", 1000)):
    program = oracle(answers)

    with UciEngine(program) as uci:
        return protocol.solve(uci, puzzle, limit, 7)


def test_limit_commands_and_validation():
    assert Limit("nodes", 20000).command() == "nodes 20000"
    assert Limit("movetime", 50).timed and not Limit("depth", 4).timed

    for bad in (("time", 5), ("nodes", 0), ("depth", -1)):
        with pytest.raises(ValueError):
            Limit(*bad)


def test_depth_kept_counts_the_last_iterations_that_hold_the_move():
    from meristats.uci import Info

    def infos(*pairs):
        return [Info(depth=d, pv=[m]) for d, m in pairs]

    assert protocol.depth_kept(infos((1, "a"), (2, "e7e5"), (3, "e7e5")), "e7e5") == 2
    assert protocol.depth_kept(infos((1, "e7e5"), (2, "a"), (3, "e7e5")), "e7e5") == 3
    assert protocol.depth_kept(infos((1, "e7e5"), (2, "e7e5"), (3, "b")), "e7e5") == 0
    assert protocol.depth_kept([], "e7e5") == 0
    assert protocol.depth_kept([Info(depth=4, pv=[])], "e7e5") == 0


def test_gives_checkmate():
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")

    assert protocol.gives_checkmate(board, "a1a8")
    assert not protocol.gives_checkmate(board, "a1a2")
    assert not protocol.gives_checkmate(board, "zzzz") and not protocol.gives_checkmate(board, "")
    assert board.fen().startswith("6k1/5ppp")


def test_a_puzzle_with_every_move_right(oracle):
    puzzle = Puzzle("p", 1500, 0, 1.0, ("t",), START, ("e7e5", "g1f3", "b8c6"))
    answers = {
        position(START): {
            "best": "e7e5",
            "infos": [[1, 10, "a2a3"], [2, 30, "e7e5"], [3, 90, "e7e5"]],
        },
        position(START, ["e7e5", "g1f3"]): {"best": "b8c6", "infos": [[1, 5, "b8c6"]]},
    }
    result = run_solve(oracle, answers, puzzle)

    assert (result.ok_mask, result.first_fail, result.solved) == (0b11, 0, True)
    assert result.nodes == 90 + 5 and result.depth_kept == (2, 1) and result.error == ""
    assert result.item == 7 and result.solver_moves == 2 and result.wall_ms > 0


def test_the_line_stops_at_the_first_wrong_move(oracle):
    puzzle = Puzzle("p", 1500, 0, 1.0, (), START, ("e7e5", "g1f3", "b8c6", "f1c4", "g8f6"))
    answers = {
        position(START): {"best": "e7e5"},
        position(START, ["e7e5", "g1f3"]): {"best": "d7d6"},
    }
    result = run_solve(oracle, answers, puzzle)

    assert (result.ok_mask, result.first_fail, result.solved) == (0b001, 2, False)
    assert result.solver_moves == 3


def test_a_wrong_first_move_sends_one_search_only(oracle):
    puzzle = Puzzle("p", 1500, 0, 1.0, (), START, ("e7e5", "g1f3", "b8c6"))
    result = run_solve(oracle, {position(START): {"best": "a7a6"}}, puzzle)

    assert (result.ok_mask, result.first_fail) == (0, 1)


def test_another_checkmate_is_right_and_another_move_is_not(oracle):
    fen, mates = two_mate_position()
    puzzle = Puzzle("m", 1500, 1, 1.0, ("mateIn1",), fen, (mates[0],))

    assert run_solve(oracle, {position(fen): {"best": mates[0]}}, puzzle).solved
    assert run_solve(oracle, {position(fen): {"best": mates[1]}}, puzzle).solved

    board = chess.Board(fen)
    other = next(m.uci() for m in board.legal_moves if m.uci() not in mates)
    assert not run_solve(oracle, {position(fen): {"best": other}}, puzzle).solved


def test_the_same_rule_holds_for_a_puzzle_that_is_not_labelled_a_mate(oracle):
    fen, mates = two_mate_position()
    puzzle = Puzzle("m", 1500, 0, 1.0, ("advantage",), fen, (mates[0],))

    assert run_solve(oracle, {position(fen): {"best": mates[1]}}, puzzle).solved


def test_an_illegal_move_is_wrong_and_not_an_error(oracle):
    puzzle = Puzzle("p", 1500, 0, 1.0, (), START, ("e7e5",))
    result = run_solve(oracle, {position(START): {"best": "e2e5"}}, puzzle)

    assert (result.ok_mask, result.first_fail, result.error) == (0, 1, "")


def test_a_bad_fen_is_an_error_item(oracle):
    puzzle = Puzzle("p", 1500, 0, 1.0, (), "not a fen", ("e7e5",))
    result = run_solve(oracle, {}, puzzle)

    assert result.error.startswith("bad FEN") and result.first_fail == 1 and not result.solved


def test_an_illegal_move_in_the_stored_line_is_an_error(oracle):
    puzzle = Puzzle("p", 1500, 0, 1.0, (), START, ("e7e5", "e2e5", "b8c6"))
    answers = {position(START): {"best": "e7e5"}}
    result = run_solve(oracle, answers, puzzle)

    assert "illegal line move e2e5" in result.error and result.ok_mask == 1


def test_item_outcome_and_solved():
    item = Item(0, 2, 0b11, 0, 5, 1.0, (2, 3))

    assert item.solved and item.outcome() == (0b11, 0, 5, (2, 3))
    assert not Item(0, 2, 0b01, 2, 5, 1.0, (2,)).solved
    assert not Item(0, 0, 0, 0, 0, 0.0, ()).solved


def test_items_round_trip_through_parquet():
    items = [
        Item(0, 2, 0b11, 0, 5, 1.5, (2, 3)),
        Item(1, 3, 0b001, 2, 99, 0.25, (1,), "boom"),
        Item(2, 1, 0, 1, 0, 0.0, ()),
    ]
    data = protocol.items_table(items)
    back = protocol.read_items(data)

    assert [
        (i.item, i.solver_moves, i.ok_mask, i.first_fail, i.nodes, i.depth_kept, i.error)
        for i in back
    ] == [
        (i.item, i.solver_moves, i.ok_mask, i.first_fail, i.nodes, i.depth_kept, i.error)
        for i in items
    ]
    assert back[0].wall_ms == pytest.approx(1.5)


def make_puzzles(count):
    return [
        Puzzle(
            f"p{i}",
            1000 + 100 * (i % 5),
            i % 3,
            1.0 + (i % 2),
            ("a", "b") if i % 2 else ("a",),
            START[:-1] + str(i + 1),
            ("e7e5", "g1f3", "b8c6"),
        )
        for i in range(count)
    ]


def test_summarise_groups_and_weights():
    puzzles = make_puzzles(10)
    results = [
        Item(i, 2, 0b11 if i < 6 else 0b01, 0 if i < 6 else 2, 100, 1.0, (2, 2)) for i in range(10)
    ]
    summary = protocol.summarise(puzzles, results)
    weights = [p.weight for p in puzzles]
    expected = sum(w for i, w in enumerate(weights) if i < 6) / sum(weights)

    assert summary["whole"]["p"] == pytest.approx(expected)
    assert summary["whole"]["first_move"] == 1.0 and summary["whole"]["mean_nodes"] == 100
    assert summary["whole"]["mean_depth_kept"] == pytest.approx(2.0)
    assert summary["whole"]["puzzle_rating"] > 0
    assert set(summary["by_rating_band"]) == {"1000-1199", "1200-1399", "1400-1599"}
    assert set(summary["by_solver_moves"]) == {"2"}
    assert set(summary["by_kind"]) == {"no mate", "mate in 1", "mate in 2"}
    assert summary["by_theme"] == {}
    assert summary["errors"] == {"count": 0, "first": []}


def test_summarise_reports_errors_and_big_themes():
    puzzles = [
        Puzzle(f"p{i}", 1500, 0, 1.0, ("big",), START, ("e7e5",))
        for i in range(protocol.THEME_MINIMUM + 5)
    ]
    results = [Item(i, 1, 1, 0, 1, 1.0, (1,)) for i in range(len(puzzles) - 1)]
    results.append(Item(len(puzzles) - 1, 1, 0, 1, 0, 0.0, (), "boom"))
    summary = protocol.summarise(puzzles, results)

    assert summary["errors"] == {"count": 1, "first": [f"{len(puzzles) - 1}: boom"]}
    assert "big" in summary["by_theme"]


def write_sample(path, puzzles):
    with open(path, "w") as file:
        for p in puzzles:
            file.write(
                "\t".join(
                    [
                        p.id,
                        str(p.rating),
                        str(p.mate),
                        f"{p.weight:.6f}",
                        ",".join(p.themes),
                        p.fen,
                        " ".join(p.moves),
                    ]
                )
                + "\n"
            )


def fake_engine_row(store, binary):
    sha = "c" * 40
    engine_id = store.add_engine("meridian", "oracle", sha, "d" * 64, {}, 7, "Oracle 1")
    return BuiltEngine(engine_id, sha, binary, "d" * 64, 7, "Oracle 1", {})


def answers_for(puzzles, wrong=()):
    answers = {}

    for p in puzzles:
        answers[position(p.fen)] = {
            "best": "a7a6" if p.id in wrong else "e7e5",
            "infos": [[1, 10, "e7e5"], [2, 40, "e7e5"]],
        }
        answers[position(p.fen, ["e7e5", "g1f3"])] = {"best": "b8c6", "infos": [[1, 8, "b8c6"]]}

    return answers


def test_a_full_run_is_stored_with_everything(config, store, oracle, tmp_path, framework_repo):
    puzzles = make_puzzles(30)
    sample = tmp_path / "sample.tsv"
    write_sample(sample, puzzles)
    program = oracle(answers_for(puzzles, wrong={"p3", "p4"}))
    engine = fake_engine_row(store, program)
    run_id, summary = protocol.run(
        config,
        store,
        engine,
        sample,
        Limit("nodes", 100),
        engine_config="network+see=0",
        workers=3,
        framework=framework_repo,
        progress=lambda t: None,
    )
    row = store.get_run(run_id)
    items = protocol.read_items(store.get_blob(row["items_blob"]))

    assert row["protocol"] == "puzzles" and row["protocol_version"] == 2
    assert row["grade"] == "verified" and row["status"] == "complete"
    assert row["config"] == "network+see=0"
    assert [i.item for i in items] == list(range(30))
    assert sum(1 for i in items if i.solved) == 28
    assert summary["items"] == 30 and 0.9 < summary["whole"]["p"] < 0.95
    assert "Threads" in row["limits_json"] and "MoveOverhead" not in row["limits_json"]
    assert row["dataset_id"] is not None
    assert (
        store.db.execute("SELECT role FROM run_blobs WHERE run_id = ?", (run_id,)).fetchone()[
            "role"
        ]
        == "items"
    )
    assert (
        datasets.file_sha256(sample)
        == store.db.execute("SELECT sha256 FROM datasets").fetchone()["sha256"]
    )


def test_the_checks_are_recorded_in_the_conditions(config, store, oracle, tmp_path, framework_repo):
    puzzles = make_puzzles(10)
    sample = tmp_path / "sample.tsv"
    write_sample(sample, puzzles)
    engine = fake_engine_row(store, oracle(answers_for(puzzles)))
    run_id, _ = protocol.run(
        config,
        store,
        engine,
        sample,
        Limit("depth", 3),
        workers=2,
        framework=framework_repo,
        progress=lambda t: None,
    )

    import json

    conditions = json.loads(store.get_run(run_id)["conditions_json"])
    names = {check["name"] for check in conditions["checks"]}

    assert {
        "framework clean",
        "engine from a SHA",
        "fingerprint",
        "one thread",
        "dataset hash",
        "determinism sample",
        "engine options",
    } <= names
    assert all(check["ok"] for check in conditions["checks"])
    assert [s["at"] for s in conditions["samples"]] == ["start", "before", "end"]


def test_a_timed_run_sends_move_overhead_zero_and_skips_determinism(
    config, store, oracle, tmp_path, framework_repo
):
    puzzles = make_puzzles(6)
    sample = tmp_path / "sample.tsv"
    write_sample(sample, puzzles)
    engine = fake_engine_row(store, oracle(answers_for(puzzles)))
    run_id, _ = protocol.run(
        config,
        store,
        engine,
        sample,
        Limit("movetime", 50),
        workers=2,
        framework=framework_repo,
        progress=lambda t: None,
    )
    row = store.get_run(run_id)

    import json

    assert json.loads(row["limits_json"])["options"]["MoveOverhead"] == 0
    assert "determinism sample" not in row["conditions_json"]


def test_a_crashing_engine_is_counted_and_the_run_goes_on(
    config, store, oracle, tmp_path, framework_repo
):
    puzzles = make_puzzles(8)
    answers = answers_for(puzzles)
    answers[position(puzzles[2].fen)] = "crash"
    sample = tmp_path / "sample.tsv"
    write_sample(sample, puzzles)
    engine = fake_engine_row(store, oracle(answers))

    from meristats.preflight import PreflightError

    with pytest.raises(PreflightError, match="determinism sample"):
        protocol.run(
            config,
            store,
            engine,
            sample,
            Limit("nodes", 10),
            workers=2,
            framework=framework_repo,
            progress=lambda t: None,
        )

    run_id, summary = protocol.run(
        config,
        store,
        engine,
        sample,
        Limit("movetime", 10),
        workers=2,
        framework=framework_repo,
        progress=lambda t: None,
    )
    assert summary["errors"]["count"] == 1, summary["errors"]
    assert "EngineCrashed" in summary["errors"]["first"][0]
    assert summary["items"] == 8
    items = protocol.read_items(store.get_blob(store.get_run(run_id)["items_blob"]))
    assert items[2].error and items[1].solved and items[3].solved


def test_an_empty_sample_is_refused(config, store, oracle, tmp_path, framework_repo):
    sample = tmp_path / "sample.tsv"
    sample.write_text("")
    engine = fake_engine_row(store, oracle({}))

    with pytest.raises(datasets.DatasetError, match="no puzzles"):
        protocol.run(config, store, engine, sample, Limit("nodes", 10), framework=framework_repo)


def test_a_config_for_an_engine_without_the_option_is_refused(
    config, store, tmp_path, fake_engine, framework_repo
):
    puzzles = make_puzzles(3)
    sample = tmp_path / "sample.tsv"
    write_sample(sample, puzzles)
    path, _ = fake_engine()
    # the fake engine has a Config option, so use a program without one
    bare = tmp_path / "bare"
    bare.write_text(
        "#!/bin/sh\nwhile read l; do case \"$l\" in uci) echo 'id name Bare'; "
        "echo uciok;; isready) echo readyok;; quit) exit 0;; esac; done\n"
    )
    bare.chmod(0o755)
    engine = fake_engine_row(store, bare)

    with pytest.raises(Exception, match="no Config option"):
        protocol.run(
            config,
            store,
            engine,
            sample,
            Limit("nodes", 10),
            engine_config="x",
            framework=framework_repo,
            progress=lambda t: None,
        )


def test_a_dirty_framework_is_refused_and_allowed_with_the_unverified_grade(
    config, store, oracle, tmp_path, framework_repo
):
    puzzles = make_puzzles(4)
    sample = tmp_path / "sample.tsv"
    write_sample(sample, puzzles)
    engine = fake_engine_row(store, oracle(answers_for(puzzles)))
    (framework_repo / "file").write_text("changed")

    from meristats.preflight import PreflightError

    with pytest.raises(PreflightError, match="framework clean"):
        protocol.run(config, store, engine, sample, Limit("depth", 2), framework=framework_repo)

    run_id, _ = protocol.run(
        config,
        store,
        engine,
        sample,
        Limit("depth", 2),
        workers=1,
        allow_dirty=True,
        framework=framework_repo,
        progress=lambda t: None,
    )
    row = store.get_run(run_id)
    assert row["grade"] == "unverified" and row["code_dirty"] == 1


def test_groups_are_in_numeric_order():
    assert sorted(["10", "2", "1", "11"], key=protocol.natural_key) == ["1", "2", "10", "11"]
    assert sorted(["1000-1199", "200-399", "2000-2199", "400-599"], key=protocol.natural_key) == [
        "200-399",
        "400-599",
        "1000-1199",
        "2000-2199",
    ]
    assert sorted(["no mate", "mate in 10", "mate in 2"], key=protocol.natural_key) == [
        "mate in 2",
        "mate in 10",
        "no mate",
    ]

    puzzles = [Puzzle(f"p{i}", 100 + 300 * i, 0, 1.0, (), START, ("e7e5",)) for i in range(8)]
    results = [Item(i, i % 11 + 1, 1, 0, 1, 1.0, (1,)) for i in range(8)]
    summary = protocol.summarise(puzzles, results)

    assert list(summary["by_rating_band"]) == sorted(
        summary["by_rating_band"], key=protocol.natural_key
    )
    assert list(summary["by_rating_band"]) == [
        "0-199",
        "400-599",
        "600-799",
        "1000-1199",
        "1200-1399",
        "1600-1799",
        "1800-1999",
        "2200-2399",
    ]


def test_a_network_file_is_sent_as_the_option_and_recorded_by_its_hash(
    config, store, oracle, tmp_path, framework_repo
):
    import hashlib
    import json

    puzzles = make_puzzles(6)
    sample = tmp_path / "sample.tsv"
    write_sample(sample, puzzles)
    network = tmp_path / "net.nnue"
    network.write_bytes(b"network")
    engine = fake_engine_row(store, oracle(answers_for(puzzles)))
    run_id, _ = protocol.run(
        config,
        store,
        engine,
        sample,
        Limit("nodes", 100),
        eval_file=network,
        workers=2,
        framework=framework_repo,
        progress=lambda t: None,
    )
    limits = json.loads(store.get_run(run_id)["limits_json"])

    assert limits["options"]["EvalFile"] == str(network.resolve())
    assert limits["eval_file_sha256"] == hashlib.sha256(b"network").hexdigest()
