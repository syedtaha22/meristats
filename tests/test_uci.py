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

import pytest

from meristats.uci import EngineCrashed, EngineError, EngineTimeout, UciEngine, parse_info


def test_parse_info_reads_depth_score_nodes_time_and_pv():
    info = parse_info(
        "info depth 12 seldepth 20 score cp -34 nodes 12345 nps 99 time 7 pv e2e4 e7e5 g1f3"
    )

    assert (info.depth, info.score_kind, info.score, info.nodes, info.time_ms) == (
        12,
        "cp",
        -34,
        12345,
        7,
    )
    assert info.pv == ["e2e4", "e7e5", "g1f3"]


def test_parse_info_reads_mate_scores_and_bounds():
    info = parse_info("info depth 3 score mate -2 lowerbound nodes 9 pv a1a2")

    assert (info.score_kind, info.score, info.nodes, info.pv) == ("mate", -2, 9, ["a1a2"])


@pytest.mark.parametrize(
    "line",
    [
        "",
        "bestmove e2e4",
        "info string depth 5",
        "info nodes 5",
        "info depth x",
        "info depth 5 score cp y",
        "readyok",
    ],
)
def test_parse_info_ignores_lines_that_are_not_search_infos(line):
    assert parse_info(line) is None


def test_handshake_collects_name_and_options(fake_engine):
    path, args = fake_engine()

    with UciEngine(path, args) as engine:
        assert engine.name == "Fake 1.0"
        assert engine.options["Hash"] == {"type": "spin", "default": "16"}
        assert engine.options["OwnBook"] == {"type": "check", "default": "false"}
        assert engine.options["Config"]["default"] == "network"
        assert engine.options["EvalFile"]["default"] == "<empty>"
        assert engine.options["Clear Hash"] == {"type": "button", "default": None}


def test_go_returns_the_move_and_the_infos(fake_engine):
    path, args = fake_engine()

    with UciEngine(path, args) as engine:
        engine.position(None, ["e2e4"])
        result = engine.go("depth 2")

    assert result.bestmove == "e2e4"
    assert [info.depth for info in result.infos] == [1, 2]
    assert result.infos[1].score_kind == "mate"
    assert result.strings == ["hello"] and result.elapsed >= 0


def test_set_option_returns_the_info_strings(fake_engine):
    path, args = fake_engine()

    with UciEngine(path, args) as engine:
        assert engine.set_option("Config", "baseline") == ["Config: baseline"]
        assert engine.set_option("Config", "bad") == ["invalid Config: unknown preset 'bad'"]
        assert engine.set_option("Hash", "32") == []
        assert engine.set_option("Clear Hash") == []


def test_an_unknown_option_is_refused_before_it_is_sent(fake_engine):
    path, args = fake_engine()

    with UciEngine(path, args) as engine, pytest.raises(EngineError, match="no option Nope"):
        engine.set_option("Nope", "1")


def test_new_game_and_position_with_a_fen(fake_engine):
    path, args = fake_engine()

    with UciEngine(path, args) as engine:
        engine.new_game()
        engine.position("8/8/8/8/8/8/8/K6k w - - 0 1")
        assert engine.go("nodes 1").bestmove == "e2e4"


def test_a_silent_engine_times_out(fake_engine):
    path, args = fake_engine("silent")

    with UciEngine(path, args) as engine, pytest.raises(EngineTimeout, match="no bestmove"):
        engine.go("depth 1", timeout=0.4)


def test_an_engine_that_answers_only_after_stop_is_accepted(fake_engine):
    path, args = fake_engine("late")

    with UciEngine(path, args) as engine:
        assert engine.go("infinite", timeout=0.4).bestmove == "e2e4"


def test_a_crashing_engine_is_reported(fake_engine):
    path, args = fake_engine("crash")

    with UciEngine(path, args) as engine, pytest.raises(EngineCrashed):
        engine.go("depth 1", timeout=5)


def test_a_missing_uciok_times_out(fake_engine):
    path, args = fake_engine("no_uciok")

    with pytest.raises(EngineTimeout):
        UciEngine(path, args, startup_timeout=0.4).start()


def test_sending_to_an_ended_engine_is_reported(fake_engine):
    path, args = fake_engine()
    engine = UciEngine(path, args).start()
    engine.quit()

    with pytest.raises(EngineCrashed):
        engine.send("isready")


def test_quit_twice_is_harmless(fake_engine):
    path, args = fake_engine()
    engine = UciEngine(path, args).start()
    engine.quit()
    engine.quit()


def test_a_program_that_does_not_exist_fails_to_start(tmp_path):
    with pytest.raises(OSError):
        UciEngine(tmp_path / "missing").start()


def test_perft_reads_the_node_count(fake_engine):
    path, args = fake_engine()

    with UciEngine(path, args) as engine:
        engine.position(None)
        nodes, seconds = engine.perft(3)

    assert nodes == 8902 and seconds >= 0


def test_an_unreadable_perft_reply_is_an_error(fake_engine):
    path, args = fake_engine("badperft")

    with UciEngine(path, args) as engine, pytest.raises(EngineError, match="unreadable perft"):
        engine.perft(3)


def test_an_engine_that_ignores_quit_is_killed(fake_engine, monkeypatch):
    from meristats import uci

    monkeypatch.setattr(uci, "QUIT_TIMEOUT", 0.3)
    path, args = fake_engine("stubborn")
    engine = UciEngine(path, args).start()
    process = engine.process
    engine.quit()

    assert process.poll() is not None and engine.process is None


def test_an_engine_that_ends_right_after_the_handshake_is_reported(fake_engine):
    path, args = fake_engine("exit_after_uci")
    engine = UciEngine(path, args).start()

    with pytest.raises(EngineCrashed):
        engine.ready(timeout=2)

    engine.quit()


def test_a_pinned_engine_runs_on_that_cpu(fake_engine):
    import os

    path, args = fake_engine()
    cpu = sorted(os.sched_getaffinity(0))[0]

    with UciEngine(path, args, cpu=cpu) as engine:
        assert engine.name == "Fake 1.0"
        affinity = os.sched_getaffinity(engine.process.pid)

    assert affinity == {cpu}
