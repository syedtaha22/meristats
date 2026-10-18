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

import math

import pytest

from meristats import stats


def test_wilson_interval_of_half():
    low, high = stats.wilson_interval(50, 100)

    assert low == pytest.approx(0.4038, abs=1e-4) and high == pytest.approx(0.5962, abs=1e-4)


def test_wilson_interval_at_the_edges_stays_in_range():
    low, high = stats.wilson_interval(0, 10)
    assert low == pytest.approx(0.0) and 0 < high < 0.4

    low, high = stats.wilson_interval(10, 10)
    assert high == pytest.approx(1.0) and 0.6 < low < 1

    assert stats.wilson_interval(0, 0) == (0.0, 1.0)


def test_weighted_share_with_equal_weights_matches_the_plain_share():
    result = stats.weighted_share([True] * 30 + [False] * 70, [2.0] * 100)

    assert result["p"] == pytest.approx(0.3) and result["n_eff"] == pytest.approx(100)
    assert result["low"] < 0.3 < result["high"]


def test_weighted_share_with_uneven_weights_has_a_smaller_effective_sample():
    weights = [10.0] * 10 + [1.0] * 90
    result = stats.weighted_share([True] * 10 + [False] * 90, weights)

    assert result["p"] == pytest.approx(100 / 190)
    assert result["n_eff"] < 100
    assert stats.weighted_share([], [])["p"] == 0.0


@pytest.mark.parametrize(
    "only_a,only_b,expected",
    [
        (0, 5, 0.0625),
        (3, 3, 1.0),
        (0, 0, 1.0),
        (10, 2, 2 * (1 + 12 + 66) / 4096),
        (2, 10, 2 * (1 + 12 + 66) / 4096),
    ],
)
def test_mcnemar_exact(only_a, only_b, expected):
    assert stats.mcnemar_exact(only_a, only_b) == pytest.approx(expected)


def test_paired_difference_counts_discordant_items():
    a = [True, True, True, False, False, True]
    b = [True, False, False, False, True, True]
    result = stats.paired_difference(a, b)

    assert result["only_a"] == 2 and result["only_b"] == 1
    assert result["difference"] == pytest.approx(1 / 6)
    assert result["low"] < result["difference"] < result["high"]


def test_paired_difference_of_identical_runs_is_zero():
    flags = [True, False, True, True]
    result = stats.paired_difference(flags, flags)

    assert result["difference"] == 0 and result["p"] == 1.0
    assert result["low"] == 0 == result["high"]


def test_paired_difference_uses_weights_and_checks_lengths():
    heavy = stats.paired_difference([True, False], [False, False], [9.0, 1.0])
    assert heavy["difference"] == pytest.approx(0.9)

    with pytest.raises(ValueError):
        stats.paired_difference([True], [True, False])


def test_puzzle_rating_follows_the_share_solved():
    ratings = list(range(400, 2800, 20)) * 5
    weights = [1.0] * len(ratings)
    easy = stats.puzzle_rating(ratings, weights, [r < 2200 for r in ratings])
    hard = stats.puzzle_rating(ratings, weights, [r < 1200 for r in ratings])

    assert 2000 < easy < 2400 and 1000 < hard < 1400 and easy > hard


def test_puzzle_rating_ignores_a_weight_of_zero():
    base = stats.puzzle_rating([1000, 2000], [1, 1], [True, False])
    same = stats.puzzle_rating([1000, 2000, 1500], [1, 1, 0], [True, False, False])

    assert base == pytest.approx(same, abs=1e-6)


def test_score_and_elo_are_inverse():
    for elo in (-300, 0, 57, 400):
        assert stats.elo_of(stats.score_of(elo)) == pytest.approx(elo)

    assert stats.elo_of(0) == -math.inf and stats.elo_of(1) == math.inf


def test_match_summary_matches_the_old_tool_on_a_recorded_result():
    # pruning (50ms) vs iterative (50ms): +135 =29 -36, score 0.748, Elo +189 (+142 to +243)
    summary = stats.match_summary(135, 29, 36)

    assert summary.games == 200 and summary.score == pytest.approx(0.7475)
    assert summary.score_low == pytest.approx(0.693, abs=0.001)
    assert summary.score_high == pytest.approx(0.802, abs=0.001)
    assert summary.elo == pytest.approx(189, abs=1) and summary.elo_low == pytest.approx(142, abs=2)
    assert summary.elo_high == pytest.approx(243, abs=2)


def test_match_summary_needs_games():
    with pytest.raises(ValueError):
        stats.match_summary(0, 0, 0)


def test_pentanomial_counts_pairs_by_half_points():
    assert stats.pentanomial([(2, 2), (0, 0), (1, 1), (2, 0), (2, 1), (0, 1)]) == [1, 1, 2, 1, 1]


def test_sprt_llr_is_zero_without_pairs_and_follows_the_evidence():
    assert stats.sprt_llr([0, 0, 0, 0, 0], 0, 10) == 0.0
    winning = stats.sprt_llr([2, 5, 20, 40, 30], 0, 10)
    losing = stats.sprt_llr([30, 40, 20, 5, 2], 0, 10)

    assert winning > 0 > losing
    assert stats.sprt_llr([10, 10, 10, 10, 10], 0, 10) == pytest.approx(
        stats.sprt_llr([20, 20, 20, 20, 20], 0, 10) / 2, rel=0.1
    )


def test_sprt_bounds_and_decisions():
    low, high = stats.sprt_bounds(0.05, 0.05)

    assert low == pytest.approx(-2.944, abs=1e-3) and high == pytest.approx(2.944, abs=1e-3)
    assert stats.sprt_decision(3.0, 0.05, 0.05) == "H1"
    assert stats.sprt_decision(-3.0, 0.05, 0.05) == "H0"
    assert stats.sprt_decision(0.0, 0.05, 0.05) == "continue"


RESULTS = [
    ("engine-x", "stockfish-2000", 90, 6, 4),
    ("engine-x", "stockfish-2400", 70, 10, 20),
    ("engine-x", "stockfish-2800", 35, 15, 50),
    ("engine-x", "stockfish-3000", 10, 12, 78),
    ("engine-y", "stockfish-2400", 40, 20, 40),
    ("engine-y", "engine-x", 30, 20, 50),
]


def test_rating_fit_matches_the_old_script_on_the_same_games():
    names = {n for r in RESULTS for n in r[:2]}
    fitted = stats.rating_fit(RESULTS, stats.fixed_ratings(names))

    # the numbers that scripts/rating_fit.py printed for these games
    assert set(fitted) == {"engine-x", "engine-y"}
    assert round(fitted["engine-x"][0]) == 2634
    assert (round(fitted["engine-x"][1]), round(fitted["engine-x"][2])) == (2590, 2678)
    assert round(fitted["engine-y"][0]) == 2478
    assert (round(fitted["engine-y"][1]), round(fitted["engine-y"][2])) == (2423, 2533)


def test_rating_fit_of_equal_results_gives_the_fixed_rating():
    fitted = stats.rating_fit([("a", "stockfish-2000", 50, 0, 50)], {"stockfish-2000": 2000.0})

    assert fitted["a"][0] == pytest.approx(2000.0, abs=1e-6)
    assert fitted["a"][1] < 2000 < fitted["a"][2]


def test_rating_fit_ignores_pairs_without_games_and_needs_a_free_player():
    assert (
        stats.rating_fit(
            [("stockfish-2000", "stockfish-2200", 5, 0, 5)],
            {"stockfish-2000": 2000.0, "stockfish-2200": 2200.0},
        )
        == {}
    )
    fitted = stats.rating_fit(
        [("a", "stockfish-2000", 0, 0, 0), ("a", "stockfish-2000", 9, 0, 1)],
        {"stockfish-2000": 2000.0},
    )
    assert fitted["a"][0] > 2000


def test_fixed_ratings_reads_the_number_in_the_name():
    assert stats.fixed_ratings(["stockfish-1800", "engine", "stockfish-2500"]) == {
        "stockfish-1800": 1800.0,
        "stockfish-2500": 2500.0,
    }


def test_a_player_with_only_full_wins_or_losses_has_no_rating():
    perfect = [("a", "stockfish-2000", 10, 0, 0), ("a", "stockfish-2400", 10, 0, 0)]
    nothing = [("b", "stockfish-2000", 0, 0, 10)]

    assert (
        stats.rating_fit(perfect, stats.fixed_ratings({"stockfish-2000", "stockfish-2400"})) == {}
    )
    assert stats.rating_fit(nothing, {"stockfish-2000": 2000.0}) == {}


def test_a_rated_player_stays_rated_when_another_has_no_rating():
    results = [
        ("a", "stockfish-2000", 6, 0, 4),
        ("a", "stockfish-2400", 2, 0, 8),
        ("b", "stockfish-2000", 10, 0, 0),
    ]
    fitted = stats.rating_fit(results, {"stockfish-2000": 2000.0, "stockfish-2400": 2400.0})

    assert set(fitted) == {"a"} and 2000 < fitted["a"][0] < 2400


def test_a_player_with_one_partial_result_among_full_ones_is_rated():
    results = [("a", "stockfish-2000", 10, 0, 0), ("a", "stockfish-2400", 4, 0, 6)]
    fitted = stats.rating_fit(results, {"stockfish-2000": 2000.0, "stockfish-2400": 2400.0})

    assert "a" in fitted
