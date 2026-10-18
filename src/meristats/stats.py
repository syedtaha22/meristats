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
from collections.abc import Sequence
from dataclasses import dataclass

Z95 = 1.959963984540054


def wilson_interval(successes: float, n: float, z: float = Z95) -> tuple[float, float]:
    """
    The Wilson score interval of a proportion. `n` may be fractional, as the effective sample
    size of a weighted sample is.
    """
    if n <= 0:
        return 0.0, 1.0

    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def weighted_share(flags: Sequence[bool], weights: Sequence[float]) -> dict:
    """
    The weighted share of true flags, with a Wilson interval that uses the effective sample
    size, (sum of weights) squared over the sum of squared weights.
    """
    total = sum(weights)

    if total <= 0:
        return {"p": 0.0, "low": 0.0, "high": 1.0, "n": len(flags), "n_eff": 0.0}

    p = sum(weight for flag, weight in zip(flags, weights) if flag) / total
    n_eff = total * total / sum(weight * weight for weight in weights)
    low, high = wilson_interval(p * n_eff, n_eff)
    return {"p": p, "low": low, "high": high, "n": len(flags), "n_eff": n_eff}


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """
    The two-sided p value of the exact McNemar test, given the numbers of items that only A
    solved and that only B solved.
    """
    n = only_a + only_b

    if n == 0:
        return 1.0

    smaller = min(only_a, only_b)
    tail = sum(math.comb(n, k) for k in range(smaller + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_difference(
    a: Sequence[bool], b: Sequence[bool], weights: Sequence[float] | None = None
) -> dict:
    """
    The difference in the share solved, A minus B, on the same items, with a 95% interval and the
    exact McNemar p value. The interval comes from the delta method on the weighted differences.
    """
    if len(a) != len(b):
        raise ValueError("the runs have different numbers of items")

    weights = list(weights) if weights is not None else [1.0] * len(a)
    total = sum(weights)
    differences = [int(x) - int(y) for x, y in zip(a, b)]
    mean = sum(w * d for w, d in zip(weights, differences)) / total
    variance = sum((w * (d - mean)) ** 2 for w, d in zip(weights, differences)) / total**2
    half = Z95 * math.sqrt(variance)
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    return {
        "difference": mean,
        "low": mean - half,
        "high": mean + half,
        "only_a": only_a,
        "only_b": only_b,
        "p": mcnemar_exact(only_a, only_b),
    }


def logistic_win_probability(rating: float, opponent: float) -> float:
    """
    The expected score of a player against another on the Elo scale.
    """
    return 1.0 / (1.0 + 10.0 ** ((opponent - rating) / 400.0))


def puzzle_rating(
    ratings: Sequence[float], weights: Sequence[float], solved: Sequence[bool]
) -> float:
    """
    The rating E for which the logistic model of solving a puzzle of a given rating fits the
    weighted results best, by likelihood. The likelihood is concave in E, so a ternary search
    finds the maximum.
    """
    data = list(zip(ratings, weights, solved))

    def loglikelihood(rating: float) -> float:
        total = 0.0

        for puzzle_rating_, weight, ok in data:
            p = min(max(logistic_win_probability(rating, puzzle_rating_), 1e-9), 1 - 1e-9)
            total += weight * math.log(p if ok else 1 - p)

        return total

    low, high = 0.0, 4000.0

    for _ in range(60):
        a, b = low + (high - low) / 3, high - (high - low) / 3

        if loglikelihood(a) < loglikelihood(b):
            low = a
        else:
            high = b

    return (low + high) / 2


# ----------------------------------------------------------------- matches


def score_of(elo: float) -> float:
    """
    The expected score of a player that is `elo` points stronger than its opponent.
    """
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def elo_of(score: float) -> float:
    """
    The Elo difference that gives an expected score, or an infinite value at 0 and 1.
    """
    if score <= 0:
        return -math.inf

    if score >= 1:
        return math.inf

    return -400.0 * math.log10(1.0 / score - 1.0)


@dataclass(frozen=True)
class MatchSummary:
    games: int
    score: float
    score_low: float
    score_high: float
    elo: float
    elo_low: float
    elo_high: float


def match_summary(wins: int, draws: int, losses: int) -> MatchSummary:
    """
    The score of a match, its 95% interval from the variance of the game results, and the Elo
    difference of each.
    """
    games = wins + draws + losses

    if games == 0:
        raise ValueError("no games")

    score = (wins + draws / 2) / games
    variance = (wins * (1 - score) ** 2 + draws * (0.5 - score) ** 2 + losses * score**2) / games
    half = Z95 * math.sqrt(variance / games)
    low, high = max(0.0, score - half), min(1.0, score + half)
    return MatchSummary(games, score, low, high, elo_of(score), elo_of(low), elo_of(high))


def pentanomial(results: Sequence[tuple[int, int]]) -> list[int]:
    """
    Counts pairs of games by their points in half points. Each pair is the results of the two
    games of an opening, each 0 for a loss, 1 for a draw and 2 for a win, for the engine under
    test. Returns the counts of the pairs that scored 0, 1, 2, 3 and 4 half points.
    """
    counts = [0] * 5

    for first, second in results:
        counts[first + second] += 1

    return counts


def sprt_llr(pairs: Sequence[float], elo0: float, elo1: float) -> float:
    """
    The log likelihood ratio of "the engine is `elo1` stronger" against "it is `elo0` stronger",
    from the pentanomial counts. It uses the normal approximation of the mean score of a pair,
    with the variance of the pair scores, so that the correlation between the two games of an
    opening is included. Every category gets half a pair, because without it the variance of the
    first pairs is far too small.
    """
    n = sum(pairs)

    if n <= 0:
        return 0.0

    counts = [count + 0.5 for count in pairs]
    total = sum(counts)
    mean = sum(count * k / 4 for k, count in enumerate(counts)) / total
    variance = sum(count * (k / 4 - mean) ** 2 for k, count in enumerate(counts)) / total
    s0, s1 = score_of(elo0), score_of(elo1)
    return n * (s1 - s0) * (2 * mean - s0 - s1) / (2 * variance)


def sprt_bounds(alpha: float, beta: float) -> tuple[float, float]:
    """
    The lower and the upper stopping bound of the test.
    """
    return math.log(beta / (1 - alpha)), math.log((1 - beta) / alpha)


def sprt_decision(llr: float, alpha: float, beta: float) -> str:
    """
    `H1` when the ratio reaches the upper bound, `H0` at the lower bound, else `continue`.
    """
    low, high = sprt_bounds(alpha, beta)

    if llr >= high:
        return "H1"

    if llr <= low:
        return "H0"

    return "continue"


# ------------------------------------------------------------------ ratings


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """
    Solves matrix * x = rhs by Gaussian elimination with partial pivoting.
    """
    n = len(rhs)
    matrix = [row[:] for row in matrix]
    rhs = rhs[:]

    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(matrix[row][column]))
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        rhs[column], rhs[pivot] = rhs[pivot], rhs[column]

        for row in range(column + 1, n):
            factor = matrix[row][column] / matrix[column][column]

            for k in range(column, n):
                matrix[row][k] -= factor * matrix[column][k]

            rhs[row] -= factor * rhs[column]

    x = [0.0] * n

    for row in reversed(range(n)):
        total = rhs[row] - sum(matrix[row][k] * x[k] for k in range(row + 1, n))
        x[row] = total / matrix[row][row]

    return x


def rating_fit(
    results: Sequence[tuple[str, str, int, int, int]],
    fixed: dict[str, float],
    start: float = 1000.0,
) -> dict[str, tuple[float, float, float]]:
    """
    Fits ratings to the results of games between several players. Each result is the names of
    two players and the wins, draws and losses of the first, and a draw is half a point. A player
    in `fixed` keeps its rating. The others are found by maximum likelihood under the logistic
    model, and each has a 95% interval of 1.96 standard errors from the information matrix.
    Returns the rating, the low end and the high end of every player that is not fixed.

    A player whose every result is a full win or a full loss has no maximum of the likelihood, so
    its rating is not determined. Such a player is left out of the result, and the others are
    fitted without its games.
    """
    names: list[str] = []

    def index(name: str) -> int:
        if name not in names:
            names.append(name)

        return names.index(name)

    pairs = []

    for a, b, wins, draws, losses in results:
        games = wins + draws + losses

        if games:
            pairs.append((index(a), index(b), games, (wins + 0.5 * draws) / games))

    undetermined = {
        name
        for name in names
        if name not in fixed
        and all(
            score in (0.0, 1.0) for a, b, _, score in pairs if names[a] == name or names[b] == name
        )
    }
    pairs = [
        (a, b, games, score)
        for a, b, games, score in pairs
        if names[a] not in undetermined and names[b] not in undetermined
    ]
    rating = [fixed.get(name, start) for name in names]
    free = [
        i
        for i, name in enumerate(names)
        if name not in fixed
        and name not in undetermined
        and any(i in (a, b) for a, b, _, _ in pairs)
    ]

    if not free:
        return {}

    slot = {player: k for k, player in enumerate(free)}
    scale = math.log(10) / 400

    def system():
        gradient = [0.0] * len(free)
        information = [[0.0] * len(free) for _ in free]

        for a, b, games, score in pairs:
            p = logistic_win_probability(rating[a], rating[b])
            g = games * (score - p) * scale
            h = games * p * (1 - p) * scale * scale

            if a in slot:
                gradient[slot[a]] += g
                information[slot[a]][slot[a]] += h

            if b in slot:
                gradient[slot[b]] -= g
                information[slot[b]][slot[b]] += h

            if a in slot and b in slot:
                information[slot[a]][slot[b]] -= h
                information[slot[b]][slot[a]] -= h

        return gradient, information

    for _ in range(100):
        gradient, information = system()
        step = [max(-200.0, min(200.0, s)) for s in _solve(information, gradient)]

        for i in free:
            rating[i] += step[slot[i]]

        if max(abs(s) for s in step) < 1e-9:
            break

    _, information = system()
    fitted = {}

    for i in free:
        unit = [0.0] * len(free)
        unit[slot[i]] = 1.0
        margin = Z95 * math.sqrt(_solve(information, unit)[slot[i]])
        fitted[names[i]] = (rating[i], rating[i] - margin, rating[i] + margin)

    return fitted


def fixed_ratings(names: Sequence[str]) -> dict[str, float]:
    """
    The fixed ratings of the players named `stockfish-<elo>`, with the number in the name as the
    rating.
    """
    return {name: float(name.split("-", 1)[1]) for name in names if name.startswith("stockfish-")}
