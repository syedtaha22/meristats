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

import sys
from pathlib import Path

from meristats import datasets, preflight, runner, stats
from meristats.build import BuiltEngine
from meristats.config import Config
from meristats.preflight import require
from meristats.protocols import match
from meristats.protocols.match import Limit, Player
from meristats.store import Store

PROTOCOL = match.LADDER
VERSION = match.LADDER_VERSION


def run(
    config: Config,
    store: Store,
    engine: BuiltEngine,
    stockfish: Path,
    stockfish_id: int,
    book: Path,
    limit: Limit,
    levels: list[int],
    *,
    a_config: str | None = None,
    rounds: int = 25,
    seed: int = 1,
    concurrency: int | None = None,
    hash_mb: int = 16,
    allow_dirty: bool = False,
    framework=...,
    check_compliance: bool = True,
    progress=lambda text: print(text, file=sys.stderr, flush=True),
) -> tuple[str, dict]:
    """
    Plays a match against Stockfish at each strength level (`UCI_LimitStrength` and `UCI_Elo`),
    each with `rounds` pairs of games and no sequential test, and fits the rating of the engine
    from all levels together. The levels are the fixed ratings `stockfish-<elo>` of the fit.

    The rating is on the scale of `UCI_Elo`, which Stockfish calibrates for longer time controls,
    so it is a rating for this speed on this machine, not a rating-list rating.
    """
    if not levels:
        raise ValueError("no levels")

    context = runner.begin(config, store, allow_dirty=allow_dirty, framework=framework)
    book_id, book_hash = datasets.register_file(
        store, "opening-book", "8moves_v3", book, {"file": book.name}, keep_copy=False
    )
    options = {"Threads": 1, "Hash": hash_mb}

    if limit.kind == "st":
        options["MoveOverhead"] = 0

    if a_config is not None:
        options["Config"] = a_config

    a = Player("A", engine.binary, options, engine.engine_id)
    checks = runner.engine_checks(engine) + [preflight.dataset_hash(book, book_hash)]

    if check_compliance and config.fastchess:
        checks.append(match.compliance(Path(config.fastchess), stockfish))

    context.checks += require(checks)
    context.sample("before")
    per_level, blobs, results = {}, [], []

    for level in levels:
        b = Player(
            "B",
            stockfish,
            {"Threads": 1, "Hash": hash_mb, "UCI_LimitStrength": "true", "UCI_Elo": level},
            stockfish_id,
        )
        work = config.cache_dir / "matches" / f"{context.started_utc.replace(':', '')}-{level}"
        pgn_text, output = match.play(
            config,
            a,
            b,
            limit,
            book,
            work,
            rounds=rounds,
            sprt=None,
            seed=seed,
            concurrency=concurrency,
        )
        games = match.parse_games(pgn_text)
        summary = match.summarise(games, None, output)
        per_level[str(level)] = summary
        results.append(
            ("A", f"stockfish-{level}", summary["wins"], summary["draws"], summary["losses"])
        )
        blobs.append((store.put_blob(pgn_text.encode(), "pgn", f"games-{level}.pgn").sha256, "pgn"))
        blobs.append(
            (
                store.put_blob(output.encode(), "tool-log", f"fastchess-output-{level}.txt").sha256,
                "tool-log",
            )
        )
        progress(f"level {level}: +{summary['wins']} ={summary['draws']} -{summary['losses']}")

    context.sample("end")
    fitted = stats.rating_fit(results, stats.fixed_ratings(r[1] for r in results))
    rating = fitted.get("A")
    summary = {
        "levels": per_level,
        "rating": None
        if rating is None
        else {"value": rating[0], "low": rating[1], "high": rating[2]},
        "games": sum(s["games"] for s in per_level.values()),
    }
    run_id = runner.finish(
        context,
        protocol=PROTOCOL,
        version=VERSION,
        engine=engine,
        config=a_config,
        limits={
            "limit": {limit.kind: limit.value},
            "levels": levels,
            "rounds": rounds,
            "seed": seed,
            "options_a": options,
            "stockfish_engine_id": stockfish_id,
        },
        dataset_id=book_id,
        summary=summary,
        items_blob=None,
        blobs=tuple(blobs),
        seed=seed,
    )
    return run_id, summary
