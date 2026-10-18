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

import argparse
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

from meristats import __version__, datasets, external
from meristats.build import BuildError, build_engine
from meristats.config import Config, ConfigError, load
from meristats.protocols import experiment as experiment_protocol
from meristats.protocols import job as job_protocol
from meristats.protocols import ladder as ladder_protocol
from meristats.protocols import match as match_protocol
from meristats.protocols import positions as positions_protocol
from meristats.protocols import puzzles as puzzles_protocol
from meristats.protocols import train as train_protocol
from meristats.protocols import rules as rules_protocol
from meristats.protocols import search_nodes as search_nodes_protocol
from meristats.protocols import spsa as spsa_protocol
from meristats.preflight import PreflightError
from meristats import magics
from meristats.publish import publish
from meristats.serve import make_server
from meristats.report import (
    ReportError,
    compare_puzzles,
    default_label,
    find_run,
    format_comparison,
    format_run,
    limit_text,
    save_comparison,
)
from meristats.store import Store, StoreError
from meristats.uci import EngineError

DATASETS = Path(__file__).resolve().parents[2] / "datasets"


def _tool(command: list[str]) -> str | None:
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=30)
        return (done.stdout or done.stderr).splitlines()[0].strip()
    except (OSError, subprocess.TimeoutExpired, IndexError):
        return None


def selftest() -> int:
    """
    Runs the tests of meristats with pytest, from a checkout.
    """
    tests = Path(__file__).resolve().parents[2] / "tests"

    if not tests.is_dir():
        print("meristats: no tests folder, run selftest from a checkout", file=sys.stderr)
        return 1

    try:
        import pytest
    except ImportError:
        print("meristats: pytest is not installed (pip install -e '.[test]')", file=sys.stderr)
        return 1

    return int(pytest.main(["-q", str(tests)]))


def doctor(config_path: Path | None) -> int:
    """
    Checks the installation and prints one line for each check. Returns 1 when a check failed.
    """
    failed = 0

    def report(ok: bool | None, name: str, detail: str = "") -> None:
        nonlocal failed
        mark = "ok  " if ok else ("warn" if ok is None else "FAIL")
        failed += ok is False
        print(f"{mark} {name}{': ' + detail if detail else ''}")

    report(sys.version_info >= (3, 12), "python", sys.version.split()[0])

    for module in ("chess", "pyarrow", "zstandard"):
        try:
            imported = importlib.import_module(module)
            report(True, f"package {module}", getattr(imported, "__version__", ""))
        except ImportError:
            report(False, f"package {module}", "not installed")

    for name, command in (
        ("git", ["git", "--version"]),
        ("cmake", ["cmake", "--version"]),
        ("g++", ["g++", "--version"]),
    ):
        line = _tool(command)
        report(line is not None, name, line or "not found")

    try:
        config = load(config_path)
    except ConfigError as error:
        report(False, "settings", str(error))
        return 1

    report(True, "settings", f"{config.workers} workers")
    report((config.engine_repo / ".git").exists(), "engine repository", str(config.engine_repo))

    for label, path in (("data folder", config.data_dir), ("cache folder", config.cache_dir)):
        probe = path if path.exists() else path.parent
        free = shutil.disk_usage(probe).free / 1e9 if probe.exists() else 0
        report(free > 5, label, f"{free:.0f} GB free at {path}")

    for label, path in (("fastchess", config.fastchess), ("stockfish", config.stockfish)):
        if path is None:
            report(None, label, "not set")
        else:
            line = _tool([str(path), "--version"] if label == "fastchess" else [str(path)])
            report(path.exists(), label, line or str(path))

    try:
        with Store(config.data_dir) as store:
            report(True, "database", f"schema {store.schema_version()}")
    except (StoreError, OSError) as error:
        report(False, "database", str(error))

    return 1 if failed else 0


# ------------------------------------------------------------------ protocols


def _limit(args) -> puzzles_protocol.Limit:
    if args.depth:
        return puzzles_protocol.Limit("depth", args.depth)

    if args.movetime:
        return puzzles_protocol.Limit("movetime", args.movetime)

    return puzzles_protocol.Limit("nodes", args.nodes or 20000)


def _run_puzzles(args, config: Config, store: Store) -> int:
    dataset = args.dataset or config.cache_dir / "datasets" / "puzzles-sample-seed1.tsv"
    engine = build_engine(config, store, args.engine)
    run_id, summary = puzzles_protocol.run(
        config,
        store,
        engine,
        dataset,
        _limit(args),
        engine_config=args.engine_config,
        eval_file=args.eval_file,
        items=args.items,
        workers=args.workers,
        hash_mb=args.hash_mb,
        allow_dirty=args.allow_dirty,
    )
    whole = summary["whole"]
    print(run_id)
    print(
        f"solved {100 * whole['p']:.2f}% ({100 * whole['low']:.2f} to {100 * whole['high']:.2f}), "
        f"first move {100 * whole['first_move']:.2f}%, puzzle rating "
        f"{whole['puzzle_rating']:.0f}, mean nodes {whole['mean_nodes']:.0f}, "
        f"errors {summary['errors']['count']}"
    )
    return 0


def _run_search_nodes(args, config: Config, store: Store) -> int:
    engine = build_engine(config, store, args.engine)
    names = [name or None for name in args.configs.split(",")] if args.configs else [None]
    run_id, summary = search_nodes_protocol.run(
        config,
        store,
        engine,
        args.suite or DATASETS / "suite.txt",
        names,
        [int(d) for d in args.depths.split(",")],
        rounds=args.rounds,
        cpu=args.cpu,
        allow_dirty=args.allow_dirty,
    )
    print(run_id)

    for name, depths in summary.items():
        for depth, row in depths.items():
            print(
                f"{name:>12} depth {depth:>2}: {row['mean_nodes']:>12.0f} nodes "
                f"{row['mean_ms']:>9.2f} ms {row['nodes_per_second'] or 0:>12.0f} nps"
            )

    return 0


def _run_rules(args, config: Config, store: Store) -> int:
    engine = build_engine(config, store, args.engine)
    run_id, summary = rules_protocol.run(
        config,
        store,
        engine,
        args.perft or DATASETS / "perft.txt",
        max_nodes=args.max_nodes,
        cpu=args.cpu,
        allow_dirty=args.allow_dirty,
    )
    print(run_id)
    print(f"{summary['counts']} counts, all correct: {summary['all_correct']}")

    for line in summary["wrong"]:
        print("WRONG", line)

    for name, speed in summary["speed"].items():
        print(f"{name:>10} depth {speed['depth']}: {speed['nodes_per_second'] / 1e6:.1f} Mnps")

    return 0 if summary["all_correct"] else 1


def _match_limit(args) -> match_protocol.Limit:
    if args.depth:
        return match_protocol.Limit("depth", args.depth)

    if args.nodes:
        return match_protocol.Limit("nodes", args.nodes)

    return match_protocol.Limit("st", args.st)


def _book(config: Config, path: Path | None) -> Path:
    return path or external.fetch_book(config.cache_dir)


def _run_match(args, config: Config, store: Store) -> int:
    a = build_engine(config, store, args.a)
    b = build_engine(config, store, args.b)
    sprt = None if args.no_sprt else match_protocol.Sprt(args.elo0, args.elo1)
    run_id, summary = match_protocol.run(
        config,
        store,
        a,
        b,
        _book(config, args.book),
        _match_limit(args),
        a_config=args.a_config,
        b_config=args.b_config,
        a_eval_file=args.a_eval_file,
        b_eval_file=args.b_eval_file,
        rounds=args.rounds,
        sprt=sprt,
        seed=args.seed,
        allow_dirty=args.allow_dirty,
        check_compliance=not args.no_compliance,
    )
    print(run_id)
    print(
        f"{summary['games']} games: +{summary['wins']} ={summary['draws']} "
        f"-{summary['losses']}, score {100 * summary.get('score', 0):.1f}%, "
        f"Elo {summary.get('elo', 0):+.0f}"
    )

    if "sprt" in summary:
        test = summary["sprt"]
        print(
            f"SPRT [{test['elo0']:g}, {test['elo1']:g}]: LLR {test['llr']:.2f} "
            f"(bounds {test['bounds'][0]:.2f}, {test['bounds'][1]:.2f}), meristats "
            f"{test['verdict']}, fastchess {test['fastchess_verdict']}"
        )

    return 0


def _run_ladder(args, config: Config, store: Store) -> int:
    engine = build_engine(config, store, args.engine)
    stockfish = config.stockfish or external.fetch_stockfish(config.cache_dir)
    stockfish_id, _ = external.register_external(store, "stockfish", stockfish)
    run_id, summary = ladder_protocol.run(
        config,
        store,
        engine,
        stockfish,
        stockfish_id,
        _book(config, args.book),
        _match_limit(args),
        [int(n) for n in args.levels.split(",")],
        a_config=args.engine_config,
        rounds=args.rounds,
        seed=args.seed,
        allow_dirty=args.allow_dirty,
    )
    print(run_id)

    for level, row in summary["levels"].items():
        print(f"stockfish-{level}: +{row['wins']} ={row['draws']} -{row['losses']}")

    rating = summary["rating"]

    if rating:
        print(
            f"rating {rating['value']:.0f} ({rating['low']:.0f} to {rating['high']:.0f}) "
            f"in {summary['games']} games"
        )

    return 0


def _run_positions(args, config: Config, store: Store) -> int:
    engine = build_engine(config, store, args.engine)
    run_id, summary = positions_protocol.run(
        config,
        store,
        engine,
        args.out,
        games=args.games,
        depth=args.depth,
        engine_config=args.engine_config,
        eval_file=args.eval_file,
        random_plies=args.random_plies,
        skip=args.skip,
        seed=args.seed,
        workers=args.workers,
        allow_dirty=args.allow_dirty,
    )
    print(run_id)
    print(
        f"{summary['games']} games (White {summary['white_wins']}, draws {summary['draws']}, "
        f"Black {summary['black_wins']}), {summary['positions']} positions, "
        f"sha256 {summary['output']['sha256'][:16]}"
    )
    return 0


def _run_train(args, config: Config, store: Store) -> int:
    engine = build_engine(config, store, args.engine)
    run_id, summary = train_protocol.run(
        config,
        store,
        engine,
        args.data,
        out=args.out,
        source=args.source,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        lam=args.lam,
        max_score=args.max_score,
        limit=args.limit,
        seed=args.seed,
        init=args.init,
        device=args.device,
        workers=args.workers,
        allow_dirty=args.allow_dirty,
    )
    seconds = summary["seconds"]
    print(run_id)
    print(
        f"best held-out loss {summary['best_held']:.5f} at epoch {summary['best_epoch']} on "
        f"{summary['device']}: {seconds['loading']:.0f}s loading, {seconds['training']:.0f}s "
        f"training ({seconds['per_epoch']:.1f}s for each epoch)"
    )
    print(f"network {summary['network']['path']} ({summary['network']['sha256'][:16]})")
    return 0


def _run_job(args, config: Config, store: Store) -> int:
    engine = build_engine(config, store, args.engine)
    tool_args = args.tool_args[1:] if args.tool_args[:1] == ["--"] else args.tool_args
    run_id, summary = job_protocol.run(
        config,
        store,
        engine,
        args.tool,
        tool_args,
        inputs=args.input,
        outputs=args.output,
        timeout=args.timeout,
        allow_dirty=args.allow_dirty,
    )
    print(run_id)
    print(
        f"exit {summary['exit_code']} in {summary['seconds']:.1f}s; "
        f"tool {summary['tool_sha256'][:12]}"
    )

    for output in summary["outputs"]:
        print(f"{output['name']}  {output['bytes']} bytes  sha256 {output['sha256']}")

    for name in summary["missing"]:
        print(f"MISSING {name}")

    return 0 if summary["exit_code"] == 0 and not summary["missing"] else 1


def _run_experiment(args, config: Config, store: Store) -> int:
    engine = build_engine(config, store, args.engine)
    settings = experiment_protocol.Settings(
        puzzle_items=args.items,
        puzzle_nodes=args.nodes,
        depths=tuple(int(d) for d in args.depths.split(",")),
        st=args.st,
        rounds=args.rounds,
        sprt=match_protocol.Sprt(args.elo0, args.elo1),
        seed=args.seed,
        cpu=args.cpu,
    )
    dataset = args.dataset or config.cache_dir / "datasets" / "puzzles-sample-seed1.tsv"
    base, results = experiment_protocol.run_changes(
        config,
        store,
        engine,
        args.base,
        args.change,
        args.suite or DATASETS / "suite.txt",
        _book(config, args.book),
        dataset,
        settings,
        allow_dirty=args.allow_dirty,
    )

    for run_id, summary in results:
        print(f"{summary['new']}: {'kept' if summary['keep'] else 'dropped'}  {run_id}")

        for reason in summary["reasons"]:
            print(f"  {reason}")

    print(f"final base: {base}")
    return 0


def _run_spsa(args, config: Config, store: Store) -> int:
    engine = build_engine(config, store, args.engine)
    run_id, summary = spsa_protocol.run(
        config,
        store,
        engine,
        args.base,
        _book(config, args.book),
        match_protocol.Limit("st", args.st),
        iterations=args.iterations,
        pairs=args.pairs,
        allow_dirty=args.allow_dirty,
    )
    print(run_id)
    print(summary["tuned"])
    return 0


def _dataset(args, config: Config, store: Store) -> int:
    folder = config.cache_dir / "datasets"
    paths = datasets.fetch_puzzle_files(folder / "lichess-puzzles", reuse_from=tuple(args.reuse))
    take_all = tuple(int(n) for n in args.take_all.split(",") if n)
    out = folder / f"puzzles-sample-seed{args.seed}.tsv"
    description = datasets.build_puzzle_sample(paths, out, args.total, take_all, args.seed)
    dataset_id, digest = datasets.register_file(store, datasets.SAMPLE_NAME, "1", out, description)
    print(f"{description['rows']} of {description['database_puzzles']} puzzles in {out}")
    print(f"sha256 {digest} (dataset {dataset_id})")
    return 0


def _fetch(args, config: Config) -> int:
    if args.what == "stockfish":
        print(external.fetch_stockfish(config.cache_dir))
    else:
        print(external.fetch_book(config.cache_dir))

    return 0


# ---------------------------------------------------------------------- parser


def _add_limits(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--nodes", type=int)
    group.add_argument("--depth", type=int)
    group.add_argument("--st", type=float, default=0.05, help="seconds per move (default 0.05)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meristats")
    parser.add_argument("--version", action="version", version=f"meristats {__version__}")
    parser.add_argument("--config", type=Path, help="the settings file")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("doctor", help="check the installation")
    commands.add_parser("selftest", help="run the tests of meristats")
    build_parser_ = commands.add_parser("build", help="build the engine at a commit")
    build_parser_.add_argument("ref", help="a commit, a branch or a tag of the engine repository")
    backup_parser = commands.add_parser("backup", help="copy the database and the blobs")
    backup_parser.add_argument("destination", type=Path)
    commands.add_parser("verify", help="check every stored blob against its hash")
    publish_parser = commands.add_parser("publish", help="write the site from the pinned runs")
    publish_parser.add_argument("--out", type=Path, default=Path("site-build"))
    magics_parser = commands.add_parser("magics", help="find the magic numbers of the sliders")
    magics_parser.add_argument("--out", type=Path, help="write the C++ source here")
    serve_parser = commands.add_parser("serve", help="serve the site on this computer")
    serve_parser.add_argument("--dir", type=Path, default=Path("site-build"))
    serve_parser.add_argument("--port", type=int, default=8000)
    pin_parser = commands.add_parser("pin", help="put runs on a page of the site")
    pin_parser.add_argument(
        "run_ids", nargs="+", metavar="run_id", help="ids, or the start of them"
    )
    pin_parser.add_argument(
        "--page", required=True, help="steps, techniques, stockfish, speed, ..."
    )
    pin_parser.add_argument(
        "--label",
        help="the label of the run, for one run only; the default is the config and the limit",
    )
    unpin_parser = commands.add_parser("unpin", help="take a label off a page of the site")
    unpin_parser.add_argument("--page", required=True)
    unpin_parser.add_argument("--label", required=True)
    commands.add_parser("pins", help="list the pinned runs")
    compare_parser = commands.add_parser(
        "compare", help="compare two puzzle runs on the same puzzles"
    )
    compare_parser.add_argument("a", help="the base run, an id or the start of one")
    compare_parser.add_argument("b", nargs="+", help="the runs to compare with the base")
    compare_parser.add_argument(
        "--save",
        action="store_true",
        help="store the comparison as a run, so that it can be pinned",
    )
    compare_parser.add_argument(
        "--pin",
        metavar="PAGE",
        help="store each comparison and pin it to PAGE, labelled with the config of its run",
    )
    compare_parser.add_argument("--allow-dirty", action="store_true")
    dataset_parser = commands.add_parser(
        "dataset", help="build the puzzle sample from the database"
    )
    dataset_parser.add_argument("what", choices=["puzzles"])
    dataset_parser.add_argument("--total", type=int, default=1_000_000)
    dataset_parser.add_argument("--take-all", default="3,4,5", help="mate lengths taken whole")
    dataset_parser.add_argument("--seed", type=int, default=1)
    dataset_parser.add_argument(
        "--reuse",
        type=Path,
        action="append",
        default=[],
        help="a folder that may already hold the parquet files",
    )
    report_parser = commands.add_parser("report", help="print the numbers and tables of a run")
    report_parser.add_argument("run_id", help="a run id, or the start of one")
    runs_parser = commands.add_parser("runs", help="list the runs")
    runs_parser.add_argument("--protocol")
    runs_parser.add_argument("--grade", choices=["verified", "unverified"])
    runs_parser.add_argument("--search", help="only runs whose config contains this text")
    runs_parser.add_argument("--last", type=int, help="only the newest runs")
    fetch_parser = commands.add_parser("fetch", help="download Stockfish or the opening book")
    fetch_parser.add_argument("what", choices=["stockfish", "book"])

    run_parser = commands.add_parser("run", help="run a protocol")
    protocols = run_parser.add_subparsers(dest="protocol", required=True)

    puzzles = protocols.add_parser("puzzles", help="play the solutions of puzzles")
    puzzles.add_argument("--engine", required=True, help="a commit of the engine repository")
    limits = puzzles.add_mutually_exclusive_group()
    limits.add_argument("--nodes", type=int)
    limits.add_argument("--depth", type=int)
    limits.add_argument("--movetime", type=int, help="milliseconds")
    puzzles.add_argument("--engine-config", help="the Config option, such as network+see=0")
    puzzles.add_argument("--eval-file", type=Path, help="a network file, for the option EvalFile")
    puzzles.add_argument("--items", type=int, default=0, help="the first N puzzles")
    puzzles.add_argument("--dataset", type=Path, help="a puzzle sample file")
    puzzles.add_argument("--workers", type=int)
    puzzles.add_argument("--hash", type=int, default=16, dest="hash_mb")
    puzzles.add_argument(
        "--allow-dirty",
        action="store_true",
        help="run from a working tree with changes (grade unverified)",
    )

    nodes = protocols.add_parser("search-nodes", help="search a suite to fixed depths")
    nodes.add_argument("--engine", required=True)
    nodes.add_argument(
        "--configs",
        default="",
        help="Config values separated by commas; empty runs the default of the engine",
    )
    nodes.add_argument("--depths", default="1,2,3,4,5,6")
    nodes.add_argument("--rounds", type=int, default=3)
    nodes.add_argument("--cpu", type=int, help="pin the engine to this logical CPU")
    nodes.add_argument("--suite", type=Path)
    nodes.add_argument("--allow-dirty", action="store_true")

    rules = protocols.add_parser("rules", help="count nodes of reference positions")
    rules.add_argument("--engine", required=True)
    rules.add_argument("--cpu", type=int)
    rules.add_argument("--max-nodes", type=int, default=rules_protocol.MAX_NODES)
    rules.add_argument("--perft", type=Path)
    rules.add_argument("--allow-dirty", action="store_true")

    match = protocols.add_parser("match", help="play two engines against each other")
    match.add_argument("--a", required=True, help="a commit: the engine under test")
    match.add_argument("--b", required=True, help="a commit: the opponent")
    match.add_argument("--a-config")
    match.add_argument("--b-config")
    match.add_argument("--a-eval-file", type=Path, help="a network file for A")
    match.add_argument("--b-eval-file", type=Path, help="a network file for B")
    _add_limits(match)
    match.add_argument(
        "--rounds", type=int, default=match_protocol.ROUNDS, help="the most pairs of games"
    )
    match.add_argument("--no-sprt", action="store_true")
    match.add_argument("--elo0", type=float, default=0.0)
    match.add_argument("--elo1", type=float, default=10.0)
    match.add_argument("--seed", type=int, default=1)
    match.add_argument("--book", type=Path)
    match.add_argument("--no-compliance", action="store_true")
    match.add_argument("--allow-dirty", action="store_true")

    positions = protocols.add_parser("positions", help="play the engine against itself for data")
    positions.add_argument("--engine", required=True)
    positions.add_argument("--out", type=Path, required=True, help="the data file to write")
    positions.add_argument("--games", type=int, default=20000)
    positions.add_argument("--depth", type=int, default=7)
    positions.add_argument("--engine-config", help="the Config option, such as full")
    positions.add_argument("--eval-file", type=Path, help="a network file, for the option EvalFile")
    positions.add_argument("--random-plies", type=int, default=8)
    positions.add_argument("--skip", type=int, default=3, help="keep a position every N plies")
    positions.add_argument("--seed", type=int, default=1)
    positions.add_argument("--workers", type=int)
    positions.add_argument("--allow-dirty", action="store_true")

    train = protocols.add_parser("train", help="train a network from labelled positions")
    train.add_argument("--engine", required=True, help="the engine that has to accept the network")
    train.add_argument("--data", type=Path, required=True, help="positions, as `positions` writes")
    train.add_argument("--out", type=Path, help="the network file to write")
    train.add_argument("--source", type=Path, help="also write the network as C++ source")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch", type=int, default=16384)
    train.add_argument("--lr", type=float, default=0.001)
    train.add_argument("--lambda", type=float, default=0.7, dest="lam")
    train.add_argument("--max-score", type=int, default=2500)
    train.add_argument("--limit", type=int, default=0, help="only the first N positions")
    train.add_argument("--seed", type=int, default=1)
    train.add_argument("--init", type=Path, help="a network file to start from")
    train.add_argument("--device", help="cuda or cpu; the default is cuda when there is one")
    train.add_argument("--workers", type=int, help="processes that read the data")
    train.add_argument("--allow-dirty", action="store_true")

    job = protocols.add_parser("job", help="build and run a training or tuning tool")
    job.add_argument("tool", choices=list(job_protocol.TOOLS))
    job.add_argument("--engine", required=True, help="the commit of the engine to build against")
    job.add_argument(
        "--input",
        type=Path,
        action="append",
        default=[],
        help="an input file (its hash is recorded)",
    )
    job.add_argument(
        "--output",
        action="append",
        default=[],
        help="a file that the tool writes, relative to its folder",
    )
    job.add_argument("--timeout", type=float)
    job.add_argument("--allow-dirty", action="store_true")
    job.add_argument(
        "tool_args",
        nargs=argparse.REMAINDER,
        help="after the tool and --: its arguments (options of this command come before the tool)",
    )

    experiment = protocols.add_parser(
        "experiment", help="test changes to the search, one after the other"
    )
    experiment.add_argument("--engine", required=True)
    experiment.add_argument("--base", default="network")
    experiment.add_argument(
        "--change",
        action="append",
        required=True,
        help="a switch or number to add, such as razoring or see=0; a leading ! tests it "
        "without keeping it",
    )
    experiment.add_argument("--items", type=int, default=200_000, help="puzzles to play")
    experiment.add_argument("--nodes", type=int, default=20_000, help="nodes for each puzzle move")
    experiment.add_argument("--depths", default="5,7", help="depths of the node count")
    experiment.add_argument("--st", type=float, default=0.05, help="seconds per move in the match")
    experiment.add_argument("--rounds", type=int, default=match_protocol.ROUNDS)
    experiment.add_argument("--elo0", type=float, default=0.0)
    experiment.add_argument("--elo1", type=float, default=10.0)
    experiment.add_argument("--seed", type=int, default=1)
    experiment.add_argument("--cpu", type=int)
    experiment.add_argument("--suite", type=Path)
    experiment.add_argument("--book", type=Path)
    experiment.add_argument("--dataset", type=Path)
    experiment.add_argument("--allow-dirty", action="store_true")

    spsa = protocols.add_parser("spsa", help="tune numeric parameters by simultaneous perturbation")
    spsa.add_argument("--engine", required=True)
    spsa.add_argument("--base", required=True, help="a preset with changes, such as network")
    spsa.add_argument("--iterations", type=int, default=120)
    spsa.add_argument("--pairs", type=int, default=24, help="pairs of games in each iteration")
    spsa.add_argument("--st", type=float, default=0.02, help="seconds per move")
    spsa.add_argument("--book", type=Path)
    spsa.add_argument("--allow-dirty", action="store_true")

    ladder = protocols.add_parser("ladder", help="play Stockfish at several strength levels")
    ladder.add_argument("--engine", required=True)
    ladder.add_argument("--engine-config")
    ladder.add_argument("--levels", default="2000,2200,2400,2600,2800,3000")
    _add_limits(ladder)
    ladder.add_argument("--rounds", type=int, default=25, help="pairs of games for each level")
    ladder.add_argument("--seed", type=int, default=1)
    ladder.add_argument("--book", type=Path)
    ladder.add_argument("--allow-dirty", action="store_true")
    return parser


RUNNERS = {
    "positions": _run_positions,
    "train": _run_train,
    "experiment": _run_experiment,
    "spsa": _run_spsa,
    "job": _run_job,
    "puzzles": _run_puzzles,
    "search-nodes": _run_search_nodes,
    "rules": _run_rules,
    "match": _run_match,
    "ladder": _run_ladder,
}


def main(argv: list[str] | None = None) -> int:
    """
    Runs the `meristats` command and returns its exit status.
    """
    args = build_parser().parse_args(argv)

    if args.command == "publish":
        try:
            config = load(args.config)
        except ConfigError:
            config = None

        if config is None:
            written = publish(args.out)
        else:
            with Store(config.data_dir) as store:
                written = publish(args.out, store)

        for path in written:
            print(path)

        pages = sum(1 for path in written if path.suffix == ".html")
        print(f"{pages} pages written to {args.out}")
        return 0

    if args.command == "magics":
        rook, bishop = magics.find_all()
        text = magics.source_text(rook, bishop)

        if args.out:
            args.out.write_text(text)
            print(f"written to {args.out}")
        else:
            print(text, end="")

        return 0

    if args.command == "serve":
        server = make_server(args.dir, args.port)
        print(f"serving {args.dir} at http://127.0.0.1:{server.server_address[1]}/")

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()

        return 0

    if args.command == "doctor":
        return doctor(args.config)

    if args.command == "selftest":
        return selftest()

    try:
        config = load(args.config)

        if args.command == "fetch":
            return _fetch(args, config)

        with Store(config.data_dir) as store:
            if args.command == "build":
                built = build_engine(config, store, args.ref)
                print(f"{built.commit_sha} {built.version_text}")
                print(f"binary  {built.binary}")
                print(f"sha256  {built.binary_sha256}")
                print(f"bench   {built.bench_nodes} nodes")
            elif args.command == "run":
                return RUNNERS[args.protocol](args, config, store)
            elif args.command == "pin":
                if args.label and len(args.run_ids) > 1:
                    raise ReportError("--label names one run, so pin several runs without it")

                for given in args.run_ids:
                    run = find_run(store, given)
                    label = args.label or default_label(run)
                    store.pin(args.page, label, run["run_id"])
                    print(f"{run['run_id']} is on {args.page} as {label}")
            elif args.command == "unpin":
                store.pin(args.page, args.label, None)
                print(f"{args.label} is off {args.page}")
            elif args.command == "compare":
                for other in args.b:
                    if args.save or args.pin:
                        run_id, result = save_comparison(
                            config, store, args.a, other, allow_dirty=args.allow_dirty
                        )
                        print(format_comparison(result))
                        print(f"stored as {run_id}")

                        if args.pin:
                            label = default_label(find_run(store, run_id))
                            store.pin(args.pin, label, run_id)
                            print(f"{run_id} is on {args.pin} as {label}")
                    else:
                        print(format_comparison(compare_puzzles(store, args.a, other)))
            elif args.command == "dataset":
                return _dataset(args, config, store)
            elif args.command == "report":
                print(format_run(store, args.run_id))
            elif args.command == "pins":
                for pin in store.pins():
                    print(f"{pin['page']:<12} {pin['label']:<30} {pin['run_id']}")
            elif args.command == "runs":
                rows = [
                    row
                    for row in store.runs(args.protocol)
                    if (not args.grade or row["grade"] == args.grade)
                    and (not args.search or args.search in (row["config"] or ""))
                ]

                for row in rows[-args.last :] if args.last else rows:
                    print(
                        f"{row['run_id']}  {row['grade']:<10} {row['protocol']:<13} "
                        f"{row['config'] or '':<28} {limit_text(row)}"
                    )
            elif args.command == "backup":
                store.backup(args.destination)
                print(f"copied to {args.destination}")
            elif args.command == "verify":
                bad = store.verify_blobs()
                print(f"{len(bad)} damaged or missing blobs")

                for sha in bad:
                    print(sha)

                return 1 if bad else 0
    except (
        ConfigError,
        BuildError,
        StoreError,
        external.ExternalError,
        ReportError,
        datasets.DatasetError,
        match_protocol.MatchError,
        PreflightError,
        search_nodes_protocol.RoundsDisagree,
        job_protocol.JobError,
        EngineError,
    ) as error:
        print(f"meristats: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
