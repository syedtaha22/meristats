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

import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from meristats.config import Config
from meristats.store import Store

FAKE_ENGINE = """
import sys, time
mode = sys.argv[1] if len(sys.argv) > 1 else "normal"
print_flush = lambda s: (sys.stdout.write(s + "\\n"), sys.stdout.flush())
config = "network"
for line in sys.stdin:
    words = line.split()
    if not words:
        continue
    cmd = words[0]
    if cmd == "uci":
        print_flush("id name Fake 1.0")
        print_flush("id author test")
        print_flush("option name Hash type spin default 16 min 1 max 1024")
        print_flush("option name Threads type spin default 1 min 1 max 64")
        print_flush("option name OwnBook type check default false")
        print_flush("option name Config type string default network")
        print_flush("option name EvalFile type string default <empty>")
        print_flush("option name Clear Hash type button")
        if mode == "no_uciok":
            continue
        print_flush("uciok")
        if mode == "exit_after_uci":
            sys.exit(0)
    elif cmd == "isready":
        print_flush("readyok")
    elif cmd == "setoption":
        if "Config" in words and "bad" in words:
            print_flush("info string invalid Config: unknown preset 'bad'")
        elif "Config" in words:
            print_flush("info string Config: " + words[-1])
    elif cmd == "go" and len(words) > 2 and words[1] == "perft":
        print_flush("Nodes searched: " + ("x" if mode == "badperft" else "8902"))
    elif cmd == "go":
        if mode == "silent":
            continue
        if mode == "crash":
            sys.exit(3)
        if mode == "late":
            pending = True
            continue
        print_flush("info depth 1 score cp 10 nodes 21 time 1 pv e2e4 e7e5")
        print_flush("info depth 2 score mate 3 nodes 90 time 2 pv e2e4")
        print_flush("info string hello")
        print_flush("bestmove e2e4 ponder e7e5")
    elif cmd == "stop":
        if mode == "late":
            print_flush("bestmove e2e4")
    elif cmd == "quit":
        if mode == "stubborn":
            continue
        break
"""


@pytest.fixture
def fake_engine(tmp_path):
    """
    Returns a function that writes the fake UCI engine and returns its path and arguments.
    """
    script = tmp_path / "fake_engine.py"
    script.write_text(FAKE_ENGINE)
    wrapper = tmp_path / "fake_engine"
    wrapper.write_text(f"#!{sys.executable}\n" + FAKE_ENGINE)
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)

    def make(mode: str = "normal"):
        return wrapper, (mode,)

    return make


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "data") as opened:
        yield opened


MAIN_CPP = textwrap.dedent("""
    #include <cstdlib>
    #include <iostream>
    #include <string>

    int main(int argc, char **argv) {
        if (argc > 1 && std::string(argv[1]) == "bench") {
            std::cout << "Total time (ms) : 5\\nNodes searched  : NODES\\n"
                         "Nodes/second    : 4000000\\n";
            return 0;
        }
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line == "uci") {
                std::cout << "id name Tiny VERSION\\nid author t\\nuciok\\n" << std::flush;
            } else if (line == "isready") {
                std::cout << "readyok\\n" << std::flush;
            } else if (line == "quit") {
                return 0;
            }
        }
        return 0;
    }
""")

CMAKE = textwrap.dedent("""
    cmake_minimum_required(VERSION 3.16)
    project(tiny CXX)
    add_executable(meridian main.cpp)
""")


def run_git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=t@example.com", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


@pytest.fixture
def tiny_repo(tmp_path):
    """
    A git repository of a tiny engine: a CMake project whose `meridian` answers `uci` and `bench`.
    Returns the path and a function that commits a new node count.
    """
    repo = tmp_path / "engine"
    repo.mkdir()
    run_git(repo, "init", "-q", "-b", "main")
    (repo / "CMakeLists.txt").write_text(CMAKE)

    def commit(nodes: int, version: str = "1.0") -> str:
        (repo / "main.cpp").write_text(
            MAIN_CPP.replace("NODES", str(nodes)).replace("VERSION", version)
        )
        run_git(repo, "add", "-A")
        run_git(repo, "commit", "-q", "-m", f"nodes {nodes}")
        return run_git(repo, "rev-parse", "HEAD")

    return repo, commit


@pytest.fixture
def config(tmp_path, tiny_repo):
    repo, _ = tiny_repo
    return Config(
        engine_repo=repo, data_dir=tmp_path / "data", cache_dir=tmp_path / "cache", workers=2
    )


ORACLE = """
import json, os, sys
answers = json.load(open(os.environ["ORACLE_ANSWERS"]))
if len(sys.argv) > 1 and sys.argv[1] == "bench":
    print("Nodes searched  : 7")
    print("Nodes/second    : 1000")
    sys.exit(0)
position = ""
config = "network"
counters = {}
def say(text):
    sys.stdout.write(text + "\\n"); sys.stdout.flush()
for line in sys.stdin:
    words = line.split()
    if not words:
        continue
    if words[0] == "uci":
        say("id name Oracle 1")
        say("option name Hash type spin default 16 min 1 max 1024")
        say("option name Threads type spin default 1 min 1 max 64")
        say("option name MoveOverhead type spin default 30 min 0 max 5000")
        say("option name Config type string default network")
        say("uciok")
    elif words[0] == "isready":
        say("readyok")
    elif words[0] == "setoption":
        if "Config" in words:
            say("info string Config: " + words[-1])
    elif words[0] == "position":
        position = line.strip()
    elif words[0] == "go" and len(words) > 2 and words[1] == "perft":
        value = answers.get("perft " + position + " " + words[2])
        say("Nodes searched: %d" % (value if value is not None else 0))
    elif words[0] == "go":
        answer = answers.get(position)
        if isinstance(answer, list):
            n = counters.get(position, 0)
            counters[position] = n + 1
            answer = answer[n % len(answer)]
        if answer is None:
            say("info string no answer for " + position)
            say("bestmove 0000")
        elif answer == "crash":
            sys.exit(5)
        else:
            for depth, nodes, move in answer.get("infos", []):
                say("info depth %d score cp 5 nodes %d time 1 pv %s" % (depth, nodes, move))
            say("bestmove " + answer["best"])
    elif words[0] == "quit":
        break
"""


@pytest.fixture
def oracle(tmp_path, monkeypatch):
    """
    Returns a function that takes the answers of an oracle engine (a map from the `position`
    command to the best move and the info lines) and returns the path of the program.
    """
    program = tmp_path / "oracle"
    program.write_text(f"#!{sys.executable}\n" + ORACLE)
    program.chmod(program.stat().st_mode | stat.S_IEXEC)

    def make(answers: dict) -> Path:
        import json

        file = tmp_path / "answers.json"
        file.write_text(json.dumps(answers))
        monkeypatch.setenv("ORACLE_ANSWERS", str(file))
        return program

    return make


@pytest.fixture
def framework_repo(tmp_path):
    """
    A clean git repository that stands for the meristats checkout.
    """
    repo = tmp_path / "framework"
    repo.mkdir()
    run_git(repo, "init", "-q", "-b", "main")
    (repo / "file").write_text("x")
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-q", "-m", "m")
    return repo


FAKE_FASTCHESS = """
import os, sys
args = sys.argv[1:]
if args and args[0] == "--compliance":
    print("Engine passed all compliance checks." if os.environ.get("FAKE_COMPLIANCE", "ok") == "ok"
          else "Step 3 failed")
    sys.exit(0)
pgn = args[args.index("-pgnout") + 1][5:]
results = os.environ["FAKE_RESULTS"].split(",")
with open(os.environ.get("FAKE_ARGS_FILE", os.devnull), "w") as f:
    f.write(" ".join(args))
with open("config.json", "w") as f:
    f.write("{}")
with open(pgn, "w") as out:
    for i, result in enumerate(results):
        white, black = ("A", "B") if i % 2 == 0 else ("B", "A")
        out.write('[Event "Fake"]\\n[Round "%d"]\\n[White "%s"]\\n[Black "%s"]\\n[Result "%s"]\\n'
                  '[Termination "%s"]\\n[PlyCount "4"]\\n[Opening "Opening %d"]\\n\\n'
                  '1. e4 e5 2. Nf3 Nc6 %s\\n\\n' % (i // 2 + 1, white, black, result,
                  "time forfeit" if os.environ.get("FAKE_FORFEIT") and i == 0 else "normal",
                  i // 2 + 1, result))
print("Results of A vs B")
print("Games: %d" % len(results))
if os.environ.get("FAKE_VERDICT"):
    print("SPRT ([0.00, 10.00]) completed - %s was accepted" % os.environ["FAKE_VERDICT"])
print("Finished match")
sys.exit(int(os.environ.get("FAKE_EXIT", "0")))
"""


@pytest.fixture
def fake_fastchess(tmp_path, monkeypatch):
    """
    Returns a function that sets the results of the fake fastchess and returns its path.
    """
    program = tmp_path / "fastchess"
    program.write_text(f"#!{sys.executable}\n" + FAKE_FASTCHESS)
    program.chmod(program.stat().st_mode | stat.S_IEXEC)

    def make(results, verdict=None, **extra):
        monkeypatch.setenv("FAKE_RESULTS", ",".join(results))

        for key, value in {"FAKE_VERDICT": verdict, **extra}.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, str(value))

        monkeypatch.setenv("FAKE_ARGS_FILE", str(tmp_path / "args.txt"))
        return program

    return make


@pytest.fixture
def config_file(tmp_path, tiny_repo):
    """
    A settings file for the command line, with its own data folder.
    """
    repo, _ = tiny_repo
    file = tmp_path / "meristats.toml"
    file.write_text(
        f'engine_repo = "{repo}"\ndata_dir = "{tmp_path / "cli-data"}"\n'
        f'cache_dir = "{tmp_path / "cli-cache"}"\nworkers = 2\n'
    )
    return file


TOOL_CMAKE = textwrap.dedent("""
    cmake_minimum_required(VERSION 3.20)
    project(tiny CXX)
    set(WARNING_FLAGS -Wall)
    find_package(Threads REQUIRED)
    add_library(chess_core STATIC core.cpp)
    target_include_directories(chess_core PUBLIC include)
    target_link_libraries(chess_core PUBLIC Threads::Threads)
    add_executable(meridian main.cpp)
""")


@pytest.fixture
def tool_repo(tmp_path):
    """
    A tiny engine repository whose CMake project has a library `chess_core`, like the real
    engine, and a folder of tool sources that use it. Returns the repository, a function that
    commits it, and the tools folder.
    """
    repo = tmp_path / "toolengine"
    (repo / "include").mkdir(parents=True)
    run_git(repo, "init", "-q", "-b", "main")
    (repo / "CMakeLists.txt").write_text(TOOL_CMAKE)
    (repo / "include" / "core.hpp").write_text("int core_value();\n")
    (repo / "core.cpp").write_text('#include "core.hpp"\nint core_value() { return 42; }\n')
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "build_book.cpp").write_text(
        textwrap.dedent("""
        #include <fstream>
        #include <iostream>
        #include <string>
        #include "core.hpp"

        int main(int argc, char **argv) {
            std::string out = "out.txt";
            int code = 0;
            for (int i = 1; i + 1 < argc; ++i) {
                if (std::string(argv[i]) == "--out") out = argv[i + 1];
                if (std::string(argv[i]) == "--exit") code = std::stoi(argv[i + 1]);
            }
            std::ifstream in(argc > 2 ? argv[2] : "/dev/null");
            std::string line;
            std::getline(in, line);
            if (out != "none") {
                std::ofstream file(out);
                file << "core " << core_value() << " first line [" << line << "]\\n";
            }
            std::cout << "tool ran\\n";
            return code;
        }
    """)
    )

    def commit(nodes: int = 100) -> str:
        (repo / "main.cpp").write_text(
            MAIN_CPP.replace("NODES", str(nodes)).replace("VERSION", "1.0")
        )
        run_git(repo, "add", "-A")
        run_git(repo, "commit", "-q", "-m", "tool engine")
        return run_git(repo, "rev-parse", "HEAD")

    return repo, commit, tools
