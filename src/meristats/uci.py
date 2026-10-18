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

import os
import select
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


QUIT_TIMEOUT = 5.0


class EngineError(Exception):
    """
    An engine did not behave as the protocol says.
    """


class EngineTimeout(EngineError):
    """
    An engine did not answer in time.
    """


class EngineCrashed(EngineError):
    """
    An engine ended while a reply was expected.
    """


@dataclass
class Info:
    """
    The fields of one `info` line that meristats uses.
    """

    depth: int | None = None
    nodes: int | None = None
    score_kind: str | None = None
    score: int | None = None
    time_ms: int | None = None
    pv: list[str] = field(default_factory=list)


@dataclass
class GoResult:
    bestmove: str
    infos: list[Info]
    strings: list[str]
    elapsed: float


def parse_info(line: str) -> Info | None:
    """
    Parses an `info` line, and returns None for a line that is not one or that has no depth.
    """
    words = line.split()

    if not words or words[0] != "info" or "string" in words[:2]:
        return None

    info = Info()
    index = 1

    while index < len(words):
        key = words[index]

        if key in ("depth", "nodes", "time") and index + 1 < len(words):
            try:
                number = int(words[index + 1])
            except ValueError:
                return None

            if key == "depth":
                info.depth = number
            elif key == "nodes":
                info.nodes = number
            else:
                info.time_ms = number

            index += 2
        elif key == "score" and index + 2 < len(words):
            try:
                info.score_kind, info.score = words[index + 1], int(words[index + 2])
            except ValueError:
                return None

            index += 3

            while index < len(words) and words[index] in ("lowerbound", "upperbound"):
                index += 1
        elif key == "pv":
            info.pv = words[index + 1 :]
            break
        else:
            index += 1

    return info if info.depth is not None else None


class UciEngine:
    """
    One engine program that speaks UCI, driven through its standard input and output.

    Every reply is read with a limit on the time. An engine that stays silent raises
    `EngineTimeout`, and one that ends raises `EngineCrashed`.
    """

    def __init__(
        self,
        path: Path,
        args: tuple[str, ...] = (),
        startup_timeout: float = 10.0,
        cpu: int | None = None,
    ):
        self.path = Path(path)
        self.args = args
        self.startup_timeout = startup_timeout
        self.cpu = cpu
        self.process: subprocess.Popen | None = None
        self.options: dict[str, dict] = {}
        self.name = ""
        self._buffer = b""

    # ------------------------------------------------------------------ plumbing

    def start(self) -> "UciEngine":
        """
        Starts the program and completes the `uci` handshake.
        """
        pin = (lambda: os.sched_setaffinity(0, {self.cpu})) if self.cpu is not None else None
        self.process = subprocess.Popen(
            [str(self.path), *self.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
            preexec_fn=pin,
        )
        self.send("uci")

        for line in self.read_until(lambda text: text == "uciok", self.startup_timeout):
            if line.startswith("id name "):
                self.name = line[len("id name ") :]
            elif line.startswith("option name "):
                self._parse_option(line)

        return self

    def _parse_option(self, line: str) -> None:
        body = line[len("option name ") :]
        name, _, rest = body.partition(" type ")
        kind, _, tail = rest.partition(" ")
        default = None

        if " default " in " " + tail:
            default = (" " + tail).split(" default ", 1)[1].split(" min ")[0].split(" var ")[0]
            default = default.strip()

        self.options[name] = {"type": kind, "default": default}

    def send(self, line: str) -> None:
        if self.process is None or self.process.poll() is not None:
            raise EngineCrashed(f"{self.path.name} is not running")

        try:
            self.process.stdin.write((line + "\n").encode())
        except (BrokenPipeError, OSError) as error:
            raise EngineCrashed(f"{self.path.name} closed its input") from error

    def _next_line(self, deadline: float) -> str:
        fd = self.process.stdout.fileno()

        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()

            if remaining <= 0:
                raise EngineTimeout(f"{self.path.name} did not answer in time")

            ready, _, _ = select.select([fd], [], [], min(remaining, 0.25))

            if ready:
                chunk = os.read(fd, 65536)

                if not chunk:
                    raise EngineCrashed(f"{self.path.name} ended")

                self._buffer += chunk
            elif self.process.poll() is not None:
                raise EngineCrashed(f"{self.path.name} ended")

        line, _, self._buffer = self._buffer.partition(b"\n")
        return line.decode("utf-8", "replace").strip()

    def read_until(self, stop, timeout: float) -> list[str]:
        """
        Reads lines until `stop(line)` is true for one, and returns all lines read, that one
        included.
        """
        deadline = time.monotonic() + timeout
        lines = []

        while True:
            line = self._next_line(deadline)
            lines.append(line)

            if stop(line):
                return lines

    def ready(self, timeout: float = 10.0) -> list[str]:
        """
        Sends `isready`, waits for `readyok`, and returns the lines before it.
        """
        self.send("isready")
        return self.read_until(lambda text: text == "readyok", timeout)[:-1]

    # ---------------------------------------------------------------- commands

    def set_option(self, name: str, value: str | None = None, timeout: float = 30.0) -> list[str]:
        """
        Sets an option and returns the `info string` lines that the engine printed in answer.
        """
        if name not in self.options:
            raise EngineError(f"{self.name or self.path.name} has no option {name}")

        self.send(f"setoption name {name}" + ("" if value is None else f" value {value}"))
        return [
            line[len("info string ") :]
            for line in self.ready(timeout)
            if line.startswith("info string ")
        ]

    def new_game(self) -> None:
        self.send("ucinewgame")
        self.ready()

    def position(self, fen: str | None, moves: tuple[str, ...] | list[str] = ()) -> None:
        base = "startpos" if fen is None else f"fen {fen}"
        self.send(f"position {base}" + (" moves " + " ".join(moves) if moves else ""))

    def go(self, command: str, timeout: float = 30.0) -> GoResult:
        """
        Sends `go <command>`, and returns the best move with the `info` lines. An engine that has
        not answered by `timeout` is told to stop, given 2 seconds, and then reported as timed out.
        """
        start = time.monotonic()
        self.send(f"go {command}")
        deadline = start + timeout
        infos, strings = [], []

        while True:
            try:
                line = self._next_line(deadline)
            except EngineTimeout:
                self.send("stop")

                try:
                    line = self._next_line(time.monotonic() + 2.0)
                except EngineTimeout as error:
                    raise EngineTimeout(f"no bestmove after go {command}") from error

                if not line.startswith("bestmove"):
                    continue

            if line.startswith("bestmove"):
                words = line.split()
                move = words[1] if len(words) > 1 else ""
                return GoResult(move, infos, strings, time.monotonic() - start)

            info = parse_info(line)

            if info is not None:
                infos.append(info)
            elif line.startswith("info string "):
                strings.append(line[len("info string ") :])

    def perft(self, depth: int, timeout: float = 600.0) -> tuple[int, float]:
        """
        Runs `go perft <depth>` on the current position and returns the node count and the time
        that the command took. The command cannot be interrupted, so the timeout is generous.
        """
        start = time.monotonic()
        self.send(f"go perft {depth}")
        lines = self.read_until(lambda text: text.startswith("Nodes searched"), timeout)
        elapsed = time.monotonic() - start

        try:
            return int(lines[-1].split(":", 1)[1]), elapsed
        except (IndexError, ValueError) as error:
            raise EngineError(f"unreadable perft reply: {lines[-1]}") from error

    def quit(self) -> None:
        """
        Asks the program to end, and ends it by force when it does not.
        """
        if self.process is None:
            return

        try:
            if self.process.poll() is None:
                self.send("quit")
                self.process.wait(timeout=QUIT_TIMEOUT)
        except (EngineError, subprocess.TimeoutExpired):
            self.process.kill()
            self.process.wait()
        finally:
            for stream in (self.process.stdin, self.process.stdout):
                try:
                    stream.close()
                except OSError:
                    pass

            self.process = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.quit()
