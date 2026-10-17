import argparse
import sys
from pathlib import Path

from meristats import __version__
from meristats.publish import publish


def main(argv: list[str] | None = None) -> int:
    """
    Runs the `meristats` command and returns its exit status.
    """
    parser = argparse.ArgumentParser(prog="meristats")
    parser.add_argument("--version", action="version", version=f"meristats {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    publish_parser = commands.add_parser("publish", help="write the site")
    publish_parser.add_argument("--out", type=Path, default=Path("site-build"))

    args = parser.parse_args(argv)

    if args.command == "publish":
        for path in publish(args.out):
            print(path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
