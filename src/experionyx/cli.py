"""Minimal command line interface: reports real package/environment facts only."""

import argparse
import platform
import sys
from collections.abc import Sequence

from experionyx import __version__


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="experionyx", description="EXPERIONYX: AI Experimental Forensics & Reliability Lab"
    )
    parser.add_argument("--version", action="version", version=f"experionyx {__version__}")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("info", help="print package and environment information")
    args = parser.parse_args(argv)

    if args.command == "info":
        print(f"experionyx {__version__}")
        print(f"python {platform.python_version()} ({platform.python_implementation()})")
        print(f"platform {platform.platform()}")
        print("status: Phase 0 foundation; no laboratory features implemented yet")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
