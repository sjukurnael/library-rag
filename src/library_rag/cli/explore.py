"""
Drive Exploration Agent CLI entry point.

The actual tool-use loop and system prompt live in agent/loop.py, the tools
in agent/tools.py, the estimation constants in agent/assumptions.py, and
Drive auth/listing in drive/client.py. This file is just argument parsing.

Run:
    python explore.py            # uses cache.json if present
    python explore.py --fresh    # bypasses cache, re-crawls Drive
    python explore.py --quiet    # suppress the [agent] tool-call trail
"""
import argparse
import sys

from dotenv import load_dotenv

# Load before importing agent so ANTHROPIC_MODEL (read at import time) is set.
load_dotenv()

from library_rag.drive.client import DriveAuthError  # noqa: E402
from library_rag.exploration.loop import run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fresh", action="store_true", help="Bypass cache.json and re-crawl Drive."
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress the [agent] tool-call trail."
    )
    args = parser.parse_args()

    try:
        sys.exit(run(fresh=args.fresh, verbose=not args.quiet))
    except DriveAuthError as e:
        print(f"\nDrive auth error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
