"""
Apply pending schema migrations.

    python -m library_rag.cli.migrate
"""
import sys

from library_rag import migrate


def main() -> int:
    applied = migrate.run()
    print(f"Applied {len(applied)} migration(s)." if applied else "Schema up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
