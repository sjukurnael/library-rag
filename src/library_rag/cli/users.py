"""
Who may sign in to the app.

    python -m library_rag.cli.users --list
    python -m library_rag.cli.users --add friend@gmail.com --note "borrowed Philemon"
    python -m library_rag.cli.users --remove friend@gmail.com

Google proves who someone is; this list decides whether they get in. An empty
list locks everyone out, including you -- which is the right default for a table
that grants access, and why this command exists: it is the bootstrap.

The same rows are editable in the Supabase dashboard's table editor. This is for
local development, where there is no dashboard, and for scripting.
"""
import argparse
import sys

from library_rag import db
from library_rag.web import auth


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--list", action="store_true", help="Show who has access.")
    parser.add_argument("--add", metavar="EMAIL", help="Grant access to an address.")
    parser.add_argument("--remove", metavar="EMAIL", help="Revoke access.")
    parser.add_argument("--note", help="With --add: a reminder of who this is.")
    args = parser.parse_args()

    with db.get_conn() as conn:
        if args.add:
            added = auth.add_user(conn, args.add, args.note)
            print(f"{'Added' if added else 'Already had'} {args.add.strip().lower()}.")
            return 0

        if args.remove:
            removed = auth.remove_user(conn, args.remove)
            if not removed:
                print(f"{args.remove.strip().lower()} was not on the list.")
                return 1
            # Removal takes effect at the next sign-in, not immediately: an
            # existing session cookie is self-contained and valid until it
            # expires. Rotating SESSION_SECRET is what kicks everyone out now.
            print(f"Removed {args.remove.strip().lower()}. Any current session "
                  f"lasts until it expires.")
            return 0

        rows = auth.list_users(conn)

    if not rows:
        print("Nobody has access. Sign-in will reject everyone -- including you.\n"
              "  python -m library_rag.cli.users --add you@example.com")
        return 0
    print(f"{len(rows)} allowed:")
    for email, note, added_at in rows:
        when = added_at.strftime("%Y-%m-%d")
        print(f"  {email:<34}{when}  {note or ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
