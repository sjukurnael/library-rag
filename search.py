"""
Retrieval smoke test -- the Phase 1 finish line: an eyeball-quality check that
vector search returns sensible passages with citations.

    python search.py "How should I study a Bible chapter?"
    python search.py "structure of the book of Romans" -k 8
    python search.py "..." --book 3

The query is embedded through the SAME pipeline/embed.py function ingest.py uses
for documents (embed_query and embed_documents share _embed / the same model),
so query and document vectors live in one space.
"""
import argparse

import config
import db
from pipeline import embed as embed_mod


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=8, help="Number of results (default 8).")
    parser.add_argument(
        "--book", type=int, default=None, help="Restrict search to one books.id."
    )
    args = parser.parse_args()

    print(f"Embedding query with {config.EMBED_MODEL} ...")
    voyage_client = embed_mod.build_client()
    query_embedding = embed_mod.embed_query(args.query, voyage_client)

    with db.get_conn() as conn:
        rows = db.search(conn, query_embedding, args.k, args.book)

    if not rows:
        print("No results. Is the DB populated? Did ingest.py run?")
        return

    print(f"\nTop {len(rows)} results for: {args.query!r}\n")
    for rank, r in enumerate(rows, start=1):
        page_start, page_end = r["page_start"], r["page_end"]
        if page_start is None:
            page_label = "p.?"
        elif page_start == page_end:
            page_label = f"p.{page_start}"
        else:
            page_label = f"pp.{page_start}-{page_end}"

        print(f"#{rank}  distance={r['distance']:.4f}  {r['title']}  ({page_label})")
        if r["heading_trail"]:
            print(f"    {r['heading_trail']}")
        print(f"    passage {r['ordinal'] + 1} of {r['total_chunks']}")
        snippet = " ".join(r["content"].strip().split())[:300]
        print(f"    {snippet}...\n")


if __name__ == "__main__":
    main()
