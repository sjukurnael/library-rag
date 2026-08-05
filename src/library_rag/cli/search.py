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

from library_rag import config, db
from library_rag.pipeline import embed as embed_mod


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=8, help="Number of results (default 8).")
    parser.add_argument(
        "--book", type=int, default=None, help="Restrict search to one books.id."
    )
    parser.add_argument(
        "--mode",
        choices=db.SEARCH_MODES,
        default=config.SEARCH_MODE,
        help=f"Retrieval mode (default {config.SEARCH_MODE}, the mode the app "
        "ships). dense/lexical run one leg alone, which is how you see what the "
        "fusion is actually contributing. Defaulting to anything other than what "
        "api.py uses would make this tool lie about the thing it exists to debug.",
    )
    args = parser.parse_args()

    print(f"Embedding query with {config.EMBED_MODEL} ...")
    voyage_client = embed_mod.build_client()
    query_embedding = embed_mod.embed_query(args.query, voyage_client)

    with db.get_conn() as conn:
        rows = db.search(
            conn, query_embedding, args.k, args.book,
            query_text=args.query, mode=args.mode,
        )

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

        # Which leg found it, when the fused modes ran: "dense 3 / lexical -"
        # reads as "the embedding ranked this 3rd, full text never saw it".
        legs = ""
        if "dense_rank" in r:
            legs = (
                f"  [dense {r['dense_rank'] or '-'} / lexical {r['lexical_rank'] or '-'}]"
            )
        print(
            f"#{rank}  distance={r['distance']:.4f}  {r['title']}  ({page_label}){legs}"
        )
        if r["heading_trail"]:
            print(f"    {r['heading_trail']}")
        print(f"    passage {r['ordinal'] + 1} of {r['total_chunks']}")
        snippet = " ".join(r["content"].strip().split())[:300]
        print(f"    {snippet}...\n")


if __name__ == "__main__":
    main()
