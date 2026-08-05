"""
Calibration report: measured pilot numbers vs. the estimates the exploration
agent's agent/assumptions.py made, plus suggested constant corrections. Run
after ingest.py has processed the pilot folder.

    python report.py

The whole point of the pilot: replace the guessed constants in
agent/assumptions.py with numbers measured on real books.
"""
from library_rag import config, db
from library_rag.exploration import assumptions as assumed
from library_rag.pipeline import extract as extract_mod

BYTES_PER_MB = 1024 * 1024


def main():
    with db.get_conn() as conn:
        rows = db.fetch_report_data(conn)

    if not rows:
        print("No books in the database yet -- run ingest.py first.")
        return

    total_mb = total_pages = total_chunks = total_tokens = 0.0
    scanned_mb = scanned_pages = 0.0
    digital_mb = digital_pages = 0.0
    status_counts = {}
    stage_totals = {"download": 0.0, "extract": 0.0, "chunk_embed": 0.0, "total": 0.0}

    print(
        f"{'id':>3}  {'title':<32} {'status':<10} {'MB':>7} {'pages':>6} "
        f"{'scan':>5} {'chunks':>7} {'tokens':>8} {'sec':>7}"
    )
    print("-" * 100)

    for book_id, title, size_bytes, pages, has_text_layer, status, chunk_count, tokens in rows:
        mb = (size_bytes or 0) / BYTES_PER_MB
        pages = pages or 0
        scanned = has_text_layer is False
        status_counts[status] = status_counts.get(status, 0) + 1

        total_mb += mb
        total_pages += pages
        total_chunks += chunk_count
        total_tokens += tokens
        if scanned:
            scanned_mb += mb
            scanned_pages += pages
        elif has_text_layer:
            digital_mb += mb
            digital_pages += pages

        manifest = extract_mod.read_manifest(book_id)
        book_total_s = manifest.get("total_seconds", 0.0) or 0.0
        stage_totals["download"] += manifest.get("download_seconds", 0.0) or 0.0
        stage_totals["extract"] += manifest.get("extract_seconds", 0.0) or 0.0
        stage_totals["chunk_embed"] += manifest.get("chunk_embed_seconds", 0.0) or 0.0
        stage_totals["total"] += book_total_s

        title_short = (title or "")[:32]
        print(
            f"{book_id:>3}  {title_short:<32} {status:<10} {mb:7.1f} {pages:6d} "
            f"{str(scanned):>5} {chunk_count:7d} {tokens:8d} {book_total_s:7.1f}"
        )

    print("-" * 100)
    status_label = ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items()))
    print(f"TOTAL  books={len(rows)}  ({status_label})")
    print(
        f"       MB={total_mb:.1f}  pages={int(total_pages)}  chunks={int(total_chunks)}  "
        f"tokens={int(total_tokens)}"
    )
    print(
        f"       wall-clock: download={stage_totals['download']:.1f}s  "
        f"extract={stage_totals['extract']:.1f}s  "
        f"chunk+embed={stage_totals['chunk_embed']:.1f}s  "
        f"total={stage_totals['total']:.1f}s"
    )

    embed_cost = total_tokens / 1_000_000 * config.EMBED_COST_PER_MTOK
    print(
        f"       measured embedding cost: ${embed_cost:.4f}  "
        f"({config.EMBED_MODEL} @ ${config.EMBED_COST_PER_MTOK}/MTok)"
    )

    measured_mb_per_digital_page = digital_mb / digital_pages if digital_pages else None
    measured_mb_per_scanned_page = scanned_mb / scanned_pages if scanned_pages else None
    measured_tokens_per_page = total_tokens / total_pages if total_pages else None
    measured_chunk_tokens = total_tokens / total_chunks if total_chunks else None

    print("\n--- measured vs. agent/assumptions.py ---")
    if measured_mb_per_digital_page is not None:
        print(
            f"MB_PER_DIGITAL_PAGE:  assumed {assumed.MB_PER_DIGITAL_PAGE}  "
            f"-> measured {measured_mb_per_digital_page:.4f}"
        )
    if measured_mb_per_scanned_page is not None:
        print(
            f"MB_PER_SCANNED_PAGE:  assumed {assumed.MB_PER_SCANNED_PAGE}  "
            f"-> measured {measured_mb_per_scanned_page:.4f}"
        )
    if measured_tokens_per_page is not None:
        print(
            f"TOKENS_PER_PAGE:      assumed {assumed.TOKENS_PER_PAGE}  "
            f"-> measured ~{measured_tokens_per_page:.0f}"
        )
    if measured_chunk_tokens is not None:
        print(
            f"CHUNK_TOKENS:         assumed {assumed.CHUNK_TOKENS}  "
            f"-> measured ~{measured_chunk_tokens:.0f}"
        )

    print("\nSuggested agent/assumptions.py edits:")
    n_edits = 0
    if measured_mb_per_digital_page is not None:
        print(
            f"  MB_PER_DIGITAL_PAGE = {measured_mb_per_digital_page:.4f}  "
            f"# was {assumed.MB_PER_DIGITAL_PAGE} -- measured on the "
            f"{config.PILOT_FOLDER_ID} pilot"
        )
        n_edits += 1
    if measured_mb_per_scanned_page is not None:
        print(
            f"  MB_PER_SCANNED_PAGE = {measured_mb_per_scanned_page:.4f}  "
            f"# was {assumed.MB_PER_SCANNED_PAGE}"
        )
        n_edits += 1
    if measured_tokens_per_page is not None:
        lo, hi = measured_tokens_per_page * 0.9, measured_tokens_per_page * 1.1
        print(
            f"  TOKENS_PER_PAGE = ({lo:.0f}, {hi:.0f})  # was {assumed.TOKENS_PER_PAGE}"
        )
        n_edits += 1
    if measured_chunk_tokens is not None:
        print(
            f"  CHUNK_TOKENS = {measured_chunk_tokens:.0f}  # was {assumed.CHUNK_TOKENS}"
        )
        n_edits += 1
    if n_edits == 0:
        print("  (no books with page counts yet -- nothing to calibrate)")

    needs_ocr = status_counts.get("needs_ocr", 0)
    if needs_ocr:
        print(
            f"\nNote: {needs_ocr} book(s) are needs_ocr (no MISTRAL_API_KEY set) "
            "and are excluded from the scanned-page constants above -- their MB "
            "is counted in totals but not in MB_PER_SCANNED_PAGE."
        )


if __name__ == "__main__":
    main()
