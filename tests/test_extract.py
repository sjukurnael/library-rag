"""Extraction probe on generated PDFs (real text vs image-only), plus the
worker-level guarantee that a corrupt PDF is recorded as failed and does not
stop the run. PDFs are built with PyMuPDF at test time -- no fixtures on disk,
no network."""
import re
from pathlib import Path

import fitz
import pytest

from library_rag import config, db, ingest, storage
from library_rag.pipeline import chunking, embed, extract


def _make_text_pdf(dest, pages=3):
    """A digital-native PDF whose text is wrapped inside the page body.

    insert_textbox, not insert_text: the latter lays one unwrapped line that
    runs off the right edge, and pymupdf4llm's layout model discards text
    outside the page body -- yielding an empty extraction. Real books wrap
    their text, so the textbox is the honest fixture. (The off-page case is
    real but rare; the zero-chunk guard in ingest.py is what catches it.)
    """
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_textbox(
            fitz.Rect(72, 90, 520, 700),
            "Inductive Bible study: observation, interpretation, application. " * 4,
            fontsize=11,
        )
    doc.save(str(dest))
    doc.close()


def _make_image_pdf(dest, pages=2):
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 300, 300))
        pix.clear_with(200)
        page.insert_image(page.rect, pixmap=pix)
    doc.save(str(dest))
    doc.close()


def test_text_pdf_probes_true_with_correct_page_count(tmp_path):
    pdf = tmp_path / "text.pdf"
    _make_text_pdf(pdf, pages=3)
    result = extract.extract(pdf, book_id=1)
    assert result.has_text_layer is True
    assert result.page_count == 3
    assert result.extractor == "pymupdf4llm"
    assert "<!-- page: 1 -->" in result.markdown


def test_image_pdf_probes_false_and_needs_ocr(tmp_path):
    pdf = tmp_path / "image.pdf"
    _make_image_pdf(pdf, pages=2)

    doc = fitz.open(pdf)
    try:
        assert extract.probe_text_layer(doc) is False
        assert doc.page_count == 2
    finally:
        doc.close()

    with pytest.raises(extract.NeedsOCRError):
        extract.extract(pdf, book_id=2, ocr_client=None)


def _make_uniform_font_pdf(dest, pages=3):
    """A PDF whose every glyph is the same size and font -- what a scanned page
    looks like after OCR writes a synthetic text layer. Structure is visible
    only from position and whitespace, never from font metadata."""
    body = (
        "Inductive study proceeds by observation, interpretation, and application. "
        "The reader asks what the text says before asking what it means. "
    ) * 5
    doc = fitz.open()
    for lesson in range(1, pages + 1):
        page = doc.new_page()
        page.insert_text((72, 90), f"Lesson {lesson}", fontsize=11, fontname="helv")
        page.insert_text((72, 130), "I. BACKGROUND", fontsize=11, fontname="helv")
        page.insert_textbox(fitz.Rect(72, 150, 520, 380), body, fontsize=11)
        page.insert_text((72, 410), "II. ANALYSIS", fontsize=11, fontname="helv")
        page.insert_textbox(fitz.Rect(72, 430, 520, 700), body, fontsize=11)
    doc.save(str(dest))
    doc.close()


def test_uniform_font_pdf_still_yields_headings(tmp_path):
    """Guards pymupdf4llm.use_layout(True) in pipeline/extract.py.

    The classic font-statistics path infers heading levels from a document-wide
    size histogram, so a uniform-font page (an OCR'd text layer) flattens it to
    zero headings -- and 77% of a real Jensen guide got wrapped in ``` fences,
    which MarkdownHeaderTextSplitter correctly refuses to split on. Chunking
    then saw 2 sections in 120 pages. The layout model segments visually and is
    unaffected.

    Measured on this fixture: use_layout(True) -> 6 headings, (False) -> 0.
    """
    pdf = tmp_path / "uniform.pdf"
    _make_uniform_font_pdf(pdf)

    result = extract.extract(pdf, book_id=1)

    headings = re.findall(r"^#{1,6} .*$", result.markdown, re.MULTILINE)
    assert headings, (
        "no markdown headings from a uniform-font PDF -- pipeline/extract.py has "
        "probably reverted to use_layout(False); see the comment there"
    )
    assert any("BACKGROUND" in h for h in headings)
    # Fenced content is invisible to the header splitter, so it must stay rare.
    assert "```" not in result.markdown


def test_extraction_output_is_chunkable_end_to_end(tmp_path):
    """Extraction and chunking must agree: whatever heading levels the extractor
    emits, config.MARKDOWN_HEADERS has to register them, or they stay in the
    body and reach the embedding model as literal '######'."""
    pdf = tmp_path / "uniform.pdf"
    _make_uniform_font_pdf(pdf)
    result = extract.extract(pdf, book_id=1)

    emitted = {len(m) for m in re.findall(r"^(#{1,6}) ", result.markdown, re.MULTILINE)}
    registered = {len(prefix) for prefix, _ in config.MARKDOWN_HEADERS}
    assert emitted <= registered, (
        f"extractor emits h{sorted(emitted)} but config registers h{sorted(registered)}"
    )

    chunks = chunking.chunk_markdown(result.markdown)
    assert chunks
    assert any(c["heading_trail"] for c in chunks), "no chunk carries a heading trail"
    for c in chunks:
        body = c["content"].split("\n\n", 1)[-1]
        assert "#" not in body, f"heading markup leaked into embedded text: {body[:80]!r}"


def test_a_book_that_extracts_to_nothing_is_failed_not_done(conn, fake_voyage, monkeypatch):
    """Zero chunks must never read as success.

    'done' retires a book from the queue permanently, so an empty extraction
    would drop it from the corpus while every status count still looked clean.
    """
    monkeypatch.setattr(embed, "count_tokens", lambda t: max(1, len(t.split())))
    monkeypatch.setattr(
        extract, "extract",
        lambda *a, **k: extract.ExtractionResult("", 2, True, "pymupdf4llm"),
    )
    db.upsert_book(conn, "empty", "Empty Book", None, None)

    _drain_worker(conn, lambda file_id, dest: Path(dest).write_bytes(b"%PDF-1.4"), fake_voyage)

    status, error = conn.execute(
        "SELECT status, error FROM books WHERE source_id = 'empty'"
    ).fetchone()
    assert status == "failed", f"empty extraction was marked {status!r}"
    assert error and "no chunks" in error


def _drain_worker(conn, download_file, voyage_client):
    """A stripped-down copy of run_worker's loop (no real clients) so we can
    assert one bad book doesn't stop the others."""
    processed = 0
    while True:
        book = db.claim_next_book(conn)
        if book is None:
            break
        if book["status"] == "failed":
            continue
        try:
            ingest.process_book(
                conn, book, download_file=download_file, ocr_client=None,
                voyage_client=voyage_client,
            )
        except Exception as e:  # noqa: BLE001
            db.set_status(conn, book["id"], "failed", str(e))
        processed += 1
    return processed


def test_corrupt_pdf_fails_and_worker_continues(conn, fake_voyage, monkeypatch):
    # Avoid tiktoken's network download in count_tokens; word count is enough.
    monkeypatch.setattr(embed, "count_tokens", lambda t: max(1, len(t.split())))

    db.upsert_book(conn, "good", "Good Book", None, None)
    db.upsert_book(conn, "bad", "Bad Book", None, None)

    def download(file_id, dest):
        if file_id == "good":
            _make_text_pdf(Path(dest), pages=2)
        else:
            Path(dest).write_bytes(b"%PDF-1.4 this is not a real pdf")

    _drain_worker(conn, download, fake_voyage)

    statuses = dict(conn.execute("SELECT source_id, status FROM books").fetchall())
    assert statuses["good"] == "done"      # the healthy book completed
    assert statuses["bad"] == "failed"     # the corrupt one is recorded, not fatal

    error = conn.execute(
        "SELECT error FROM books WHERE source_id = 'bad'"
    ).fetchone()[0]
    assert error

    n_good = conn.execute(
        "SELECT COUNT(*) FROM chunks c JOIN books b ON b.id = c.book_id "
        "WHERE b.source_id = 'good'"
    ).fetchone()[0]
    assert n_good > 0


# --------------------------------------- the working PDF, once it is mirrored --
#
# process_book deletes its working PDF after extraction, but ONLY when the
# bucket has confirmed a copy. These pin both halves of that rule, because it is
# the one code path in the project that removes a file the user did not ask it
# to remove.


def _one_book_through(conn, monkeypatch, fake_voyage, *, mirrored: bool):
    monkeypatch.setattr(embed, "count_tokens", lambda t: max(1, len(t.split())))
    monkeypatch.setattr(storage, "put_original", lambda md5, path: mirrored)
    # sync_outputs is the markdown's business, tested in test_storage.py; keep
    # it out of the way so this asserts only on the PDF.
    monkeypatch.setattr(extract, "sync_outputs", lambda book_id: True)

    db.upsert_book(conn, "pdf-life", "A Book", None, None)
    _drain_worker(conn, lambda file_id, dest: _make_text_pdf(Path(dest), pages=2),
                  fake_voyage)
    book_id, status = conn.execute(
        "SELECT id, status FROM books WHERE source_id = 'pdf-life'"
    ).fetchone()
    assert status == "done", f"book did not finish: {status!r}"
    return config.PDF_DIR / f"{book_id}.pdf"


def test_the_working_pdf_goes_once_the_mirror_confirms(conn, fake_voyage, monkeypatch):
    pdf = _one_book_through(conn, monkeypatch, fake_voyage, mirrored=True)
    assert not pdf.exists(), "a confirmed mirror should leave no second copy"


def test_an_unconfirmed_mirror_keeps_the_local_pdf(conn, fake_voyage, monkeypatch):
    """Storage off, or a bucket outage: False means "no confirmed second copy",
    and the only copy there is must survive."""
    pdf = _one_book_through(conn, monkeypatch, fake_voyage, mirrored=False)
    assert pdf.exists(), "an unmirrored book must keep its local PDF"


def test_a_book_parked_for_ocr_clears_its_pdf_too(conn, fake_voyage, monkeypatch):
    """The needs_ocr early return used to skip the cleanup entirely, so every
    scanned book on an install without a Mistral key kept a full PDF forever."""
    monkeypatch.setattr(storage, "put_original", lambda md5, path: True)
    monkeypatch.setattr(
        extract, "extract",
        lambda *a, **k: (_ for _ in ()).throw(extract.NeedsOCRError("scanned")),
    )
    db.upsert_book(conn, "scan", "A Scan", None, None)
    _drain_worker(conn, lambda file_id, dest: _make_text_pdf(Path(dest), pages=1),
                  fake_voyage)

    book_id, status = conn.execute(
        "SELECT id, status FROM books WHERE source_id = 'scan'"
    ).fetchone()
    assert status == "needs_ocr"
    assert not (config.PDF_DIR / f"{book_id}.pdf").exists()
