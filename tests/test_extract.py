"""Extraction probe on generated PDFs (real text vs image-only), plus the
worker-level guarantee that a corrupt PDF is recorded as failed and does not
stop the run. PDFs are built with PyMuPDF at test time -- no fixtures on disk,
no network."""
from pathlib import Path

import fitz
import pytest

import db
import ingest
from pipeline import embed, extract


def _make_text_pdf(dest, pages=3):
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_text(
            (72, 100),
            "Inductive Bible study: observation, interpretation, application. " * 4,
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

    statuses = dict(conn.execute("SELECT drive_file_id, status FROM books").fetchall())
    assert statuses["good"] == "done"      # the healthy book completed
    assert statuses["bad"] == "failed"     # the corrupt one is recorded, not fatal

    error = conn.execute(
        "SELECT error FROM books WHERE drive_file_id = 'bad'"
    ).fetchone()[0]
    assert error

    n_good = conn.execute(
        "SELECT COUNT(*) FROM chunks c JOIN books b ON b.id = c.book_id "
        "WHERE b.drive_file_id = 'good'"
    ).fetchone()[0]
    assert n_good > 0
