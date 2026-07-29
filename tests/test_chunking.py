"""chunk_markdown(): heading trails, page-range mapping, junk skipping, dense
ordinals, and overlap when a section is split. Pure functions -- no DB, no
network."""
from pathlib import Path

import config
from pipeline import chunking

FIXTURE = Path(__file__).parent / "fixtures" / "sample_book.md"


def test_headings_pages_and_junk_skip():
    md = FIXTURE.read_text(encoding="utf-8")
    chunks = chunking.chunk_markdown(md)

    assert chunks, "expected at least one chunk"

    # Junk section (# Index) is dropped entirely.
    assert all("index" not in c["heading_trail"].lower() for c in chunks)

    # Chapter 1 is wholly on page 1.
    ch1 = [c for c in chunks if "Getting Started" in c["heading_trail"]]
    assert ch1
    assert ch1[0]["page_start"] == 1
    assert ch1[0]["page_end"] == 1
    # Heading trail is prepended into the content.
    assert ch1[0]["content"].startswith(ch1[0]["heading_trail"])

    # The inductive-study section starts on page 1 (carried) and runs onto
    # page 2 (its <!-- page: 2 --> marker), so it spans 1-2.
    inductive = [c for c in chunks if "Inductive Study" in c["heading_trail"]]
    assert inductive
    assert inductive[0]["page_start"] == 1
    assert inductive[0]["page_end"] == 2

    # Page markers never leak into stored content.
    assert all("<!-- page:" not in c["content"] for c in chunks)


def test_junk_filter_matches_whole_heading_not_substrings():
    # "index" / "contents" as substrings would condemn all four of these.
    assert chunking._is_junk("Index")
    assert chunking._is_junk("Contents")
    assert chunking._is_junk("Bibliography")
    assert not chunking._is_junk("The Contents of the Covenant")
    assert not chunking._is_junk("An Index of Divine Names")
    assert not chunking._is_junk("Indexing Paul's Argument")
    # Judged on the section's OWN heading, so a junk h1 cannot drop a real child.
    assert not chunking._is_junk("Index of Themes > Covenant in Deuteronomy")


def test_junk_sections_are_reported_to_caller():
    md = FIXTURE.read_text(encoding="utf-8")
    dropped = []
    chunking.chunk_markdown(md, dropped=dropped)
    assert dropped == ["Index"], "the fixture's `# Index` should be reported, not silent"


def test_small_sibling_sections_are_merged():
    # Eight `###` subsections of ~40 chars each: without merging every one
    # becomes its own chunk, far below CHUNK_SIZE_CHARS.
    md = "<!-- page: 1 -->\n\n# Chapter 1\n\n" + "\n\n".join(
        f"### Point {i}\n\nA short study note about the passage." for i in range(1, 9)
    )
    chunks = chunking.chunk_markdown(md)

    assert len(chunks) == 1, f"expected one merged chunk, got {len(chunks)}"
    # Trail collapses to the shared parent...
    assert chunks[0]["heading_trail"] == "Chapter 1"
    # ...but each sibling's own heading survives inline, so nothing is lost.
    for i in range(1, 9):
        assert f"Point {i}" in chunks[0]["content"]


def test_top_level_sections_are_never_merged_together():
    # Two short h1 sections share no parent -- fusing chapters would be worse
    # than a short chunk.
    md = "<!-- page: 1 -->\n\n# Chapter 1\n\nShort.\n\n# Chapter 2\n\nAlso short."
    trails = [c["heading_trail"] for c in chunking.chunk_markdown(md)]
    assert trails == ["Chapter 1", "Chapter 2"]


def test_no_chunk_is_heading_only():
    # A section past CHUNK_SIZE_CHARS used to split right after its heading,
    # emitting a chunk whose whole body was "# Romans".
    body = " ".join(
        f"Sentence {i} discusses a distinct point about the epistle." for i in range(120)
    )
    chunks = chunking.chunk_markdown(f"<!-- page: 1 -->\n\n# Romans\n\n{body}")

    assert len(chunks) >= 2
    for c in chunks:
        prose = c["content"].split("\n\n", 1)[1]
        assert not chunking._is_heading_only(prose)
        assert len(prose.strip()) > 60, f"near-empty chunk: {prose!r}"


def test_heading_is_not_duplicated_in_content():
    md = FIXTURE.read_text(encoding="utf-8")
    for c in chunking.chunk_markdown(md):
        trail = c["heading_trail"]
        if not trail:
            continue
        own = trail.rpartition(" > ")[2]
        # The trail is prepended once; the markdown heading must not also remain.
        assert c["content"].count(own) == 1, f"{own!r} repeated in {c['content'][:120]!r}"
        assert "#" not in c["content"].split("\n\n", 1)[1]


def test_ordinals_are_dense_and_overlap_on_split():
    # One section long enough to force RecursiveCharacterTextSplitter to split.
    body = "Inductive Bible study rewards patient observation. " * 200  # ~10k chars
    md = f"<!-- page: 1 -->\n\n# Big Section\n\n{body}"

    chunks = chunking.chunk_markdown(md)

    assert len(chunks) >= 2  # actually split
    # Ordinals are a dense 0..n-1 sequence in order.
    assert [c["ordinal"] for c in chunks] == list(range(len(chunks)))

    # Overlap: total characters across chunks exceed the source body, because
    # consecutive chunks re-share ~CHUNK_OVERLAP_CHARS of text.
    total_chars = sum(len(c["content"]) for c in chunks)
    assert total_chars > len(body)
    assert config.CHUNK_OVERLAP_CHARS > 0
