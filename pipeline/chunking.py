"""
Markdown -> chunks, ready to embed.

Three passes. MarkdownHeaderTextSplitter groups the document by heading, so
every chunk knows its heading trail. Undersized sibling sections are then merged
up to MIN_CHUNK_CHARS -- the header splitter only ever *splits*, so without this
a study guide's dense `###` subheadings each become their own ~40-char chunk.
Finally RecursiveCharacterTextSplitter subdivides anything still over
CHUNK_SIZE_CHARS.

Page ranges are recovered from the `<!-- page: N -->` markers
pipeline/extract.py leaves between pages, tracked across a section's sub-chunks
so a chunk in the middle of a long section still gets a sensible page range
instead of the whole section's span. Each returned chunk carries an `ordinal`
(0..n-1 across the whole book, in reading order).
"""
import re

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

import config

PAGE_MARKER_RE = re.compile(r"<!--\s*page:\s*(\d+)\s*-->")
# A whole markdown heading line, used to detect pieces that carry no prose.
HEADING_LINE_RE = re.compile(r"^\s{0,3}#{1,6}[^\n]*$", re.MULTILINE)


def chunk_markdown(markdown: str, dropped: list | None = None) -> list:
    """Return a list of {ordinal, heading_trail, page_start, page_end, content}
    dicts. content already has the heading trail prepended; ordinal is a dense
    0-based sequence over the whole document in reading order.

    Pass a list as `dropped` to collect the headings of junk sections that were
    skipped -- otherwise the filter is silent, and silent content loss is very
    hard to notice downstream.

    A single running `last_page` walks the whole document in order, so a chunk
    always starts on the page the previous chunk ended on -- even when the
    header splitter attributed a page's leading `<!-- page: N -->` marker to the
    tail of the previous section (headings that fall at the top of a page).
    """
    # strip_headers=True: the heading is already carried (with its ancestors) in
    # heading_trail, which is prepended to every chunk's content below. Leaving
    # it in the body too would repeat it in every chunk.
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=config.MARKDOWN_HEADERS, strip_headers=True
    )

    sections = []
    for section in header_splitter.split_text(markdown):
        trail = " > ".join(v for v in section.metadata.values() if v)
        if trail and _is_junk(trail):
            # Junk sections are dropped, but their page markers still advance
            # the counter so later chunks keep the right page numbers.
            sections.append((trail, section.page_content, True))
            if dropped is not None:
                dropped.append(trail)
            continue
        sections.append((trail, section.page_content, False))

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE_CHARS,
        chunk_overlap=config.CHUNK_OVERLAP_CHARS,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    last_page = None
    for heading_trail, content, is_junk in _merge_small_sections(sections):
        if is_junk:
            markers = _markers(content)
            if markers:
                last_page = max(markers)
            continue

        for piece in recursive_splitter.split_text(content):
            pages_in_piece = _markers(piece)
            clean_piece = PAGE_MARKER_RE.sub("", piece).strip()
            # A long section can split right after its heading, leaving a piece
            # with no prose at all. Advance the page counter, store nothing.
            if not clean_piece or _is_heading_only(clean_piece):
                if pages_in_piece:
                    last_page = max(pages_in_piece)
                continue

            if pages_in_piece:
                page_start = last_page if last_page is not None else min(pages_in_piece)
                page_end = max(pages_in_piece)
                last_page = page_end
            else:
                page_start = page_end = last_page

            content_out = (
                f"{heading_trail}\n\n{clean_piece}" if heading_trail else clean_piece
            )
            chunks.append(
                {
                    "heading_trail": heading_trail,
                    "page_start": page_start,
                    "page_end": page_end,
                    "content": content_out,
                }
            )

    for ordinal, c in enumerate(chunks):
        c["ordinal"] = ordinal
    return chunks


def _merge_small_sections(sections: list) -> list:
    """Merge runs of undersized sibling sections into one.

    Two sections are siblings when they share the same non-empty parent trail
    ("Ch 3 > A" and "Ch 3 > B" share "Ch 3"). A run keeps absorbing siblings
    while it is under MIN_CHUNK_CHARS. The merged section's trail collapses to
    the shared parent, and each absorbed section's own heading is re-inserted as
    a body line so no structure is lost.

    Only sections SHORTER than MIN_CHUNK_CHARS are absorbed. A section already
    big enough to stand alone keeps its own precise trail, and -- more
    importantly -- this bounds a merged body at under 2 * MIN_CHUNK_CHARS, well
    below CHUNK_SIZE_CHARS. Without that bound a small run could swallow an
    oversized section, push the merged body past CHUNK_SIZE_CHARS, and get its
    inline heading stranded as a content-free chunk by the recursive splitter --
    reintroducing the very bug the inline-heading binding avoids.

    The parent must be non-empty, so top-level sections are never merged with
    each other -- fusing two chapters into one chunk would be worse than a short
    chunk. Junk sections are passed through untouched (the caller still needs
    their page markers).
    """
    merged = []
    run = []  # list of (own_heading, content)
    run_trail = None
    run_parent = None
    run_len = 0

    def flush():
        nonlocal run, run_trail, run_parent, run_len
        if run:
            if len(run) == 1:
                merged.append((run_trail, run[0][1], False))
            else:
                # Single "\n" (not "\n\n") keeps a heading attached to its own
                # text: the recursive splitter prefers "\n\n" boundaries, so it
                # will not orphan the heading unless it has to.
                body = "\n\n".join(
                    f"{own}\n{content.strip()}" if own else content.strip()
                    for own, content in run
                )
                merged.append((run_parent, body, False))
        run, run_trail, run_parent, run_len = [], None, None, 0

    for trail, content, is_junk in sections:
        if is_junk:
            flush()
            merged.append((trail, content, True))
            continue

        parent, _, own = trail.rpartition(" > ")
        if (
            run
            and run_len < config.MIN_CHUNK_CHARS
            and len(content) < config.MIN_CHUNK_CHARS
            and parent
            and parent == run_parent
        ):
            run.append((own, content))
            run_len += len(content)
            continue

        flush()
        run = [(own, content)]
        run_trail = trail
        run_parent = parent
        run_len = len(content)

    flush()
    return merged


def _markers(text: str) -> list:
    return [int(m) for m in PAGE_MARKER_RE.findall(text)]


def _is_heading_only(text: str) -> bool:
    """True when nothing but markdown heading lines remain -- no prose to embed."""
    return not HEADING_LINE_RE.sub("", text).strip()


def _is_junk(heading_trail: str) -> bool:
    """Judge a section by its OWN heading, matched in full against
    config.JUNK_HEADINGS. Deliberately not a substring test over the whole
    trail: "index" as a substring also condemns "An Index of Divine Names", and
    testing the trail lets a junk `h1` drop every real section nested under it.
    """
    own = heading_trail.rpartition(" > ")[2]
    return own.strip().lower().rstrip(".:").strip() in config.JUNK_HEADINGS
