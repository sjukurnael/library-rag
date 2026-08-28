"""
Build a book's profile: the handful of vectors the librarian searches.

Retrieval is scoped to a study session (see migrations/0010, 0011), so nothing
searches every chunk any more. But the librarian still has to search every
BOOK, and a book needs a representation small enough that 43,000 of them fit in
cache. This module makes it, and it makes it out of data that already exists:
the chunk embeddings were paid for at ingest, so a profile costs arithmetic and
nothing else. No LLM, no API call, no re-download.

The shape is k-means over one book's own chunk vectors. Each centroid is the
MEAN of its cluster, so it matches no actual chunk and is deliberately more
general than any single passage -- which is what makes it a good answer to
"does this book cover X" rather than "which paragraph says X".

Why several vectors and not one: measured on 203 books against 22 questions
with known ground-truth books, ranking all 203 by best-matching vector --

                          median rank   hit@1   hit@10
    one whole-book mean        6        36.4%    68.2%
    ~6 topic centroids         1        63.6%    95.5%

A single mean buries a chapter inside a big book. That case -- a reader asking
about Paul's conversion, and the answer sitting in one chapter of a book called
"Analysis of the Apostles" -- is the one the whole session design rests on.
"""
import collections

import numpy as np

from library_rag import config, db


def cluster_count(n_chunks: int) -> int:
    """How many topic vectors a book of this size gets: round(sqrt(n)), clamped.

    sqrt rather than a constant because the failure is symmetric. Too few and a
    900-page systematic theology averages christology and ecclesiology into one
    blur; too many and a five-chunk fragment is split into clusters that are
    noise. Never more than there are chunks -- k-means cannot find 8 groups in
    5 points, and asking it to would produce duplicate centroids.
    """
    k = round(np.sqrt(max(n_chunks, 1)))
    k = max(config.PROFILE_MIN_CLUSTERS, min(config.PROFILE_MAX_CLUSTERS, int(k)))
    return min(k, n_chunks)


def _kmeans(V: np.ndarray, k: int) -> tuple:
    """Lloyd's algorithm on unit vectors. Returns (centroids, assignment).

    Cosine distance, so everything is normalised and similarity is a plain dot
    product -- which is why the centroids are re-normalised after every update.
    The mean of unit vectors is NOT a unit vector, and leaving it short would
    quietly bias every later comparison towards clusters that happen to be
    tightly packed.

    Init is farthest-first, not random: a rebuild has to produce byte-identical
    centroids, or an eval harness comparing two runs is measuring the seed
    instead of the corpus. An emptied cluster keeps its previous centroid rather
    than being re-seeded, for the same reason.
    """
    if k >= len(V):
        return V.copy(), np.arange(len(V))

    centroids = [V[0]]
    for _ in range(k - 1):
        far = 1.0 - np.max(V @ np.array(centroids).T, axis=1)
        centroids.append(V[int(np.argmax(far))])
    C = np.array(centroids)

    assign = np.zeros(len(V), dtype=int)
    for _ in range(config.PROFILE_KMEANS_ITERS):
        assign = np.argmax(V @ C.T, axis=1)
        moved = np.array([
            V[assign == j].mean(axis=0) if np.any(assign == j) else C[j]
            for j in range(k)
        ])
        moved /= np.linalg.norm(moved, axis=1, keepdims=True)
        if np.allclose(moved, C, atol=1e-6):
            C = moved
            break
        C = moved
    return C, np.argmax(V @ C.T, axis=1)


def _strip_shared_root(trails: list) -> list:
    """Drop a leading heading component that every trail in the book shares.

    Scanned books routinely acquire a junk root: a ProQuest dissertation opens
    with an "INFORMATION TO USERS" banner, and pymupdf4llm reads the biggest
    text on page one as h1, so every heading_trail in the book begins with it.
    A component present in every trail cannot distinguish one cluster from
    another -- which is the only job a label has -- so it is pure noise in a
    result list the reader is meant to skim.

    Stripped only while the first component is unanimous. A trail that IS just
    the root keeps it -- that cluster genuinely has no subheading, and blanking
    it would leave the label empty -- which is also what terminates the loop:
    once the bare-root trail and its stripped siblings disagree about the first
    component, there is nothing shared left to remove.
    """
    parts = [[c.strip() for c in t.split(">")] for t in trails]
    while len({p[0] for p in parts}) == 1 and any(len(p) > 1 for p in parts):
        parts = [p[1:] if len(p) > 1 else p for p in parts]
    return [" > ".join(p) for p in parts]


def _label(member_idx: np.ndarray, headings: list, contents: list,
           centroid: np.ndarray, V: np.ndarray, all_trails: list) -> str:
    """A human-readable name for a cluster, derived from its members.

    The label is cosmetic -- it never touches retrieval -- but it is what lets a
    recommendation justify itself, and a reader can only overrule the librarian
    if they can see why it chose something.

    Most-common heading beats nearest-chunk heading: the centroid is the average
    of the whole cluster, so the heading most of its members sit under describes
    it better than whichever single chunk happens to be closest. Falls back to
    the opening words of the nearest chunk, which always exists -- so no cluster
    is ever unlabelled, however structureless the PDF.
    """
    trails = [headings[i] for i in member_idx
              if headings[i] and headings[i].strip()]
    if trails:
        best = collections.Counter(trails).most_common(1)[0][0]
        return _strip_shared_root(all_trails + [best])[-1] if all_trails else best
    nearest = member_idx[int(np.argmax(V[member_idx] @ centroid))]
    words = (contents[nearest] or "").split()[: config.PROFILE_LABEL_WORDS]
    return " ".join(words) if words else None


def build(conn, book_id: int) -> dict:
    """Compute and store one book's profile. Returns a small summary of it."""
    rows = db.chunks_for_profile(conn, book_id)
    if not rows:
        return {"book_id": book_id, "vectors": 0, "source": None}

    # float32, not the stored float16: a 200-chunk mean accumulates 200 additions,
    # and half precision loses enough over that to move a centroid visibly.
    V = np.array([r[0].to_numpy() for r in rows], dtype=np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    headings = [r[1] for r in rows]
    contents = [r[2] for r in rows]

    # The book's structure, recovered from the chunks rather than the PDF:
    # pymupdf4llm maps font sizes onto heading levels, so distinct heading_trails
    # in reading order ARE its table of contents. First-occurrence order, deduped
    # -- sorting would scramble a TOC into an index. Computed before clustering
    # because the labels need it to know which root prefix is shared book-wide.
    seen, all_trails = set(), []
    for h in headings:
        h = (h or "").strip()
        if h and h not in seen:
            seen.add(h)
            all_trails.append(h)

    k = cluster_count(len(V))
    C, assign = _kmeans(V, k)

    # ordinal 0 is the whole-book mean. It answers the altitude the clusters
    # cannot: "I'm studying systematic theology" matches no single chapter well
    # but matches this strongly.
    #
    # Skipped at k == 1, where the sole cluster centroid IS the mean of every
    # chunk and storing both would be the same vector twice. ~15% of this corpus
    # is single-topic fragments, so the duplicate is not rare enough to ignore.
    vectors = []
    if k > 1:
        whole = V.mean(axis=0)
        whole /= np.linalg.norm(whole)
        vectors.append((0, None, len(V), whole.tolist()))

    for j in range(len(C)):
        members = np.flatnonzero(assign == j)
        if members.size == 0:
            continue
        vectors.append((
            j + 1,
            _label(members, headings, contents, C[j], V, all_trails),
            int(members.size),
            C[j].tolist(),
        ))

    trail = _strip_shared_root(all_trails) if all_trails else []
    source = "headings" if trail else "text"
    toc = " / ".join(trail) if trail else None

    db.store_book_profile(conn, book_id, toc, source, vectors)
    return {"book_id": book_id, "vectors": len(vectors), "source": source,
            "chunks": len(V), "k": k}
