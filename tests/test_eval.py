"""
evaluate.py: the scoring logic, and the retrieval it scores.

Two halves, and the split matters.

The pure half tests the scoring functions directly -- page overlap, first-hit
rank, hit-rate and MRR arithmetic. A harness that miscounts is worse than no
harness, because it reports a number that gets believed.

The corpus half stands up a small synthetic library in Postgres and runs real
db.search over it in all three modes. The embedder is conftest.lexical_vector
(hashed bag-of-words), not Voyage: the point is to check that the SQL ranks and
that the harness can tell a good run from a bad one, with no network and no API
key. It cannot tell you whether voyage-4-lite is a good model -- that is what
`python -m evaluate` against the live corpus is for, and the numbers it produced
are recorded in config.SEARCH_MODE's comment.

The load-bearing test here is test_the_harness_fails_when_retrieval_is_broken.
A gate that cannot go red is decoration.
"""
import pytest

import config
import db
import evaluate
from tests.conftest import lexical_vector

# --------------------------------------------------------- synthetic corpus --

CORPUS = [
    (
        "Gardening Handbook",
        [
            (
                [4, 6],
                "Pruning roses. Cut back rose bushes in late winter while the plant "
                "is dormant. Remove dead or crossing canes first, then shorten the "
                "remaining canes to an outward facing bud. Clean pruning shears "
                "between plants so disease does not spread from one rose to another.",
            ),
            (
                [7, 9],
                "Building a compost heap. Alternate green material such as grass "
                "clippings with brown material such as dried leaves and straw. Turn "
                "the compost heap every few weeks so air reaches the centre, and keep "
                "it as damp as a wrung out sponge but never sodden.",
            ),
            (
                [10, 12],
                "Tomato blight. Blight appears as dark patches on tomato leaves and "
                "stems in warm wet weather, then rots the fruit. Remove and destroy "
                "affected plants, never compost them, and grow tomatoes under cover "
                "where the foliage can stay dry.",
            ),
        ],
    ),
    (
        "Bicycle Maintenance",
        [
            (
                [2, 3],
                "Chain lubrication. Wipe the bicycle chain clean, then apply one drop "
                "of lubricant to each roller while backpedalling. Wipe off the excess, "
                "because lubricant left on the outside of the chain collects grit and "
                "grinds away the drivetrain faster than no lubricant at all.",
            ),
            (
                [4, 5],
                "Brake pads. Replace brake pads once the grooves in the pad surface "
                "have worn flat. Worn pads let the metal backing touch the rim, which "
                "destroys the braking surface of the wheel and costs far more than the "
                "pads would have.",
            ),
            (
                [6, 8],
                "Truing a wheel. A buckled wheel is straightened by tightening and "
                "loosening spoke nipples a quarter turn at a time. Work slowly and "
                "check the rim against the brake pad after each adjustment.",
            ),
        ],
    ),
    (
        "Bread Baking",
        [
            (
                [11, 13],
                "Feeding a sourdough starter. Discard most of the starter, then feed "
                "the remainder with equal weights of flour and water. A healthy "
                "starter doubles within a few hours at room temperature and smells "
                "sharp and yeasty rather than of acetone.",
            ),
            (
                [14, 16],
                "Kneading dough. Kneading develops the gluten network that traps gas. "
                "The dough is ready when a small piece can be stretched thin enough to "
                "let light through without tearing, which bakers call the window pane "
                "test.",
            ),
            (
                [17, 18],
                "Oven spring. Oven spring is the rapid rise a loaf makes in the first "
                "minutes of baking, as trapped gas expands and the yeast makes one "
                "last burst before the heat kills it. Steam in the oven keeps the "
                "crust soft long enough for the loaf to expand fully.",
            ),
        ],
    ),
]

QUESTIONS = [
    {
        "id": "prune-roses",
        "question": "When should I cut back rose bushes?",
        "expect": [{"book": "Gardening", "pages": [4, 6]}],
    },
    {
        "id": "tomato-blight",
        "question": "What causes dark patches on tomato leaves?",
        "expect": [{"book": "Gardening", "pages": [10, 12]}],
    },
    {
        "id": "chain-lube",
        "question": "How do I apply lubricant to a bicycle chain?",
        "expect": [{"book": "Bicycle", "pages": [2, 3]}],
    },
    {
        "id": "brake-pads",
        "question": "When do worn brake pads need replacing?",
        "expect": [{"book": "Bicycle", "pages": [4, 5]}],
    },
    {
        "id": "sourdough-starter",
        "question": "How do I feed a sourdough starter with flour and water?",
        "expect": [{"book": "Bread", "pages": [11, 13]}],
    },
    {
        "id": "oven-spring",
        "question": "What is oven spring when baking a loaf?",
        "expect": [{"book": "Bread", "pages": [17, 18]}],
    },
]


@pytest.fixture
def corpus(conn):
    """Load CORPUS into the test database, embedded with lexical_vector."""
    for title, chunks in CORPUS:
        db.upsert_book(conn, f"drive-{title}", title, f"md5-{title}", 1000)
        book_id = conn.execute(
            "SELECT id FROM books WHERE drive_file_id = %s", (f"drive-{title}",)
        ).fetchone()[0]
        db.insert_chunks_and_finish(
            conn,
            book_id,
            [
                {
                    "ordinal": i,
                    "heading_trail": None,
                    "page_start": pages[0],
                    "page_end": pages[1],
                    "content": text,
                    "token_count": len(text.split()),
                    "embedding": lexical_vector(text),
                }
                for i, (pages, text) in enumerate(chunks)
            ],
        )
    return conn


def _vectors():
    return {q["id"]: lexical_vector(q["question"]) for q in QUESTIONS}


# --------------------------------------------------------------- scoring --

def test_a_hit_needs_the_right_book_and_an_overlapping_page_span():
    target = {"book": "Gardening", "pages": [10, 12]}
    row = lambda title, s, e: {"title": title, "page_start": s, "page_end": e}  # noqa: E731

    assert evaluate.matches(row("Gardening Handbook", 10, 12), target)
    assert evaluate.matches(row("gardening handbook", 8, 10), target), "overlap counts"
    assert evaluate.matches(row("Gardening Handbook", 12, 20), target), "overlap counts"
    assert not evaluate.matches(row("Gardening Handbook", 1, 9), target)
    assert not evaluate.matches(row("Gardening Handbook", 13, 30), target)
    assert not evaluate.matches(row("Bread Baking", 10, 12), target), "wrong book"
    assert not evaluate.matches(row("Gardening Handbook", None, None), target)


def test_overlap_not_containment_survives_a_rechunk():
    """The reason ground truth is a page span and not a chunk id: re-chunking
    moves every boundary, and an eval set that reports all-misses the moment
    chunking changes is useless exactly when it is needed."""
    target = {"book": "Bread", "pages": [14, 16]}
    merged = {"title": "Bread Baking", "page_start": 11, "page_end": 18}
    split = {"title": "Bread Baking", "page_start": 15, "page_end": 15}
    assert evaluate.matches(merged, target)
    assert evaluate.matches(split, target)


def test_first_hit_rank_is_one_based_and_none_on_a_miss():
    expect = [{"book": "Bicycle", "pages": [4, 5]}]
    rows = [
        {"title": "Bread Baking", "page_start": 11, "page_end": 13},
        {"title": "Bicycle Maintenance", "page_start": 4, "page_end": 5},
    ]
    assert evaluate.first_hit_rank(rows, expect) == 2
    assert evaluate.first_hit_rank(rows[:1], expect) is None
    assert evaluate.first_hit_rank([], expect) is None


def test_hit_rate_and_mrr_arithmetic():
    s = evaluate.score([{"rank": 1}, {"rank": 2}, {"rank": None}, {"rank": 4}])
    assert s["n"] == 4
    assert s["hits"] == 3
    assert s["hit_rate"] == 0.75
    # A miss contributes 0 to the numerator but still counts in the denominator;
    # averaging over hits only would make a harness look better the more it
    # missed.
    assert s["mrr"] == pytest.approx((1 + 0.5 + 0.25) / 4)
    assert evaluate.score([]) == {"n": 0, "hit_rate": 0.0, "mrr": 0.0, "hits": 0}


def test_the_question_set_shipped_in_the_repo_is_well_formed():
    """Cheap, and it catches the failure mode where a hand-edited JSON file
    silently drops a field and every question starts scoring as a miss."""
    for path in ("eval/questions.json", "eval/questions_paraphrase.json"):
        data = evaluate.load_questions(config.BASE_DIR / path)
        ids = [q["id"] for q in data["questions"]]
        assert len(ids) == len(set(ids)), f"duplicate question ids in {path}"
        for q in data["questions"]:
            assert q["question"].strip(), f"{q['id']} has no question text"
            assert q["expect"], f"{q['id']} has no expected passage"
            for t in q["expect"]:
                start, end = t["pages"]
                assert 0 < start <= end, f"{q['id']} has a nonsense page span"


# -------------------------------------------------------------- retrieval --

@pytest.mark.parametrize("mode", ["dense", "hybrid"])
def test_the_shipping_modes_retrieve_the_right_passage(corpus, mode):
    results = evaluate.run_questions(corpus, QUESTIONS, _vectors(), 3, mode=mode)
    s = evaluate.score(results)
    missed = [r["id"] for r in results if r["rank"] is None]
    assert s["hit_rate"] == 1.0, f"{mode} missed {missed}"


def test_the_lexical_leg_alone_is_weaker_than_the_modes_that_ship(corpus):
    """Not held to 1.0 on purpose. A bag-of-words ranker with no IDF is a
    contributor, not a retrieval system, and pinning it at perfect here would be
    asserting something this codebase does not believe -- see the measurements
    in config.SEARCH_MODE, where the lexical leg scored 63.6% and 9.1% on the
    real corpus."""
    s = evaluate.score(
        evaluate.run_questions(corpus, QUESTIONS, _vectors(), 3, mode="lexical")
    )
    assert 0 < s["hit_rate"] <= 1.0, "the lexical leg found nothing at all"


def test_the_harness_fails_when_retrieval_is_broken(corpus, monkeypatch):
    """The gate must be able to go red.

    Every embedding is replaced with one from an unrelated text, so the dense
    leg is ranking noise. If hit-rate stayed high here, it would mean the
    questions are answerable from any passage and the whole eval set proves
    nothing.
    """
    corpus.execute(
        "UPDATE chunks SET embedding = %s",
        (db.HalfVector(lexical_vector("unrelated filler text about nothing")),),
    )
    corpus.commit()

    s = evaluate.score(
        evaluate.run_questions(corpus, QUESTIONS, _vectors(), 3, mode="dense")
    )
    assert s["hit_rate"] < 0.5, (
        "retrieval was deliberately broken and the harness still passed -- "
        "it is not measuring what it claims to"
    )


def test_book_scoping_restricts_every_mode(corpus):
    bread_id = corpus.execute(
        "SELECT id FROM books WHERE title = 'Bread Baking'"
    ).fetchone()[0]
    q = "how do I lubricate a bicycle chain"
    for mode in ("dense", "hybrid", "lexical"):
        rows = db.search(
            corpus, lexical_vector(q), 5, bread_id, query_text=q, mode=mode
        )
        # Subset, not equality: the query is about a different book, so the
        # lexical leg legitimately matches nothing inside Bread Baking. The
        # invariant being tested is that no mode ESCAPES the scope, which an
        # empty result satisfies and a foreign book_id does not.
        assert {r["book_id"] for r in rows} <= {bread_id}, f"{mode} escaped the book"
    for mode in ("dense", "hybrid"):
        rows = db.search(
            corpus, lexical_vector(q), 5, bread_id, query_text=q, mode=mode
        )
        assert rows, f"{mode} returned nothing even though the book has chunks"


def test_lexical_only_finds_a_rare_token_the_question_never_paraphrases(corpus):
    """What the lexical leg is FOR: an exact token. 'nipples' appears in one
    chunk in the corpus and nowhere else."""
    rows = db.search(
        corpus, lexical_vector("spoke nipples"), 3,
        query_text="spoke nipples", mode="lexical",
    )
    assert rows
    assert "nipple" in rows[0]["content"].lower()


def test_an_all_stopword_query_degrades_instead_of_erroring(corpus):
    """websearch/to_tsvector both reduce 'the of a and' to nothing. The lexical
    leg must return no rows rather than raise on an empty tsquery cast."""
    for tsq in ("and", "or"):
        rows = db.search(
            corpus, lexical_vector("the of a and"), 3,
            query_text="the of a and", mode="lexical", tsquery_mode=tsq,
        )
        assert rows == []


def test_hybrid_still_returns_a_distance_for_a_lexical_only_hit(corpus, monkeypatch):
    """agent/research.py reads `distance` on every row to decide weak vs strong.
    A lexical-only hit arriving with distance NULL would read as a perfect
    match and be reported to the model as strong evidence.

    The candidate pool is squeezed to 2 to manufacture the case at all: this
    corpus has 9 chunks and the real pool is 50, so the dense leg returns
    everything and a lexical-only hit cannot exist. On the real corpus it is the
    normal state of affairs, which is exactly why it needs a test here.
    """
    monkeypatch.setattr(config, "HYBRID_CANDIDATES", 2)
    rows = db.search(
        corpus, lexical_vector("compost heap"), 8,
        query_text="spoke nipples", mode="hybrid",
    )
    assert rows
    assert all(r["distance"] is not None for r in rows)
    assert any(r["dense_rank"] is None for r in rows), (
        "no lexical-only hit in this result set, so the assertion above proved "
        "nothing -- the fixture needs a token the dense leg does not surface"
    )


def test_unknown_mode_and_missing_query_text_are_rejected(corpus):
    with pytest.raises(ValueError, match="unknown search mode"):
        db.search(corpus, lexical_vector("x"), 3, mode="magic")
    with pytest.raises(ValueError, match="needs query_text"):
        db.search(corpus, lexical_vector("x"), 3, mode="hybrid")
