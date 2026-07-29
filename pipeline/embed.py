"""
The ONLY place that calls the Voyage API. ingest.py (documents) and search.py
(queries) both import this, so the exact same model embeds both sides of every
vector comparison -- a query embedded with a different model than the documents
would silently return garbage rankings.

The Voyage client is passed in (build one with build_client()), never held as a
module-level singleton, so tests can inject a deterministic fake with the same
tiny surface: a .embed(texts, model=..., input_type=...) returning an object
with .embeddings and .total_tokens.
"""
import random
import time

import tiktoken

import config

_encoding = None


def build_client():
    """Construct a real Voyage client. Raises if VOYAGE_API_KEY is unset."""
    import voyageai  # imported here so tests never need the dependency configured

    if not config.VOYAGE_API_KEY:
        raise RuntimeError("VOYAGE_API_KEY is not set (see .env.example)")
    return voyageai.Client(api_key=config.VOYAGE_API_KEY)


def embed_documents(texts: list, client) -> tuple:
    """Embed chunk texts (input_type='document'), batched.
    Returns (embeddings, total_tokens_reported_by_the_api)."""
    return _embed(texts, client, input_type="document")


def embed_query(text: str, client) -> list:
    """Embed a single query (input_type='query') with the SAME model as the
    documents -- this is the shared code path that keeps query/doc vectors in
    one space."""
    embeddings, _ = _embed([text], client, input_type="query")
    return embeddings[0]


def count_tokens(text: str) -> int:
    """tiktoken approximation, used for the per-chunk token_count column."""
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return len(_encoding.encode(text))


def _embed(texts: list, client, input_type: str) -> tuple:
    all_embeddings = []
    total_tokens = 0
    for i in range(0, len(texts), config.EMBED_BATCH_SIZE):
        batch = texts[i : i + config.EMBED_BATCH_SIZE]
        result = _embed_batch_with_retry(client, batch, input_type)
        all_embeddings.extend(result.embeddings)
        total_tokens += result.total_tokens
    return all_embeddings, total_tokens


def _embed_batch_with_retry(client, batch: list, input_type: str, max_retries: int = 5):
    for attempt in range(max_retries):
        try:
            return client.embed(batch, model=config.EMBED_MODEL, input_type=input_type)
        except Exception as e:
            status = getattr(e, "status_code", None) or getattr(
                getattr(e, "response", None), "status_code", None
            )
            if status == 429 and attempt < max_retries - 1:
                delay = (2**attempt) + random.uniform(0, 1)
                print(f"  [embed] 429 rate limited, retrying in {delay:.1f}s ...")
                time.sleep(delay)
                continue
            raise
