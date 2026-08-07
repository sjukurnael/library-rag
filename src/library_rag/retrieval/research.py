"""
Backward-compatible entry point for the research agent.

Prefer importing from retrieval.loop (prompt + run) and retrieval.tools
(search / list_books) directly. This module re-exports the public surface so
existing `from library_rag.retrieval import research` call sites keep working.
"""
from library_rag.retrieval.loop import (  # noqa: F401
    MAX_ITERATIONS,
    MODEL,
    SYSTEM_PROMPT,
    TOOL_SCHEMAS,
    run,
)
from library_rag.retrieval.tools import (  # noqa: F401
    DEFAULT_K,
    MAX_K,
    MAX_SEARCHES,
    RELEVANCE_CUTOFF,
    Session,
    db,
    embed_mod,
)
