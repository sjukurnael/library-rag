"""Prompt caching for the two agent loops.

Both loops re-send the whole conversation on every turn, which is how tool use
works -- but it means a 31-turn librarian run pays full input price for the same
early tool results thirty times over. The cost of a run is therefore quadratic
in its length, and look_inside results are passages, so the turns are not small.

Caching does not change what the model sees. It changes what it is charged for:
a cache read is a tenth of the input price, a write is 1.25x. So the break-even
is a single re-read, and every turn after the first is a re-read by definition.

Two breakpoints, which is well inside the limit of four:

  - the system prompt and tool schemas, static for a whole run (~2.1k tokens,
    above the 1024-token minimum a block needs to be cacheable at all);
  - the end of the conversation, moved forward each turn, so turn N writes a
    cache covering everything up to it and turn N+1 reads it.

The second one is why this is worth doing. The static prefix is the small half.
"""


def cacheable_system(prompt: str) -> list[dict]:
    """The system prompt as a cached block.

    A plain string cannot carry cache_control, so the caller passes the list
    form instead. Placed on system, the breakpoint covers the tool schemas too:
    the cached prefix is ordered tools, then system, then messages.
    """
    return [{"type": "text", "text": prompt,
             "cache_control": {"type": "ephemeral"}}]


def move_breakpoint(messages: list[dict]) -> None:
    """Move the conversation breakpoint to the end of the last message.

    Mutates in place, and clears the previous one first: a breakpoint left
    behind is a second cache entry written for a prefix nothing will read again.

    Only dict blocks are touched. Assistant turns are appended as the SDK's own
    content objects rather than dicts, and they are never the last message when
    this is called -- the loop always appends tool results after them -- so
    skipping them costs nothing and avoids mutating SDK models.
    """
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block.pop("cache_control", None)

    last = messages[-1].get("content")
    if isinstance(last, list) and last and isinstance(last[-1], dict):
        last[-1]["cache_control"] = {"type": "ephemeral"}
