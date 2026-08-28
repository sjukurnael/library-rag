"""Prompt caching, which fails silently in both directions.

A breakpoint that never gets set means every turn pays full input price for the
whole conversation again; a breakpoint left behind on an old turn writes a cache
entry nothing will ever read. Neither raises, neither changes a single word of
what the model returns, and neither shows up anywhere except the bill. So the
placement is asserted directly.
"""
from library_rag import prompt_cache


EPHEMERAL = {"type": "ephemeral"}


def _breakpoints(messages):
    """Every (message index, block index) currently carrying a breakpoint."""
    found = []
    for i, message in enumerate(messages):
        content = message.get("content")
        if isinstance(content, list):
            for j, block in enumerate(content):
                if isinstance(block, dict) and "cache_control" in block:
                    found.append((i, j))
    return found


def test_the_system_block_is_cacheable():
    """A bare string cannot carry cache_control, so the list form is the whole
    point of the wrapper."""
    system = prompt_cache.cacheable_system("you are a librarian")
    assert system == [{"type": "text", "text": "you are a librarian",
                       "cache_control": EPHEMERAL}]


def test_the_breakpoint_lands_on_the_end_of_the_last_message():
    messages = [
        {"role": "user", "content": "find me books on the atonement"},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a",
                                      "content": "{}"}]},
    ]
    prompt_cache.move_breakpoint(messages)
    assert _breakpoints(messages) == [(1, 0)]


def test_only_one_breakpoint_survives_a_move():
    """The reason to clear before setting. Four is the hard limit, a run is up
    to forty turns, and a stale breakpoint is a paid-for cache write covering a
    prefix that will never be read again."""
    messages = [{"role": "user", "content": "brief"}]
    for turn in range(6):
        messages.append({"role": "assistant", "content": []})
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": str(turn), "content": "{}"}]})
        prompt_cache.move_breakpoint(messages)

        marks = _breakpoints(messages)
        assert marks == [(len(messages) - 1, 0)], (
            f"turn {turn}: expected one breakpoint on the last message, got {marks}"
        )


def test_a_first_turn_with_a_plain_string_is_left_alone():
    """The opening message is the brief -- a string, not a block list. There is
    nothing to hang a breakpoint on and nothing worth caching yet; the system
    prompt carries its own. This must not raise."""
    messages = [{"role": "user", "content": "find me books on the atonement"}]
    prompt_cache.move_breakpoint(messages)
    assert _breakpoints(messages) == []


def test_sdk_content_objects_are_not_mutated():
    """Assistant turns are appended as the SDK's own content models, not dicts.
    Calling .pop on those would raise, and setting an unexpected attribute would
    be sent back to the API on the next turn."""
    class Block:
        type = "text"
        text = "thinking out loud"

    block = Block()
    messages = [
        {"role": "assistant", "content": [block]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a",
                                      "content": "{}"}]},
    ]
    prompt_cache.move_breakpoint(messages)

    assert not hasattr(block, "cache_control")
    assert _breakpoints(messages) == [(1, 0)]
