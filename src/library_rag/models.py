"""The two models a reader may pick between, and the rule that a request cannot
name anything else.

The choice reaches the server as "sonnet" or "opus" -- a label, never a model
id. A request that could name the model would be a request that could name any
model: a typo bills nothing and 400s, but a deliberate string could point this
system's traffic at whatever the caller likes, on this project's key. So the
wire format is a closed set and the ids live here.

Defaults still come from LIBRARIAN_MODEL / TUTOR_MODEL in the environment. This
overrides per run, for the reader who wants one shortlist thought about harder;
it is not a way to change what the deployment normally uses.
"""

SONNET = "claude-sonnet-5"
OPUS = "claude-opus-5"

# label -> model id. The keys are the whole of what a client may send.
CHOICES = {"sonnet": SONNET, "opus": OPUS}

# What each label costs, for the UI to say so rather than making the reader
# look it up. $/1M tokens, (input, output).
PRICES = {"sonnet": (2.00, 10.00), "opus": (5.00, 25.00)}


def resolve(label: str | None, fallback: str) -> str:
    """A label to a model id, or the deployment's default when unset.

    Unknown labels fall back rather than raising: the API bounds this field at
    the edge, so anything reaching here has already been validated, and a run
    that fails because of a stray query parameter is a worse outcome than one
    that quietly uses the default.
    """
    if not label:
        return fallback
    return CHOICES.get(label, fallback)
