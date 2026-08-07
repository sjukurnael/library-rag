"""
The one piece of frontend logic worth testing in isolation: readEventStream's
guard against a stream that ends without its terminal event.

That case is a silent failure by construction -- the read loop simply exits, no
exception is raised, `catch` never runs, and the page keeps a half-drawn trail
forever. It is also invisible in manual testing, because it only happens when
the server dies mid-response.

Run through node rather than a browser harness: app.js is a plain script with no
module system, and adding a bundler to test forty lines would cost more than it
protects. GitHub's ubuntu-latest ships node, so this runs in CI.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parent / "frontend"

needs_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)


def _run(script: str):
    result = subprocess.run(
        ["node", str(FRONTEND / script)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


@needs_node
def test_the_event_stream_guard_behaves():
    _run("stream_guard.mjs")


@needs_node
def test_both_pages_boot_with_their_shared_script():
    """A page missing its <script src="/static/app.js"> renders as blank: the
    first top-level use of $ or esc is a ReferenceError, which halts the rest of
    the script. Nothing else in the suite sees it -- GET / still returns 200,
    and a missing dependency is not a syntax error. It shipped exactly once."""
    _run("pages_boot.mjs")
