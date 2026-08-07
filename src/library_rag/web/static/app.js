// Helpers shared by index.html (research) and library.html (the Drive browser).
// Extracted when the second page arrived: `esc` in particular must have exactly
// one definition, because a second copy that drifts is an XSS hole nobody
// notices until a book title contains a bracket.

const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"]/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

// `status` records the LAST COMPLETED stage, so the work actually in flight is
// the one AFTER it. Rendering the raw enum said "downloaded…" through the 40s of
// extraction and "discovered…" for a book nobody was working on -- which is why
// eight abandoned books looked identical to eight busy ones for three days.
const STAGE = {
  discovered: 'Downloading',
  downloaded: 'Reading pages',
  extracted:  'Splitting into passages',
  chunked:    'Generating embeddings',
};

// Statuses the pipeline never leaves on its own. Polling one is a request loop
// with no possible outcome, and calling it "in progress" is a lie.
const TERMINAL = new Set(['failed', 'needs_ocr']);

// Seconds since the worker last claimed a book. Silence is the thing worth
// surfacing: a stage running for 6m is how a stall becomes visible instead of
// looking like ordinary work.
function age(s) {
  if (s == null) return '';
  return s < 90 ? ` ${Math.round(s)}s` : ` ${Math.round(s / 60)}m`;
}

function mb(n) { return n == null ? 'size unknown' : `${n} MB`; }

// ------------------------------------------------------------- progress --
// Every wait in this app is long enough to be mistaken for a hang: a search
// embeds a query over the network, the agents run for tens of seconds, and a
// Drive sync takes minutes. A static "searching…" cannot distinguish working
// from stalled, so each one gets a live elapsed count and escalating wording.

// When the message changes, in seconds. Chosen against measured latencies: a
// title search is ~1s, a research run 20-40s, so 8s is already unusual for the
// first and ordinary for the second -- which is why the caller passes `normal`.
function progress(el, { label, normal = 8, hint = '' } = {}) {
  const t0 = Date.now();
  let timer = null;
  const tick = () => {
    const s = Math.round((Date.now() - t0) / 1000);
    let text = label;
    if (s >= normal * 4) {
      text = `${label} — ${s}s. This is much longer than usual; it may have stalled.`;
    } else if (s >= normal) {
      text = `${label} — ${s}s${hint ? '. ' + hint : ''}`;
    } else if (s >= 2) {
      text = `${label} ${s}s`;
    }
    el.innerHTML = `<div class="prog">${esc(text)}</div>`;
    timer = setTimeout(tick, 1000);
  };
  tick();
  return {
    stop: () => clearTimeout(timer),
    // Seconds elapsed, so a failure message can say how long it waited before
    // giving up rather than leaving the user to guess.
    seconds: () => Math.round((Date.now() - t0) / 1000),
  };
}

// fetch() has NO default timeout: a request the server never answers hangs
// forever and the page waits with it. Every non-streaming call goes through
// this instead so a dead backend surfaces as a message rather than a spinner.
async function fetchJSON(url, opts = {}, timeoutMs = 30000) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const r = await fetch(url, { ...opts, signal: ctrl.signal });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.detail || `${r.status} ${r.statusText}`);
    return d;
  } catch (e) {
    if (e.name === 'AbortError') {
      throw new Error(`No response after ${Math.round(timeoutMs / 1000)}s — ` +
                      `the server may be down or restarting.`);
    }
    throw e;
  } finally {
    clearTimeout(t);
  }
}

// Read a `data: {json}\n\n` event stream, calling onEvent for each frame.
// Both the research loop and the browse agent stream this shape; a second
// hand-rolled reader would be a second place for the framing to go wrong.
//
// `terminal` names the event that means "finished normally". Without it, a
// stream cut short -- server restart, dropped connection, a crash inside the
// generator after the response headers were already sent -- just ends the read
// loop. No exception is raised, so `catch` never runs and the page keeps
// whatever half-drawn trail it had, silently, forever. Ending without the
// terminal event is a failure and is reported as one.
//
// `onStall` fires when no frame has arrived for `stallMs`. Not a timeout: these
// runs are legitimately long, and killing one that is merely slow is worse than
// waiting. It only changes what the user is told.
async function readEventStream(response, onEvent, opts = {}) {
  const { terminal = null, onStall = null, stallMs = 45000 } = opts;
  const reader = response.body.getReader();
  const dec = new TextDecoder();
  let buf = '', sawTerminal = false, stallTimer = null;

  const arm = () => {
    clearTimeout(stallTimer);
    if (onStall) stallTimer = setTimeout(() => onStall(stallMs / 1000), stallMs);
  };
  arm();
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split('\n\n');
      buf = parts.pop();
      for (const p of parts) {
        if (!p.startsWith('data: ')) continue;
        arm();
        const ev = JSON.parse(p.slice(6));
        if (terminal && ev.type === terminal) sawTerminal = true;
        onEvent(ev);
      }
    }
  } finally {
    clearTimeout(stallTimer);
  }
  if (terminal && !sawTerminal) {
    throw new Error('The connection ended before the run finished — the server ' +
                    'may have restarted. Nothing was lost; try again.');
  }
}
