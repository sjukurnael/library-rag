// Helpers shared by index.html (chat), library.html (the Drive browser) and
// queue.html (processing). Extracted when the second page arrived: `esc` in
// particular must have exactly one definition, because a second copy that
// drifts is an XSS hole nobody notices until a book title contains a bracket.

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

// Sizes arrive as megabytes; a folder rolls up to gigabytes fast enough that
// "3481.7 MB" would be the common case, so switch units at 1 GB.
function mb(n) {
  if (n == null) return 'size unknown';
  return n >= 1024 ? `${(n / 1024).toFixed(1)} GB` : `${n} MB`;
}

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

// ---------------------------------------------------------------- shell --
// Every page shares the sidebar: nav, the list of indexed books (what the
// index can actually answer from), and a one-line status. One poller feeds
// all of it; pages that need the same data (the queue dashboard, the chat
// status line) register a listener instead of fetching twice.

const bookListeners = [];
function onBooks(fn) { bookListeners.push(fn); }

let bookTimer = null;
async function refreshBooks() {
  clearTimeout(bookTimer);
  let d = null, err = null;
  try { d = await fetchJSON('/api/books', {}, 15000); }
  catch (e) { err = e; }
  for (const fn of bookListeners) fn(d, err);
  // Only a book actually moving through the pipeline justifies polling; one
  // sitting at a terminal status never changes on its own.
  const working = d && (d.pending || []).some(b => !TERMINAL.has(b.status));
  if (working || err) bookTimer = setTimeout(refreshBooks, working ? 3000 : 15000);
}

// The sidebar's own listener. Registered unconditionally: every page has the
// sidebar, and a page that somehow lacks it just no-ops on the null lookups.
onBooks((d, err) => {
  const list = $('#navbooks'), count = $('#bookcount'), foot = $('#sidefoot');
  if (!list) return;
  if (err) {
    if (foot) foot.textContent = 'index unavailable — is the server up?';
    return;
  }
  const working = (d.pending || []).filter(b => !TERMINAL.has(b.status));
  list.innerHTML =
    working.map(b =>
      `<li class="working" title="${esc(b.title)}"><span class="t">${esc(b.title)}</span>` +
      `<span class="m">${esc(STAGE[b.status] || b.status)}…</span></li>`
    ).join('') +
    d.books.map(b =>
      `<li title="${esc(b.title)} — ${b.pages ?? '?'} pages, ${b.chunks} chunks">` +
      `<span class="t">${esc(b.title)}</span><span class="m">${b.chunks}</span>` +
      `<button class="del" data-id="${b.id}" data-title="${esc(b.title)}" ` +
      `title="Remove this book">&times;</button></li>`
    ).join('');
  if (count) count.textContent = String(d.books.length);
  if (foot) foot.textContent =
    `${d.books.length} books · ${d.total_chunks.toLocaleString()} passages indexed`;
});

// The index list is a dropdown, and it ALWAYS starts collapsed -- the point is
// that the sidebar stays calm until the list is asked for. Deliberately not
// remembered across loads: a persisted "open" makes the list permanent again
// the moment anyone expands it once, which is exactly the state this exists
// to avoid.
const _booksToggle = $('#bookstoggle'), _booksList = $('#navbooks');
function setBooksOpen(open) {
  if (!_booksToggle || !_booksList) return;
  _booksList.hidden = !open;
  _booksToggle.classList.toggle('open', open);
  _booksToggle.setAttribute('aria-expanded', String(open));
}
if (_booksToggle && _booksList) {
  _booksToggle.addEventListener('click', () => setBooksOpen(_booksList.hidden));
}

// Delete is a hard delete of the row, its chunks and its files, so confirm by
// name rather than a bare "are you sure" that says nothing about what goes.
// Delegated at the document so the sidebar list and the queue page share it.
document.addEventListener('click', async e => {
  const btn = e.target.closest('.del');
  if (!btn || !btn.dataset.id) return;
  const { id, title } = btn.dataset;
  if (!confirm(`Remove "${title}" from the library?\n\nThis deletes its passages and the uploaded file. You can add it again later.`)) return;
  btn.disabled = true;
  try {
    await fetchJSON(`/api/books/${id}`, { method: 'DELETE' });
    refreshBooks();
  } catch (err) {
    btn.disabled = false;
    alert(`Could not remove "${title}" — ${err.message || err}`);
  }
});

// ----------------------------------------------------------- drive auth --
// Every page shows whether Google Drive is connected, bottom-left. Indexing a
// book downloads it from Drive, so an expired token is worth knowing about
// BEFORE a click fails -- and the fix (sign in) should be one click from
// anywhere, not something only the Drive page can do.

let _authConnecting = false;

async function paintAuth() {
  const el = $('#sideauth');
  if (!el) return;
  let a;
  try { a = await fetchJSON('/api/drive/auth/status', {}, 15000); }
  catch { return; }
  el.hidden = false;
  if (a.ok) {
    el.innerHTML = `<span class="authdot ok"></span> Google Drive connected`;
  } else if (a.reason === 'missing_client') {
    // No credentials.json: signing in can't work until the server is set up,
    // so a button would be a lie. Show why instead.
    el.innerHTML = `<span class="authdot bad"></span> Drive not configured`;
    el.title = a.detail || '';
  } else if (_authConnecting) {
    el.innerHTML = `<span class="authdot warn"></span> Waiting for Google — approve in the other tab…`;
  } else {
    el.innerHTML = `<button class="btn connectbtn" id="sideconnect">Sign in to Google Drive</button>`;
    el.title = '';
  }
  return a;
}

document.addEventListener('click', async e => {
  const btn = e.target.closest('#sideconnect');
  if (!btn || _authConnecting) return;
  // Opened SYNCHRONOUSLY, before any await: window.open after an await has
  // lost its user-gesture context and popup blockers eat it silently (same
  // hard-won lesson as the Drive page's Connect button).
  const tab = window.open('', '_blank');
  _authConnecting = true;
  paintAuth();
  try {
    const d = await fetchJSON('/api/drive/auth/start', {}, 15000);
    if (tab && !tab.closed) tab.location = d.url;
    // The callback lands in the OTHER tab; polling is how this one learns.
    const started = Date.now();
    const poll = setInterval(async () => {
      if (Date.now() - started > 300000) { _authConnecting = false; clearInterval(poll); paintAuth(); return; }
      const a = await fetchJSON('/api/drive/auth/status', {}, 15000).catch(() => null);
      if (a && a.ok) { _authConnecting = false; clearInterval(poll); paintAuth(); }
    }, 3000);
  } catch (err) {
    if (tab && !tab.closed) tab.close();
    _authConnecting = false;
    paintAuth();
    alert(`Could not start Google sign-in — ${err.message || err}`);
  }
});

paintAuth();

// ------------------------------------------------------------- librarian --
// The browse agent, shared by the Drive page (free-text interest) and the Bible
// page (a verse). Extracted here for the same reason esc() is: it was about to
// have a second copy, and a second copy is the one that drifts. The agent is
// identical either way -- /api/browse takes a free-text `interest`, and a verse
// is just a particularly well-specified one.

const TOOL_LABEL = {
  search_drive: 'search', browse_folder: 'browse',
  estimate_pipeline: 'estimate', recommend: 'shortlist',
};

function toolLine(e) {
  const i = e.input || {};
  const arg = i.query || i.name_contains || i.folder_id || '';
  return `<div class="bstep"><span class="lbl">${esc(TOOL_LABEL[e.name] || e.name)}</span>` +
         `${arg ? `“${esc(String(arg))}”` : ''}</div>`;
}

function resultLine(s) {
  const bits = [];
  if (s.returned != null) bits.push(`${s.returned} found`);
  if (s.total_pdfs != null && s.truncated) bits.push(`of ${s.total_pdfs}`);
  if (s.already_indexed) bits.push(`${s.already_indexed} already yours`);
  // "hit the limit" used to sit here and was shown on EVERY mirror search --
  // a KNN ranker always fills its page, so it announced a ceiling that was not
  // real. Only the live-Drive fallback can genuinely run short.
  if (s.hit_limit) bits.push('hit the limit');
  // The honest version of "is this topic actually in the collection": titles
  // both rankers agreed on. Only worth saying when it is zero -- that is the
  // case where a full-looking list of 50 is really just nearest neighbours.
  if (s.word_and_meaning === 0) bits.push('nothing matched on words — nearest by meaning only');
  if (s.picks != null) bits.push(`${s.picks} picked`);
  return bits.length
    ? `<div class="bstep"><span class="lbl"></span>${esc(bits.join(' · '))}</div>` : '';
}

function recCard(b, i) {
  // A pick whose file_id the agent never actually saw is shown as an error, not
  // rendered as a card -- a dead Drive link that looks real is worse than none.
  if (b.unknown) {
    return `<div class="rec bad"><div class="body"><div class="t">Unusable suggestion</div>` +
           `<div class="why">${esc(b.error)}</div></div></div>`;
  }
  const size = b.size_mb != null ? `${b.size_mb} MB` : 'size unknown';
  const action = b.indexed
    ? `<span class="badge good">Already yours</span>`
    : `<button class="btn add" data-i="${i}">Index</button>`;
  return `<div class="rec${b.indexed ? ' owned' : ''}">` +
    `<div class="body"><div class="t">${esc(b.title)}</div>` +
    `<div class="why">${esc(b.why || '')}</div>` +
    `<div class="m">${esc(size)} · <a href="${esc(b.url)}" target="_blank" rel="noopener">view in Drive</a></div>` +
    `</div>${action}</div>`;
}

/** Run the librarian, streaming its trail into trailEl and cards into outEl.
 *  Resolves to the recommendations, which the caller keeps so its Index button
 *  can look one up by index. */
async function runLibrarian(interest, trailEl, outEl) {
  let recs = [];
  const trail = [];
  trailEl.innerHTML = '<div class="bstep">looking…</div>';
  outEl.innerHTML = '';

  const r = await fetch('/api/browse', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ interest }),
  });
  if (!r.ok) throw new Error(r.statusText);

  await readEventStream(r, ev => {
    if (ev.type === 'tool') trail.push(toolLine(ev));
    else if (ev.type === 'results') trail.push(resultLine(ev.summary));
    else if (ev.type === 'tool_error') {
      trail.push(`<div class="bstep berr">${esc(ev.name)} failed — ${esc(ev.message)}</div>`);
    } else if (ev.type === 'recommendations') {
      recs = ev.books;
      outEl.innerHTML = recs.map(recCard).join('');
    } else if (ev.type === 'answer') {
      outEl.insertAdjacentHTML('beforeend',
        `<div class="sub" style="margin-top:10px">${esc(ev.text)}</div>`);
    } else if (ev.type === 'error') {
      outEl.insertAdjacentHTML('beforeend', `<div class="berr">${esc(ev.message)}</div>`);
    }
    trailEl.innerHTML = trail.join('');
  }, {
    terminal: 'done',
    onStall: secs => {
      trail.push(`<div class="bstep">no update for ${secs}s — still connected.</div>`);
      trailEl.innerHTML = trail.join('');
    },
  });
  return recs;
}

/** Wire the Index buttons inside `root`. `lookup(i)` returns the pick at index
 *  i -- passed as a function because the caller's recs array is replaced on
 *  every run, and a captured reference would go stale.
 *
 *  The agent recommends; this button acts. Nothing the agent does writes to the
 *  library -- adding a book is always an explicit choice made here.
 */
function wireIndexButtons(root, lookup) {
  root.addEventListener('click', async e => {
    const btn = e.target.closest('.add');
    if (!btn) return;
    const book = lookup(Number(btn.dataset.i));
    if (!book) return;
    btn.disabled = true;
    btn.textContent = 'Queueing…';
    try {
      const r = await fetch('/api/library/drive', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_ids: [book.file_id] }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || `failed (${r.status})`);
      btn.textContent = d.added.length ? 'Queued' : 'Already yours';
      // The queue page and the sidebar report the ingest from here on -- no
      // second progress mechanism needed.
      if (typeof refreshBooks === 'function') refreshBooks();
    } catch (err) {
      btn.disabled = false;
      btn.textContent = 'Retry';
      root.insertAdjacentHTML('beforeend',
        `<div class="berr">${esc(String(err.message || err))}</div>`);
    }
  });
}

// --------------------------------------------------------------- signed in --
// Who is using the app, bottom of the sidebar. Silent when sign-in is off, so
// local development looks exactly as it did before the gate existed.

async function paintUser() {
  const foot = $('#sidefoot');
  if (!foot) return;
  let me;
  try {
    me = await fetchJSON('/api/auth/me', {}, 10000);
  } catch { return; }          // never let the gate's chrome break a page
  if (!me.enabled || !me.email) return;
  foot.insertAdjacentHTML('beforebegin',
    `<div class="sideuser"><span class="who" title="${esc(me.email)}">${esc(me.email)}</span>` +
    `<a href="/logout">Sign out</a></div>`);
}

paintUser();
