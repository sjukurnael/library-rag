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

// What to SHOW for a book, which is not the same as its status.
//
// `discovered` means "queued", and the label above assumes a worker is about to
// pick it up -- true when the only way to queue something was on the machine
// that also ran the worker. Deployed, the click and the worker are different
// computers, and a row nobody has claimed rendered as "Downloading…" forever.
//
// claim_age_s is NULL until a worker claims the book, so it distinguishes the
// two exactly. An unclaimed row says so instead of animating a stage that is
// not happening.
function stageLabel(b) {
  if (b.status === 'discovered' && b.claim_age_s == null) return 'Queued';
  return STAGE[b.status] || b.status;
}

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

// Raw byte counts, which the mirror's totals arrive as. Separate from mb()
// rather than dividing at the call site: these span half a megabyte to 143 GB,
// so the unit has to be picked per value, and picking it here is what stops
// one line reading "506 MB" and the line below it "0.5 GB".
function bytes(n) {
  if (n == null) return 'size unknown';
  // Bare "0", not "0 KB". This reads inside "0 of 142.8 GB", where a unit on
  // the empty side is noise attached to nothing.
  if (!n) return '0';
  if (n >= 1073741824) return `${(n / 1073741824).toFixed(1)} GB`;
  if (n >= 1048576) return `${Math.round(n / 1048576)} MB`;
  return `${Math.round(n / 1024)} KB`;
}

// A share of a whole, worded so it can never claim more than it has.
//
// Both guards exist because this app's real numbers sit at the ends of the
// range: 0.35% would print as "0%" and read as "nothing has been indexed" when
// 199 books have, and a whole drive bar one book would round to a finished
// "100%". Neither rounding is wrong by much and both are wrong about the thing
// the reader is actually asking.
function pct(part, whole) {
  if (!whole || !part) return '0%';
  const p = (part / whole) * 100;
  if (p < 0.1) return '<0.1%';
  if (p >= 99.95 && part < whole) return '99.9%';
  return `${p.toFixed(p < 10 ? 1 : 0)}%`;
}

// --------------------------------------------------------------- failure --
// A dead end with no way forward is what makes a page feel broken rather than
// merely slow, so a failure states what broke and offers the action that fixes
// it.
//
// Returns HTML rather than writing into an element, because callers put it in
// very different places -- innerHTML of a trail, insertAdjacentHTML at the top
// of a list. It lived in library.html for a while and the classroom pages
// called it anyway: six call sites, all of them a ReferenceError the moment
// anything failed, in the one code path whose entire job is to report failure.
//
// The retry callback cannot survive a trip through a string, so it is parked in
// a registry and reached by id from one delegated listener.
const _retries = new Map();
let _retryId = 0;

function failed(message, err, retry) {
  const detail = err ? String(err.message || err) : '';
  let attr = '';
  if (retry) { _retries.set(++_retryId, retry); attr = ` data-retry="${_retryId}"`; }
  return `<div class="failnote"><span class="fmsg">${esc(message)}` +
         `${detail ? ' — ' + esc(detail) : ''}</span>` +
         (retry ? `<button class="retry"${attr}>Try again</button>` : '') +
         `</div>`;
}

document.addEventListener('click', e => {
  const b = e.target.closest('.retry[data-retry]');
  if (!b) return;
  const fn = _retries.get(Number(b.dataset.retry));
  if (fn) fn();
});

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
// But a stream that ends WITHOUT the terminal event has usually said why first.
// Once the response headers are out the server cannot answer with a 500, so a
// crash mid-run arrives as an `error` frame and then the stream stops -- both
// halves of the same event. Reporting only the second half is how a flat
// Anthropic balance came to read as "the server may have restarted" for a week:
// the real reason was sent, rendered, and then overwritten by the guess. So the
// last `error` message wins, and the generic sentence below is the last resort
// it was meant to be -- for a connection that died with nothing to say.
//
// `onStall` fires when no frame has arrived for `stallMs`. Not a timeout: these
// runs are legitimately long, and killing one that is merely slow is worse than
// waiting. It only changes what the user is told.
async function readEventStream(response, onEvent, opts = {}) {
  const { terminal = null, onStall = null, stallMs = 45000 } = opts;
  const reader = response.body.getReader();
  const dec = new TextDecoder();
  let buf = '', sawTerminal = false, stallTimer = null, reason = null;

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
        // Captured before onEvent, which is allowed to throw on this very
        // frame (the chat page does) -- and then this line would never run.
        if (ev.type === 'error' && ev.message) reason = String(ev.message);
        onEvent(ev);
      }
    }
  } finally {
    clearTimeout(stallTimer);
  }
  if (terminal && !sawTerminal) {
    throw new Error(reason ||
      'The connection ended before the run finished — the server may have ' +
      'restarted. Nothing was lost; try again.');
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
// How much of the indexed list to ask for. Raised by "show more" rather than
// paged with an offset: the list is sorted by title and re-fetched every few
// seconds while work is in flight, so an offset window would shuffle under the
// reader as books finish and join it.
let booksLimit = 50;
function showMoreBooks(n = 50) { booksLimit += n; return refreshBooks(); }

async function refreshBooks() {
  clearTimeout(bookTimer);
  let d = null, err = null;
  try { d = await fetchJSON(`/api/books?limit=${booksLimit}`, {}, 20000); }
  catch (e) { err = e; }
  for (const fn of bookListeners) fn(d, err);
  // Only a book actually moving through the pipeline justifies polling; one
  // sitting at a terminal status never changes on its own.
  const working = d && (d.pending || []).some(b => !TERMINAL.has(b.status));
  if (working || err) bookTimer = setTimeout(refreshBooks, working ? 3000 : 15000);
}

// The sidebar footer. All that survives of the sidebar's book feed: the list
// it used to render was every book in the library -- 8,679 rows behind a
// dropdown -- which was unreadable long before it was expensive, and expensive
// enough (1.1 MB, ~10s per fetch) to make the pages carrying it feel broken.
//
// Only queue.html still calls refreshBooks(), because the queue IS this data.
// Everywhere else the footer comes from paintLibraryFoot() and 258 bytes.
onBooks((d, err) => {
  const foot = $('#sidefoot');
  if (!foot) return;
  foot.textContent = err
    ? 'index unavailable — is the server up?'
    : `${d.total_books.toLocaleString()} books · ${d.total_chunks.toLocaleString()} passages indexed`;
});

// Delete is a hard delete of the row, its chunks and its files, so confirm by
// name rather than a bare "are you sure" that says nothing about what goes.
// Delegated at the document so the sidebar list and the queue page share it.
document.addEventListener('click', async e => {
  // [data-title], not bare .del: a classroom's Delete button is also a .del
  // with a data-id, so this handler used to fire alongside it and offer to
  // delete the BOOK whose id matched the classroom's. Book buttons are the
  // ones that carry a title; classroom buttons carry data-name.
  const btn = e.target.closest('.del[data-title]');
  if (!btn || !btn.dataset.id) return;
  const { id, title } = btn.dataset;
  const ok = await confirmDialog({
    title: `Remove "${title}"?`,
    note: 'This deletes its passages and the uploaded file, and takes it off '
        + 'every classroom it is on. You can add it again later.',
    confirm: 'Remove book',
  });
  if (!ok) return;
  btn.disabled = true;
  try {
    await fetchJSON(`/api/books/${id}`, { method: 'DELETE' });
    refreshBooks();
  } catch (err) {
    btn.disabled = false;
    await openDialog({ title: `Could not remove "${title}"`,
                       note: String(err.message || err),
                       fields: [], confirm: 'OK', cancel: 'Close' });
  }
});

// ----------------------------------------------------------- drive auth --
// Every page shows whether Google Drive is connected, bottom-left. Indexing a
// book downloads it from Drive, so an expired token is worth knowing about
// BEFORE a click fails -- and the fix (sign in) should be one click from
// anywhere, not something only the Drive page can do.

let _authConnecting = false;
// Whether Drive is reachable, so UI that depends on it can say so up front.
let _driveOk = false;

async function paintAuth() {
  let a;
  try { a = await fetchJSON('/api/drive/auth/status', {}, 15000); }
  catch { return; }
  // Assign before painting, and before the missing-element bail below. Every
  // page calls paintAuth() at boot, so this is what keeps _driveOk true;
  // leaving it at its `false` initialiser disabled every Index button on the
  // librarian panel even with Drive connected. The bail has to come after,
  // because bible.html renders librarian picks without a #sideauth indicator
  // -- returning early there would keep its buttons dead for the same reason.
  _driveOk = !!a.ok;
  const el = $('#sideauth');
  if (!el) return a;
  el.hidden = false;
  if (a.ok) {
    el.innerHTML = `<span class="authdot ok"></span> Google Drive connected`;
  } else if (a.reason === 'cannot_connect') {
    // No OAuth client configured here, so a Connect button would be a lie.
    // Deliberately NOT the red "bad" dot: on the deployed app this is the
    // intended configuration rather than a fault, and a red light beside a
    // working page reads as something broken that the reader cannot fix.
    el.innerHTML = `<span class="authdot"></span> Drive read-only here`;
    el.title = 'Browsing and search use the local mirror. Indexing runs elsewhere.';
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
// identical either way -- /api/librarian takes a free-text brief, and a verse
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
  // _driveOk is refreshed by paintAuth() on every page. Without Drive the
  // Index endpoint raises before it queues anything, so an enabled button is a
  // promise the server cannot keep -- and a 503 on click is a worse way to
  // learn that than a disabled button that says so.
  const action = b.indexed
    ? `<span class="badge good">Already yours</span>`
    : _driveOk
      ? `<button class="btn add" data-i="${i}">Index</button>`
      : `<button class="btn add" disabled title="Indexing needs Google Drive, `
        + `which is not connected on this server.">Index</button>`;
  return `<div class="rec${b.indexed ? ' owned' : ''}">` +
    `<div class="body"><div class="t">${esc(b.title)}</div>` +
    `<div class="why">${esc(b.why || '')}</div>` +
    `<div class="m">${esc(size)} · <a href="${esc(b.url)}" target="_blank" rel="noopener">view in Drive</a></div>` +
    `</div>${action}</div>`;
}

/** Run the librarian, streaming its trail into trailEl and cards into outEl.
 *  Resolves to the recommendations, which the caller keeps so its Index button
 *  can look one up by index.
 *
 *  `count` is the MOST books to shortlist, not how many come back -- the agent
 *  returns fewer when the collection runs out of relevant titles, which is the
 *  intended behaviour and not an error to surface.
 *
 *  Omitted, the field is left off the request entirely rather than defaulted
 *  here, so the default lives in exactly one place (LibrarianRequest.count).
 *  The Bible page, which digs on a single verse and wants a handful, calls this
 *  with three arguments and gets that default.
 *
 *  Points at /api/librarian. This used to drive the browsing agent, which could
 *  only ever see filenames; that agent is gone and the librarian reads the books
 *  themselves. The event contract is the one the old loop yielded -- tool /
 *  results / tool_error / recommendations / answer / done -- which is why an
 *  endpoint and a field name were the whole change. */
async function runLibrarian(interest, trailEl, outEl, count) {
  let recs = [];
  const trail = [];
  trailEl.innerHTML = '<div class="bstep">looking…</div>';
  outEl.innerHTML = '';

  const r = await fetch('/api/librarian', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(count ? { brief: interest, count } : { brief: interest }),
  });
  // A 422 here is the count field failing its 1-50 bound. Worth reading the
  // body for: "Unprocessable Entity" alone tells the user nothing about which
  // field or what the limit is.
  if (!r.ok) {
    const detail = await r.json().catch(() => null);
    const msg = detail?.detail?.[0]?.msg || detail?.detail;
    throw new Error(msg ? `${r.statusText} — ${msg}` : r.statusText);
  }

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


// -------------------------------------------------------------- dialogs --
// One in-app dialog for asking a question and for confirming something you
// cannot undo. Replaces window.prompt and window.confirm, which are OS chrome
// in the wrong typeface, cannot be labelled, cannot explain what they are
// asking for, and cannot refuse an empty answer -- prompt() returns a string or
// null and has no way to say "that one is needed" without opening a second box.
//
// Resolves to an object of field values, or null if dismissed. Building the DOM
// here rather than in every page keeps it on pages that never declared it.

function openDialog({ title, note, fields = [], confirm = 'Save',
                      cancel = 'Cancel', danger = false, onChange = null }) {
  return new Promise(resolve => {
    const wrap = document.createElement('div');
    wrap.className = 'modal';
    wrap.innerHTML = `
      <div class="modalbox sm" role="dialog" aria-modal="true">
        <form class="dlg" novalidate>
          <div class="dlgtitle">${esc(title)}</div>
          ${note ? `<div class="dlgnote">${esc(note)}</div>` : ''}
          ${fields.map(f => `<label class="field">
              <span class="lab">${esc(f.label)}</span>
              ${f.choices
                ? `<select name="${esc(f.name)}">${f.choices.map(([v, t]) =>
                     `<option value="${esc(v)}"${String(v) === String(f.value) ? ' selected' : ''}>${esc(t)}</option>`
                   ).join('')}</select>`
                : f.multiline
                  ? `<textarea name="${esc(f.name)}" rows="3"
                       placeholder="${esc(f.placeholder || '')}">${esc(f.value || '')}</textarea>`
                  : `<input name="${esc(f.name)}" autocomplete="off"
                       placeholder="${esc(f.placeholder || '')}" value="${esc(f.value || '')}">`}
              ${f.hint ? `<span class="hint" data-for="${esc(f.name)}">${esc(f.hint)}</span>` : ''}
            </label>`).join('')}
          <div class="dlgfoot">
            <button type="button" class="btn ghost cancel">${esc(cancel)}</button>
            <button type="submit" class="btn ${danger ? 'danger' : 'primary'}">${esc(confirm)}</button>
          </div>
        </form>
      </div>`;
    document.body.appendChild(wrap);

    const form = wrap.querySelector('form');
    if (onChange) {
      wrap.addEventListener('change', () => {
        const cur = {};
        for (const f of fields) {
          const el = form.elements[f.name];
          cur[f.name] = el ? el.value : '';
        }
        onChange(cur, wrap);
      });
      // Paint once up front, so the dialog opens saying the right thing.
      queueMicrotask(() => wrap.dispatchEvent(new Event('change')));
    }

    const first = wrap.querySelector('input, textarea, select');
    // Focus, and put the caret after any existing value rather than selecting
    // it: this dialog renames as often as it creates, and a pre-selected name
    // is one keystroke from being destroyed.
    if (first) {
      first.focus();
      if (first.value && first.setSelectionRange)
        first.setSelectionRange(first.value.length, first.value.length);
    } else {
      wrap.querySelector('.btn.primary, .btn.danger').focus();
    }

    let done = false;
    const close = value => {
      if (done) return;
      done = true;
      document.removeEventListener('keydown', onKey, true);
      wrap.remove();
      resolve(value);
    };
    function onKey(e) {
      if (e.key === 'Escape') { e.stopPropagation(); close(null); }
      // Enter in a textarea is a newline; Cmd/Ctrl-Enter submits from anywhere.
      if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) form.requestSubmit();
    }
    // Capture, so Escape closes this and not something behind it.
    document.addEventListener('keydown', onKey, true);
    wrap.addEventListener('click', e => { if (e.target === wrap) close(null); });
    wrap.querySelector('.cancel').addEventListener('click', () => close(null));

    form.addEventListener('submit', e => {
      e.preventDefault();
      const out = {};
      for (const f of fields) {
        const el = form.elements[f.name];
        out[f.name] = (el ? el.value : '').trim();
      }
      // A required field that is empty keeps the dialog open and says so, which
      // is the one thing prompt() could not do at all.
      const missing = fields.find(f => f.required && !out[f.name]);
      if (missing) {
        const el = form.elements[missing.name];
        el.focus();
        el.closest('.field').querySelector('.hint')?.remove();
        el.insertAdjacentHTML('afterend',
          `<span class="hint err">${esc(missing.needed || 'This one is needed.')}</span>`);
        return;
      }
      close(out);
    });
  });
}

// A yes/no with no fields. Separate name because the call reads better and the
// destructive case should be obvious at the call site.
function confirmDialog({ title, note, confirm = 'Delete', danger = true }) {
  return openDialog({ title, note, fields: [], confirm, danger })
    .then(r => r !== null);
}

// ------------------------------------------------------ the library foot --
// The "N books" line in the sidebar footer, for pages that do NOT list books.
//
// It used to come from refreshBooks(), which fetches every book in the library:
// 8,679 rows, 1.1 MB, 9.7 seconds -- to render two numbers. Worse, that poll
// repeats every 3 seconds while anything is being processed, and 353 books are,
// so a classroom page left open issued a continuous stream of 1.1 MB requests
// for a footer with no list under it. When one of them passed the 15s timeout
// the footer read "index unavailable — is the server up?", which is how a
// working server came to accuse itself.
//
// /api/drive/sync/progress answers the same question in 258 bytes and 0.47s.
async function paintLibraryFoot() {
  const foot = $('#sidefoot');
  if (!foot) return;
  try {
    const s = await fetchJSON('/api/drive/sync/progress', {}, 15000);
    foot.textContent = `${(s.indexed_files || 0).toLocaleString()} books ready`
      + (s.working_files ? ` · ${s.working_files.toLocaleString()} arriving` : '');
  } catch {
    // Silent. This is a footnote; a page whose real work is fine should not be
    // captioned with an error about a number nobody asked for.
    foot.textContent = '';
  }
}

// ------------------------------------------------------- classroom list --
// The sidebar list, shared by the classrooms page and by each classroom. A
// classroom is a conversation, so the list of them belongs where a chat app
// keeps its conversations: down the left, always visible, one click apart.

async function paintRoomList(currentId) {
  const el = $('#roomlist');
  if (!el) return null;
  try {
    const d = await fetchJSON('/api/classrooms', {}, 15000);
    el.innerHTML = d.classrooms.length
      ? d.classrooms.map(c => {
          // A shelf with nothing on it cannot answer anything, so it says so
          // rather than showing a 0 that reads like any other count.
          const ct = c.book_count ? String(c.book_count) : '—';
          return `<li><a href="/classroom/${c.id}"` +
                 `${c.id === currentId ? ' class="on"' : ''}` +
                 ` title="${esc(c.name)}"><span class="nm">${esc(c.name)}</span>` +
                 `<span class="ct">${ct}</span></a></li>`;
        }).join('')
      : '<li class="none">No classrooms yet. Make one to start studying.</li>';
    return d;
  } catch (e) {
    el.innerHTML = `<li class="none">Could not load your classrooms.</li>`;
    return null;
  }
}

// Named, then opened. Creating one and leaving the reader on the list is a
// dead end -- the next thing anyone wants is to put books on it.
async function newClassroom() {
  // Name only. A classroom needs a label you can pick out of a sidebar, and
  // nothing more: the `brief` column is display-only -- the librarian reads the
  // topic you type into ITS box, never the classroom's stored description -- so
  // a second field here asked for something no code went on to use.
  const r = await openDialog({
    title: 'New classroom',
    note: 'A classroom is a small shelf of books. Every answer comes from those '
        + 'books and nothing else.',
    confirm: 'Create classroom',
    fields: [
      { name: 'name', label: 'What are you studying?', required: true,
        needed: 'Give it a name so you can find it again.',
        placeholder: 'The atonement' },
    ],
  });
  if (!r) return;
  try {
    const d = await fetchJSON('/api/classrooms', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: r.name }),
    }, 15000);
    location.href = `/classroom/${d.classroom.id}`;
  } catch (e) {
    await openDialog({ title: 'Could not create it', note: String(e.message || e),
                       fields: [], confirm: 'OK', cancel: 'Close' });
  }
}


// ------------------------------------------------------ classroom picker --

/* "Add to classroom" as a dialog, not a <select>.
 *
 * A native dropdown was the first thing here and it was wrong twice over: it
 * looks like 2005, and it shows a bare list of names when what a reader needs
 * to choose between shelves is how big each one is and which one they were just
 * looking at. A dialog has room for both.
 *
 * Shared here rather than duplicated per page because two pages ask the same
 * question -- the run's book card and the indexed list -- and a picker that
 * drifts between them is worse than either.
 */
let _roomCache = null;

async function _rooms(force) {
  if (!_roomCache || force) {
    const d = await fetchJSON('/api/classrooms', {}, 15000);
    _roomCache = d.classrooms || [];
  }
  return _roomCache;
}

function _pickerEl() {
  let el = $('#roompick');
  if (el) return el;
  el = document.createElement('div');
  el.className = 'modal';
  el.id = 'roompick';
  el.hidden = true;
  el.innerHTML = `
    <div class="modalbox pick" role="dialog" aria-modal="true" aria-labelledby="rpt">
      <div class="modalhead">
        <span class="t" id="rpt">Add to classroom</span>
        <button class="x" id="rpx" title="Close">&times;</button>
      </div>
      <div class="rpsub" id="rpsub"></div>
      <div class="rplist" id="rplist"></div>
    </div>`;
  document.body.appendChild(el);
  const close = () => { el.hidden = true; };
  el.addEventListener('click', e => { if (e.target === el) close(); });
  el.querySelector('#rpx').addEventListener('click', close);
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !el.hidden) close();
  });
  return el;
}

/* Open the picker for one book. `preferId` floats a shelf to the top and marks
 * it -- the run's own classroom, where the book is nearly always going. */
async function pickClassroom({ bookId, title, preferId = null,
                               addedBy = 'reader', rationale = null,
                               onAdded = null }) {
  const el = _pickerEl();
  const list = el.querySelector('#rplist');
  el.querySelector('#rpsub').textContent = title || '';
  list.innerHTML = '<div class="prog">loading…</div>';
  el.hidden = false;

  let rooms;
  try {
    rooms = await _rooms();
  } catch (err) {
    list.innerHTML = failed('Could not load your classrooms', err,
                            () => pickClassroom({ bookId, title, preferId, addedBy, rationale, onAdded }));
    return;
  }

  if (!rooms.length) {
    list.innerHTML = `<div class="empty">No classrooms yet.
      <a href="/">Make one</a> and it will show up here.</div>`;
    return;
  }

  const order = [...rooms].sort((a, b) =>
    (b.id === preferId) - (a.id === preferId) || a.name.localeCompare(b.name));

  list.innerHTML = order.map(r => `
    <button class="rprow" data-room="${r.id}">
      <span class="rpn">${esc(r.name)}</span>
      <span class="rpm">${r.book_count} book${r.book_count === 1 ? '' : 's'}${
        r.id === preferId ? ' · <b>this run</b>' : ''}</span>
      <span class="rpgo">Add</span>
    </button>`).join('');

  list.querySelectorAll('.rprow').forEach(btn => {
    btn.addEventListener('click', async () => {
      const roomId = Number(btn.dataset.room);
      const go = btn.querySelector('.rpgo');
      list.querySelectorAll('.rprow').forEach(b => b.disabled = true);
      go.textContent = 'Adding…';
      try {
        const r = await fetch(`/api/classrooms/${roomId}/books`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ book_ids: [Number(bookId)], added_by: addedBy,
                                 rationale: rationale || null }),
        });
        if (!r.ok) {
          const d = await r.json().catch(() => ({}));
          throw new Error(d.detail || r.statusText);
        }
        go.textContent = 'Added ✓';
        btn.classList.add('ok');
        _roomCache = null;              // the count on that shelf just changed
        if (onAdded) onAdded(roomId, order.find(x => x.id === roomId));
        setTimeout(() => { el.hidden = true; }, 700);
      } catch (err) {
        go.textContent = 'Add';
        list.querySelectorAll('.rprow').forEach(b => b.disabled = false);
        btn.insertAdjacentHTML('afterend',
          `<div class="rperr">${esc(String(err.message || err))}</div>`);
      }
    });
  });
}
