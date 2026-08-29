/* The tutor: the conversation you have with one classroom's books.
 *
 * Lifted almost unchanged out of the old global chat page, which is gone. Three
 * things are different, and all three follow from the shelf:
 *
 *   - every question carries its classroom_id, because there is no longer a
 *     "search everything" to fall back on;
 *   - the transcript is stored per classroom, so switching rooms does not drag
 *     one conversation into another;
 *   - `suggestions` renders add-to-shelf cards rather than sources. The tutor
 *     may name a book it has not read, and those must never look citable.
 *
 * The conversation is DATA first, pixels second: every turn is stored as
 * {question, run_id, status, events}, and all rendering is derived by replaying
 * those raw events through applyEvent -- the same function that handles them
 * live. That is what lets a reload rebuild everything, and a still-running turn
 * re-attach to its server-side run and continue where the buffer ends.
 *
 * Depends on classroom.html having already defined CID, SHELF and addBooks().
 */

function md(text, citeFn) {
  const blocks = esc(text).trim().split(/\n{2,}/);
  const inline = t => {
    // Non-greedy, and NOT [^*]+ -- that form fails on "**bold with *italic*
    // inside**" and leaves the raw ** on the page. Bold runs before italic so
    // the inner single asterisks are still there to match on the next pass.
    t = t.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
         .replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s.,;:)]|$)/g, '$1<em>$2</em>')
         .replace(/`([^`]+)`/g, '<code>$1</code>')
         .replace(/&quot;([^&]{1,200}?)&quot;/g, '“$1”');
    return citeFn ? citeFn(t) : t;
  };
  return blocks.map(b => {
    b = b.trim();
    const h = b.match(/^(#{1,6})\s+(.*)$/s);
    if (h) {
      const lvl = Math.min(h[1].length + 2, 6);
      return `<h${lvl} class="mdh">${inline(h[2].trim())}</h${lvl}>`;
    }
    if (/^([-*+]|\d+[.)])\s/.test(b)) {
      const ordered = /^\d/.test(b);
      const items = b.split(/\n(?=([-*+]|\d+[.)])\s)/)
                     .filter(x => x && /^([-*+]|\d+[.)])\s/.test(x));
      const li = items.map(i =>
        `<li>${inline(i.replace(/^([-*+]|\d+[.)])\s+/, '').trim())}</li>`).join('');
      return ordered ? `<ol>${li}</ol>` : `<ul>${li}</ul>`;
    }
    if (/^&gt;\s/.test(b)) {
      return `<blockquote>${inline(b.replace(/^&gt;\s?/gm, ''))}</blockquote>`;
    }
    return `<p>${inline(b).replace(/\n/g, '<br>')}</p>`;
  }).join('');
}

// Sources are PER TURN: two questions can both have a source [1], and each
// citation must resolve to the passage its own run was given.
const TURNS = [];

function showSource(ti, n) {
  const s = (TURNS[ti] || {})[n];
  if (!s) return;
  document.querySelectorAll('.answer cite.on').forEach(c => c.classList.remove('on'));
  const el = document.querySelector(`.answer cite[data-t="${ti}"][data-n="${n}"]`);
  if (el) el.classList.add('on');

  $('#ptitle').textContent = s.book;
  const rows = [
    ['Section', s.heading_trail ? s.heading_trail.replace(/\*\*/g, '') : '—'],
    ['Pages', s.pages + (s.book_pages ? ` (book has ${s.book_pages})` : '')],
    ['Passage', s.ordinal != null ? `${s.ordinal + 1} of ${s.total_chunks}` : '—'],
    ['Length', s.tokens != null ? `${s.tokens} tokens` : '—'],
    ['Distance', `${s.distance}${s.distance <= 0.70 ? '' : ' (weak match)'}`],
    ['Citation', `[${s.n}]`],
    ['Chunk id', s.chunk_id],
    ['Book id', s.book_id],
  ];
  $('#pmeta').innerHTML = rows
    .map(([k, v]) => `<dt>${k}</dt><dd>${esc(String(v))}</dd>`).join('');
  $('#praw').textContent = s.content;

  // Two links, because they answer different questions and neither can be the
  // other. Our mirrored copy takes #page= and lands on the sentence that was
  // cited; Drive's viewer ignores page anchors, so a Drive link always opens at
  // page one. But a reader who wants THE BOOK -- the whole thing, in the folder
  // it lives in, with everything shelved around it -- wants Drive, not our copy.
  //
  // The Drive link is offered whenever there is a book, not only when there is a
  // page: /api/books/{id}/source redirects to the Drive original and falls back
  // to our bytes only for the four uploads, which have no Drive file to open.
  const pdf = $('#ppdf');
  if (s.book_id != null) {
    const atPage = s.page_start != null
      ? `<a class="alt" href="/api/books/${s.book_id}/pdf#page=${s.page_start}" ` +
        `target="_blank" rel="noopener" ` +
        `title="Our mirrored copy, which is the only one that can open at a ` +
        `given page -- Drive's viewer ignores page anchors">` +
        `or jump to p.${s.page_start} in our copy ↗</a>`
      : '';
    pdf.innerHTML =
      `<a href="/api/books/${s.book_id}/source" target="_blank" rel="noopener" ` +
      `title="The whole original in Google Drive, in the folder it lives in">` +
      `Open the PDF in Drive ↗</a>` + atPage;
    pdf.hidden = false;
  } else {
    pdf.hidden = true;
  }
  $('#panel').classList.add('show');
}

function hidePanel() {
  $('#panel').classList.remove('show');
  document.querySelectorAll('.answer cite.on').forEach(c => c.classList.remove('on'));
}

$('#pclose').addEventListener('click', hidePanel);
document.addEventListener('keydown', e => { if (e.key === 'Escape') hidePanel(); });
document.addEventListener('click', e => {
  const cite = e.target.closest('.answer cite');
  if (cite) { showSource(cite.dataset.t, cite.dataset.n); e.stopPropagation(); return; }
  if (!e.target.closest('#panel')) hidePanel();
});

const citeify = (text, ti) => md(text, html =>
  html.replace(/\[(\d+)\]/g, (m, n) => (TURNS[ti] || {})[n]
    ? `<cite data-t="${ti}" data-n="${n}" title="${esc(TURNS[ti][n].book)} — ${esc(TURNS[ti][n].pages)}">${n}</cite>`
    : m));

// Keyed by classroom. Two shelves are two conversations, and merging them would
// put answers from one set of books under questions asked of another.
// sessionStorage, not localStorage: a closed tab is a finished conversation,
// while the SHELF it was about is durable and lives in Postgres.
const CHAT_KEY = `chat-v1:${CID}`;
const CHAT = (() => {
  try { return JSON.parse(sessionStorage.getItem(CHAT_KEY)) || { turns: [] }; }
  catch { return { turns: [] }; }
})();
if (!Array.isArray(CHAT.turns)) CHAT.turns = [];
function save() {
  try { sessionStorage.setItem(CHAT_KEY, JSON.stringify(CHAT)); }
  catch { /* quota -- display still works */ }
}

const LIVE = new Set();
const scrollDown = () => { const m = $('#msgs'); m.scrollTop = m.scrollHeight; };

// Shown when the classroom has no conversation yet. It changes with the shelf:
// an empty shelf cannot answer anything, so it points at the librarian instead
// of inviting a question that would come back with nothing.
function paintChatBlank() {
  const m = $('#msgs');
  if (!m || m.querySelector('.turn')) return;
  const empty = typeof SHELF !== 'undefined' && !SHELF.length;
  m.innerHTML = `<div class="blank">
      <div class="mark">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"
             stroke-linecap="round" stroke-linejoin="round">
          <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
          <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
        </svg>
      </div>
      <div class="t">${empty ? 'This shelf is empty' : 'Ask about these books'}</div>
      <div class="d">${empty
        ? 'Add a few books first — the tutor answers only from what is on the shelf.'
        : 'Every answer comes from the books on this shelf, with the passage it came from.'}</div>
    </div>`;
}

function addTurn(question) {
  const ti = TURNS.length;
  TURNS.push({});
  const blank = $('#msgs').querySelector('.blank');
  if (blank) blank.remove();
  $('#msgs').insertAdjacentHTML('beforeend',
    `<div class="turn" id="turn-${ti}">
       <div class="msg user">${esc(question)}</div>
       <div class="msg bot">
         <details class="trace"><summary id="tsum-${ti}">Reading…</summary>
           <div class="trace-body" id="trace-${ti}"></div>
         </details>
         <div class="answer" id="ans-${ti}"></div>
         <div class="suggests" id="sug-${ti}"></div>
       </div>
     </div>`);
  scrollDown();
  return ti;
}

function turnCtx(ti) {
  return {
    ti, steps: [], searches: 0, rawAnswer: '', suggestions: [],
    sum: $(`#tsum-${ti}`), trace: $(`#trace-${ti}`), ans: $(`#ans-${ti}`),
    sug: $(`#sug-${ti}`),
    paint() { this.trace.innerHTML = this.steps.map(s => s.html).join(''); },
  };
}

// Books the tutor named but has NOT read. Rendered under the answer as an
// offer, never as a source: they carry no chunk id, so there is nothing for a
// citation to point at and nothing to open in the panel.
function paintSuggestions(ctx) {
  if (!ctx.suggestions.length) { ctx.sug.innerHTML = ''; return; }
  ctx.sug.innerHTML =
    `<div class="sughead">Not on this shelf</div>` +
    ctx.suggestions.map((b, i) => {
      const on = (typeof SHELF !== 'undefined') &&
                 SHELF.some(s => s.book_id === b.book_id);
      // The same link the librarian's cards carry. It matters more here: the
      // tutor has NOT read these books and can only report their headings, so
      // opening the PDF is the only way to check before putting one on a shelf.
      const open = b.book_id
        ? `<a class="openpdf" href="/api/books/${b.book_id}/source" target="_blank"
              rel="noopener" title="Open the original in Google Drive">Open in Drive ↗</a>`
        : '';
      return `<div class="pick${on ? ' owned' : ''}">
        <div class="t">${esc(b.title || '')}</div>
        ${b.covers ? `<div class="why">${esc(b.covers)}</div>` : ''}
        <div class="foot">${b.pages ? b.pages + ' pages' : ''}${open}${
          on ? '<span class="badge good">on this shelf</span>'
             : `<button class="btn primary sadd" data-t="${ctx.ti}" data-i="${i}">Add</button>`
        }</div></div>`;
    }).join('');
}

const SUGGESTED = {};   // turn index -> the books that turn suggested

// Every turn's cards repainted at once. The same book can be suggested by more
// than one turn, and leaving the other copies offering "Add" invites a click
// that silently does nothing. Called by classroom.html whenever the shelf moves
// -- including removals, which the old code did not repaint at all.
function repaintSuggestions() {
  for (let ti = 0; ti < TURNS.length; ti++) {
    const ctx = turnCtx(ti);
    if (!ctx.sug) continue;
    ctx.suggestions = SUGGESTED[ti] || [];
    paintSuggestions(ctx);
  }
}

function applyEvent(ev, ctx) {
  if (ev.type === 'thinking') {
    ctx.steps.push({ html: `<div class="step"><span class="lbl">reasoning</span><span class="think">${esc(ev.text)}</span></div>` });
  } else if (ev.type === 'search') {
    if (ev.query) ctx.searches++;
    ctx.sum.textContent = `Reading… ${ctx.searches} search${ctx.searches === 1 ? '' : 'es'}`;
    const label = ev.tool === 'suggest_books' ? 'looking further' : 'search';
    ctx.steps.push({ html: ev.query
      ? `<div class="step"><span class="lbl">${label}</span><span class="q">“${esc(ev.query)}”</span>${ev.book_id ? ` <span class="badge">book ${ev.book_id}</span>` : ''}</div>`
      : `<div class="step"><span class="lbl">tool</span><span class="q">the shelf</span></div>` });
  } else if (ev.type === 'results') {
    const last = ctx.steps[ctx.steps.length - 1];
    if (last) last.html += `<div class="hits">${(ev.hits || []).map(h =>
      `<div class="${h.weak ? 'w' : ''}"><b>[${h.n}]</b> d=${h.distance} · ${esc(h.book).slice(0, 40)} ${h.pages}${h.heading ? ' · ' + esc(h.heading).slice(0, 54) : ''}</div>`
    ).join('')}${ev.strong === 0 ? '<div class="w">— nothing strong on this shelf</div>' : ''}</div>`;
  } else if (ev.type === 'suggestions') {
    ctx.suggestions = ev.books || [];
    const last = ctx.steps[ctx.steps.length - 1];
    if (last) last.html += `<div class="hits"><div class="w">— ${ctx.suggestions.length} book(s) elsewhere in the library</div></div>`;
  } else if (ev.type === 'tool_error') {
    ctx.steps.push({ html: `<div class="step berr"><span class="lbl">error</span>${esc(ev.message)}</div>` });
  } else if (ev.type === 'answer') {
    ctx.rawAnswer = ev.text;
  } else if (ev.type === 'done') {
    TURNS[ctx.ti] = Object.fromEntries((ev.sources || []).map(s => [String(s.n), s]));
    ctx.ans.innerHTML = citeify(ctx.rawAnswer, ctx.ti);
    if (ev.suggestions && ev.suggestions.length) ctx.suggestions = ev.suggestions;
    SUGGESTED[ctx.ti] = ctx.suggestions;
    paintSuggestions(ctx);
    const n = (ev.sources || []).length;
    ctx.sum.textContent = `Read · ${ctx.searches} search${ctx.searches === 1 ? '' : 'es'} · ${n} source${n === 1 ? '' : 's'}`;
  } else if (ev.type === 'error') {
    throw new Error(ev.message);
  }
}

function renderFailure(ctx, message) {
  ctx.sum.textContent = 'The tutor stopped';
  ctx.ans.innerHTML = `<p class="err">${esc(String(message))}</p>`;
  scrollDown();
}

async function attachStream(turn, ctx) {
  const ctrl = new AbortController();
  LIVE.add(ctrl);
  try {
    const r = await fetch(
      `/api/research/${encodeURIComponent(turn.run_id)}/events?after=${turn.events.length}`,
      { signal: ctrl.signal });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);

    await readEventStream(r, ev => {
      turn.events.push(ev);
      applyEvent(ev, ctx);
      save();
      ctx.paint();
      scrollDown();
    }, {
      terminal: 'done',
      onStall: secs => {
        ctx.steps.push({ html: `<div class="step status warn"><span class="lbl">waiting</span>` +
          `no update for ${secs}s — still connected, but the model may be stuck.</div>` });
        ctx.paint();
      },
    });
    turn.status = 'done';
    save();
  } catch (err) {
    if (ctrl.signal.aborted) return;
    turn.status = 'failed';
    turn.error = String(err.message || err);
    save();
    renderFailure(ctx, turn.error);
  } finally {
    LIVE.delete(ctrl);
  }
}

$('#msgs').addEventListener('click', async e => {
  const btn = e.target.closest('.sadd');
  if (!btn) return;
  const books = SUGGESTED[btn.dataset.t] || [];
  const b = books[Number(btn.dataset.i)];
  if (!b) return;
  btn.disabled = true;
  const ok = await addBooks([b.book_id], 'reader', null);
  if (ok) repaintSuggestions(); else btn.disabled = false;
});

$('#f').addEventListener('submit', async e => {
  e.preventDefault();
  const question = $('#q').value.trim();
  if (!question) return;
  $('#q').value = '';
  $('#go').disabled = true;

  const turn = { question, run_id: null, status: 'running', events: [], error: null };
  const ti = CHAT.turns.push(turn) - 1;
  save();
  addTurn(question);
  const ctx = turnCtx(ti);
  ctx.steps.push({ html: `<div class="step status loading"><span class="lbl">start</span>thinking…</div>` });
  ctx.paint();
  scrollDown();

  try {
    const d = await fetchJSON('/api/research', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      // modelOf lives on the classroom page, the only page with a tutor.
      // Guarded so this file stays loadable anywhere else.
      body: JSON.stringify({ question, classroom_id: CID,
        model: (typeof modelOf === 'function' ? modelOf('tmodel') : undefined) }),
    }, 20000);
    turn.run_id = d.run_id;
    save();
    ctx.steps.length = 0;
    await attachStream(turn, ctx);
  } catch (err) {
    if (turn.status === 'running') {
      turn.status = 'failed';
      turn.error = String(err.message || err);
      save();
      renderFailure(ctx, turn.error);
    }
  } finally {
    $('#go').disabled = false;
    $('#q').focus();
  }
});

// ---- Restore ----
for (let ti = 0; ti < CHAT.turns.length; ti++) {
  const turn = CHAT.turns[ti];
  addTurn(turn.question);
  const ctx = turnCtx(ti);
  let shownFailure = false;
  for (const ev of turn.events) {
    try { applyEvent(ev, ctx); }
    catch (err) { renderFailure(ctx, String(err.message || err)); shownFailure = true; }
  }
  ctx.paint();
  if (turn.status === 'failed' && !shownFailure) {
    renderFailure(ctx, turn.error || 'The tutor stopped');
  } else if (turn.status === 'running') {
    if (turn.run_id) {
      $('#go').disabled = true;
      attachStream(turn, ctx).finally(() => { $('#go').disabled = false; $('#q').focus(); });
    } else {
      turn.status = 'failed';
      turn.error = 'Interrupted before the run started — ask again.';
      save();
      renderFailure(ctx, turn.error);
    }
  }
}
scrollDown();
paintChatBlank();
