// Run by tests/test_frontend.py.
//
// The librarian's action buttons are gated on `_driveOk`, because indexing
// downloads from Drive and the endpoint raises before queueing anything when
// Drive is not connected -- a 503 on click is a worse way to learn that than a
// button that says so up front.
//
// The gate has a failure mode that no other test can see. `_driveOk` starts
// false and is only ever *read*; if nothing assigns it, every button renders
// permanently disabled on a perfectly healthy server. That is not a syntax
// error, not a ReferenceError, and not visible to pages_boot.mjs (which only
// asks whether the script runs). The page looks fine. The buttons just never
// work, and the tooltip blames Drive.
//
// So drive the real contract: run paintAuth() against a stubbed status
// endpoint, then ask recCard() what it rendered. Both polarities matter --
// enabled when Drive is up is the bug that shipped, disabled when it is down is
// the property the gate exists for.

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const here = path.dirname(fileURLToPath(import.meta.url));
const STATIC = path.join(here, '../../src/library_rag/web/static');
const shared = fs.readFileSync(path.join(STATIC, 'app.js'), 'utf8');

// Enough DOM for paintAuth to paint into. It looks up '#sideauth' and sets
// innerHTML/hidden/title; nothing here needs to be faithful beyond that.
function makeDom(hasSideauth) {
  const node = new Proxy({}, {
    get(_t, p) {
      if (p === 'classList') return { add(){}, remove(){}, toggle(){} };
      if (p === 'dataset' || p === 'style') return {};
      if (typeof p === 'symbol') return undefined;
      return typeof p === 'string' && /^(add|remove|insert|append|query|scroll|focus|closest|set|get)/.test(p)
        ? () => node : '';
    },
    set() { return true; },
  });
  return {
    querySelector: (sel) => (sel === '#sideauth' && !hasSideauth) ? null : node,
    querySelectorAll: () => [],
    addEventListener: () => {},
    createElement: () => node,
    body: node,
  };
}

// Evaluate app.js and hand back the two functions under test, bound to a
// stubbed /api/drive/auth/status.
function load({ driveOk, hasSideauth }) {
  const fetch = async (url) => ({
    ok: true,
    status: 200,
    headers: { get: () => 'application/json' },
    json: async () => url.includes('/api/drive/auth/status')
      ? { ok: driveOk, reason: driveOk ? null : 'no_token' }
      : {},
  });
  const fn = new Function(
    'document', 'window', 'fetch', 'location', 'confirm', 'console',
    'sessionStorage', 'localStorage', 'alert', 'prompt', 'setInterval',
    `${shared}\n;return { paintAuth, recCard };`);
  const store = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
  return fn(makeDom(hasSideauth), { open: () => null, location: {}, addEventListener: () => {} },
    fetch, { href: '', hash: '', pathname: '/library' }, () => true, console,
    store, store, () => {}, () => null, () => 0);
}

const BOOK = { title: 'A Book', why: 'because', file_id: 'abc', size_mb: 1, indexed: false };

const cases = [
  { name: 'Drive connected -> Index button is enabled',
    driveOk: true, hasSideauth: true, wantDisabled: false },
  { name: 'Drive down -> Index button is disabled',
    driveOk: false, hasSideauth: true, wantDisabled: true },
  // bible.html renders librarian picks but has no #sideauth indicator. If
  // paintAuth bails on the missing element before recording the status, this
  // page's buttons are dead for a reason that has nothing to do with Drive.
  { name: 'page without #sideauth still learns Drive is up',
    driveOk: true, hasSideauth: false, wantDisabled: false },
];

let fail = 0;
for (const c of cases) {
  let got, err = null;
  try {
    const { paintAuth, recCard } = load(c);
    await paintAuth();
    const html = recCard(BOOK, 0);
    if (!/<button/.test(html)) throw new Error(`rendered no button: ${html.slice(0, 120)}`);
    got = /disabled/.test(html);
  } catch (e) { err = `${e.name}: ${e.message}`; }

  const ok = !err && got === c.wantDisabled;
  if (!ok) fail++;
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${c.name}` +
    (err ? `  [${err}]` : ok ? '' : `  [rendered ${got ? 'disabled' : 'enabled'}]`));
}
console.log(`\n${cases.length - fail} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
