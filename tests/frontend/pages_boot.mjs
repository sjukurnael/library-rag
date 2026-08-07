// Run by tests/test_frontend.py.
//
// Does each page actually BOOT? Both pages' inline scripts depend on helpers
// that live in app.js ($, esc, STAGE, TERMINAL, age, fetchJSON, progress,
// readEventStream). Forgetting the <script src="/static/app.js"> tag is a
// ReferenceError on the first top-level use, which halts the rest of the script
// -- so the page renders its static HTML and nothing else. It looks blank.
//
// Nothing else in the suite can see this. The Python tests assert GET / returns
// 200, which is true of a page whose JavaScript never runs; `node --check`
// passes, because a missing dependency is not a syntax error; and it happened.
//
// So: evaluate app.js and the page's inline script against a minimal DOM stub,
// in the same order the browser would, and fail on any error.

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const here = path.dirname(fileURLToPath(import.meta.url));
const STATIC = path.join(here, '../../src/library_rag/web/static');

// A DOM stub that answers everything permissively. The point is to surface
// ReferenceErrors and missing wiring, not to simulate a browser -- anything
// this stub has to model faithfully is a sign the test is reaching too far.
function makeDom() {
  const node = new Proxy({}, {
    get(_t, prop) {
      if (prop === 'classList') return { add(){}, remove(){}, toggle(){} };
      if (prop === 'dataset') return {};
      if (prop === 'style') return {};
      if (prop === 'files') return [];
      if (prop === 'parentElement' || prop === 'closest') return () => node;
      if (typeof prop === 'symbol') return undefined;
      return typeof prop === 'string' && /^(add|remove|insert|append|query|scroll|focus|requestSubmit|closest|open|write|set|get)/.test(prop)
        ? () => node
        : '';
    },
    set() { return true; },
  });
  return {
    querySelector: () => node,
    querySelectorAll: () => [],
    addEventListener: () => {},
    createElement: () => node,
    body: node,
  };
}

const results = [];
for (const page of ['index.html', 'library.html', 'queue.html']) {
  const html = fs.readFileSync(path.join(STATIC, page), 'utf8');

  // 1. The page must actually link the shared bundle.
  const linksJs = /<script[^>]+src=["']\/static\/app\.js["']/.test(html);
  const linksCss = /<link[^>]+href=["']\/static\/app\.css["']/.test(html);

  // 2. And its inline script must run with app.js loaded first.
  const inline = [...html.matchAll(/<script>\n([\s\S]*?)<\/script>/g)]
    .map(m => m[1]).join('\n');
  const shared = fs.readFileSync(path.join(STATIC, 'app.js'), 'utf8');

  let bootError = null;
  try {
    const fn = new Function(
      'document', 'window', 'fetch', 'location', 'confirm', 'console',
      `${shared}\n;{\n${inline}\n}`);
    fn(makeDom(), { open: () => null, location: {} },
       async () => ({ ok: true, json: async () => ({ books: [], pending: [], total_chunks: 0 }) }),
       { href: '' }, () => true, console);
  } catch (e) {
    bootError = `${e.name}: ${e.message}`;
  }
  results.push({ page, linksJs, linksCss, bootError });
}

let fail = 0;
for (const r of results) {
  const ok = r.linksJs && r.linksCss && !r.bootError;
  if (!ok) fail++;
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${r.page}` +
    (r.linksJs ? '' : '  [does not link /static/app.js]') +
    (r.linksCss ? '' : '  [does not link /static/app.css]') +
    (r.bootError ? `  [${r.bootError}]` : ''));
}
console.log(`\n${results.length - fail} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
