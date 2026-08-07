// Run by tests/test_frontend.py. Node only -- app.js is a plain browser script
// with no module system, so it is evaluated as source rather than imported.
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = fs.readFileSync(
  path.join(here, '../../src/library_rag/web/static/app.js'), 'utf8');
// app.js is a plain script; evaluate it and pull out what we test. `document`
// is a real parameter, not `this` -- a bare identifier never resolves through
// the call receiver, and app.js now touches document at the top level.
const mod = new Function('document',
  src + '\nreturn { readEventStream, progress, fetchJSON };')(
  { querySelector: () => null, querySelectorAll: () => [], addEventListener: () => {} });

function fakeResponse(frames, { cut = false } = {}) {
  const enc = new TextEncoder();
  let i = 0;
  return { body: { getReader: () => ({
    read: async () => {
      if (i >= frames.length) return { done: true };
      return { done: false, value: enc.encode(`data: ${JSON.stringify(frames[i++])}\n\n`) };
    }
  })}};
}

let pass = 0, fail = 0;
const t = (name, ok) => { ok ? pass++ : fail++; console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${name}`); };

// 1. a complete stream delivers every event and does not throw
const seen = [];
await mod.readEventStream(fakeResponse([{type:'tool'},{type:'answer'},{type:'done'}]),
  e => seen.push(e.type), { terminal: 'done' });
t('complete stream yields all events', JSON.stringify(seen) === '["tool","answer","done"]');

// 2. a stream that ends WITHOUT the terminal event must throw
let threw = null;
try {
  await mod.readEventStream(fakeResponse([{type:'tool'},{type:'answer'}]),
    () => {}, { terminal: 'done' });
} catch (e) { threw = e.message; }
t('truncated stream throws', !!threw && /ended before the run finished/.test(threw));

// 3. no terminal configured -> old permissive behaviour, no throw
threw = null;
try { await mod.readEventStream(fakeResponse([{type:'tool'}]), () => {}); }
catch (e) { threw = e.message; }
t('no terminal configured never throws', threw === null);

// 4. events still delivered before the throw (partial work is not discarded)
const partial = [];
try {
  await mod.readEventStream(fakeResponse([{type:'tool'},{type:'results'}]),
    e => partial.push(e.type), { terminal: 'done' });
} catch {}
t('events before a cut are still delivered', partial.length === 2);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
