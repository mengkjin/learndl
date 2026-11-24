const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const html = fs.readFileSync(path.join(__dirname, '../src/api/learndl_monitor/storage.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function page(remembered = '') {
  const elements = new Map();
  const requests = [];
  const state = {
    busy: false, error: '', snapshot: '', progress: [], revision: 0,
    sources: [{ id: 'server', source_host: 'server', finished: '2026-11-23T10:00:00+00:00', complete: true, revision: 'first' }],
  };
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      hidden: false, textContent: '', parentElement: {},
      scrollHeight: 0, scrollTop: 0, clientHeight: 0,
      replaceChildren(...children) { this.children = children; },
      setAttribute(name, value) { this[name] = value; },
      removeAttribute(name) { delete this[name]; },
    });
    return elements.get(id);
  }
  const values = new Map([['learndl-storage-source', remembered]]);
  const context = vm.createContext({
    document: { getElementById: element, documentElement: {}, createElement: () => ({}) },
    localStorage: { getItem: key => values.get(key), setItem: (key, value) => values.set(key, value) },
    setInterval() {}, confirm: () => true,
    fetch: async (url, options) => {
      requests.push({ url, options });
      return { ok: true, json: async () => structuredClone(state) };
    },
  });
  vm.runInContext(script, context);
  return { state, element, context, requests };
}

const settle = () => new Promise(resolve => setImmediate(resolve));

test('remote default, update notification and explicit reload preserve reading position', async () => {
  const app = page();
  await settle();
  assert.equal(app.element('source').value, 'remote:server');
  assert.equal(app.element('report').sandbox, 'allow-scripts');
  const previous = app.element('report').src;
  app.state.sources[0].revision = 'second';
  await vm.runInContext('refresh()', app.context);
  assert.equal(app.element('report').src, previous);
  assert.equal(app.element('newReport').hidden, false);
  app.element('newReport').onclick();
  assert.notEqual(app.element('report').src, previous);
  assert.equal(app.element('newReport').hidden, true);
  app.element('source').value = 'local';
  app.element('source').onchange();
  await settle();
  assert.equal(app.element('report').hidden, true);
  assert.equal(app.element('scan').hidden, false);
  assert.ok(app.requests.every(request => request.url.endsWith('/status')));
});

test('remembered remote source wins over an existing local snapshot', async () => {
  const app = page('remote:server');
  app.state.snapshot = 'local.html';
  await settle();
  assert.equal(app.element('source').value, 'remote:server');
  assert.equal(app.element('email').hidden, true);
});
