const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const html = fs.readFileSync(path.join(__dirname, '../src/api/learndl_monitor/storage.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function page(remembered = '', search = '') {
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
  const messages = [];
  element('report').contentWindow = { postMessage: (message, target) => messages.push({ message, target }) };
  const context = vm.createContext({
    document: { querySelector: selector => selector === '#language, #lang' ? element('language') : null, getElementById: element, documentElement: {}, createElement: () => ({}) },
    localStorage: { getItem: key => values.get(key), setItem: (key, value) => values.set(key, value) },
    setInterval() {}, confirm: () => true, URL, URLSearchParams,
    window: { location: new URL('http://127.0.0.1:8503/test' + search) },
    history: { replaceState() {} },
    fetch: async (url, options) => {
      requests.push({ url, options });
      return { ok: true, json: async () => structuredClone(state) };
    },
  });
  context.location = context.window.location;
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../src/api/learndl_monitor/navigation.js'), 'utf8'), context);
  vm.runInContext(script, context);
  return { state, element, context, requests, messages };
}

const settle = () => new Promise(resolve => setImmediate(resolve));

test('remote default, update notification and explicit reload preserve reading position', async () => {
  const app = page();
  await settle();
  assert.equal(app.element('source').value, 'server');
  assert.equal(app.element('remote-tab')['aria-selected'], 'true');
  assert.equal(app.element('report').sandbox, 'allow-scripts');
  const previous = app.element('report').src;
  app.state.sources[0].revision = 'second';
  await vm.runInContext('refresh()', app.context);
  assert.equal(app.element('report').src, previous);
  assert.equal(app.element('newReport').hidden, false);
  app.element('newReport').onclick();
  assert.notEqual(app.element('report').src, previous);
  assert.equal(app.element('newReport').hidden, true);
  app.element('local-tab').onclick();
  await settle();
  assert.equal(app.element('report').hidden, true);
  assert.equal(app.element('local-tools').hidden, false);
  assert.ok(app.requests.every(request => request.url.endsWith('/status')));
});

test('remembered remote source wins over an existing local snapshot', async () => {
  const app = page('remote:server');
  app.state.snapshot = 'local.html';
  await settle();
  assert.equal(app.element('source').value, 'server');
  assert.equal(app.element('local-tools').hidden, true);
});

test('language from the console and language changes reach a sandboxed report without reloading it', async () => {
  const app = page('', '?lang=en');
  await settle();
  assert.equal(app.element('remote-tab').textContent, 'Server');
  const report = app.element('report');
  const address = report.src;
  report.onload();
  assert.equal(app.messages.at(-1).message.language, 'en');
  assert.equal(app.messages.at(-1).target, '*');
  app.element('language').onchange({target: {value: 'zh'}});
  assert.equal(app.element('remote-tab').textContent, '服务器');
  assert.equal(app.messages.at(-1).message.language, 'zh');
  assert.equal(report.src, address);
});

test('server tab stays selected while waiting for the first server report', async () => {
  const app = page();
  app.state.sources = [];
  await settle();
  app.element('remote-tab').onclick();
  await vm.runInContext('refresh()', app.context);
  assert.equal(app.element('remote-tab')['aria-selected'], 'true');
  assert.equal(app.element('empty').hidden, false);
  assert.match(app.element('empty').textContent, /Syncthing/);
});
