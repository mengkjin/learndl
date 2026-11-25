import renderTerminal from './live_output.js';

const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="task-token"]').content;
const mode = ({'/remote': 'remote', '/finished': 'finished'})[location.pathname] || 'running';
const remote = mode === 'remote';
const finished = mode === 'finished';
const styleKey = finished ? 'learndl-tasks-finished-style' : 'learndl-tasks-output-style';
const sizeKey = 'learndl-tasks-terminal-size';
const words = {
  title: ['任务中心', 'Task Center'], settings: ['设置', 'Settings'],
  outputStyle: ['输出样式', 'Output Style'], colored: ['彩色文本', 'Colored text'],
  terminalSize: ['终端字号', 'Terminal font size'],
  plain: ['纯文本', 'Text only'],
  running: ['运行中', 'Running'], remote: ['远程运行', 'Remote Running'],
  finished: ['已结束', 'Finished'], recent: ['最近结束', 'Recently finished'],
  server: ['服务器', 'Server'], filters: ['状态、脚本与日期', 'Status, script and dates'],
  remoteFilters: ['服务器与任务状态', 'Server and task status'], taskList: ['任务列表', 'Task list'],
  complete: ['完成', 'Complete'], failed: ['错误', 'Error'], killed: ['已终止', 'Killed'],
  starting: ['启动中', 'Starting'], queued: ['排队中', 'Queued'], stale: ['状态过期', 'Stale'],
  exited: ['进程已退出', 'Process exited'],
  startTime: ['开始时间', 'Started'], ended: ['结束时间', 'Ended'],
  scriptName: ['脚本名', 'Script name'], startDate: ['开始日期', 'Start date'],
  endDate: ['结束日期', 'End date'], dateFrom: ['从', 'From'], dateTo: ['到', 'Through'],
  invalidRange: ['日期范围的起始日期应早于或等于截止日期', 'The first date must be on or before the last date'],
  history: ['全部', 'History'], lastDay: ['最近一天', 'Last day'],
  lastWeek: ['最近一周', 'Last week'], lastMonth: ['最近一月', 'Last month'],
  previous: ['上一页', 'Previous'], next: ['下一页', 'Next'],
  followTask: ['跟随最新任务', 'Follow latest task'],
  kill: ['终止任务', 'Terminate task'], confirmText: ['确认终止此任务', 'I confirm termination of this task'],
  cancel: ['取消', 'Cancel'], details: ['任务详情', 'Task details'],
  source: ['输出文件', 'Output file'], generation: ['日志代次', 'Log generation'],
  earlier: ['加载更早输出', 'Load earlier output'],
  reload: ['重新加载日志', 'Reload logs'], outputPage: ['输出页', 'Output page'],
  bottom: ['回到底部', 'Jump to bottom'], report: ['完整 HTML 报告', 'Full HTML report'],
  download: ['下载报告', 'Download report'], status: ['状态', 'Status'],
  script: ['脚本', 'Script'], duration: ['耗时', 'Duration'], origin: ['来源', 'Source'],
  background: ['后台', 'Background'], taskId: ['任务 ID', 'Task ID'],
  phase: ['调度阶段', 'Managed phase'], children: ['子进程', 'Child PIDs'],
  sampled: ['采样时间', 'Sampled'], lastOutput: ['最近输出', 'Last output'],
  availableUntil: ['保留至', 'Available until'], empty: ['暂无运行任务', 'No running tasks'],
  noFinished: ['没有符合条件的任务', 'No matching tasks'],
  selectFinished: ['从列表选择任务查看详情和输出', 'Select a task from the list to view its details and output'],
  noHosts: ['等待 Syncthing 同步服务器快照', 'Waiting for server snapshots in Syncthing'],
  expired: ['快照已过期', 'Snapshot is out of date'],
  syncing: ['部分输出正在同步', 'Some output is still syncing'],
  collecting: ['服务器正在收集日志', 'Server is collecting output'],
  missing: ['源日志有缺失', 'Some source output is missing'],
  noOutput: ['暂无输出', 'No output yet'],
  unavailable: ['输出文件或缓存不可用', 'Output source or cache unavailable'],
  records: ['条记录', 'records'], results: ['条任务', 'tasks'],
  lifecycle: ['生命周期检查', 'Lifecycle check'],
  cacheHealth: ['输出缓存维护', 'Output cache maintenance'],
  timeoutProtection: ['超时保护', 'Timeout protection'],
  watchdogHealthy: ['Watchdog 正常', 'Watchdog healthy'],
  disabled: ['已停用', 'Disabled'],
  heartbeatMissing: ['心跳过期或缺失', 'Heartbeat missing or stale'],
  versionPatch: ['版本升级与补丁', 'Version upgrade and patches'],
  patch_running: ['正在执行补丁，新任务等待', 'Applying patch · new tasks are waiting'],
  patch_ready: ['版本已就绪', 'Version ready'],
  patch_pending: ['等待升级', 'Upgrade pending'],
  patch_partial: ['升级尚未完成', 'Upgrade incomplete'],
  patch_failed: ['补丁失败，等待恢复', 'Patch failed · recovery required'],
  patch_interrupted: ['补丁已中断，等待恢复', 'Patch interrupted · recovery required'],
  patch_uninitialized: ['等待初始化', 'Awaiting initialization'],
  patch_unavailable: ['无法读取补丁状态', 'Patch status unavailable'],
  patchLastRunning: ['上次记录为执行中，等待更新', 'Last reported running · awaiting update'],
  patch_unconfirmed: ['执行已停止，等待核对补丁记录', 'Execution stopped · awaiting reconciliation'],
  dataVersion: ['数据版本', 'Data version'], targetVersion: ['目标版本', 'Target version'],
  patchRemaining: ['待验证补丁', 'Patches remaining'], attempt: ['尝试次数', 'Attempt'],
  terminateGroup: ['整个进程组将被终止，自动恢复也将停用。', 'The entire process group will stop and automatic recovery will be disabled.'],
  terminateProcess: ['将终止选中的脚本进程。', 'The selected script process will stop.'],
};
let lang = monitorNavigation.language();
let style = localStorage.getItem(styleKey) === 'text' ? 'text' : 'colored';
let hosts = [];
let host = localStorage.getItem('learndl-tasks-host') || '';
let cursorStack = [null];
let nextCursor = null;
let selectedFinished = null;
let identity = '';
let task = null;
let source = 0;
let generation = null;
let outputPage = null;
let loadedReport = '';
let currentBlock = null;
let older = [];
let earlierSyncing = false;
let confirmation = null;
let revision = 0;
let manualRequests = 0;
let listSignature = '';
let scriptFilterTimer;
const pinned = new Map();

const t = key => words[key]?.[lang === 'en' ? 1 : 0] || key;
const labelStatus = status => t(status === 'error' ? 'failed' : status);
const taskStatus = row => {
  if (remote && row.status === 'stale') return t('exited');
  return labelStatus(row.phase === 'queued' ? 'queued' : row.status);
};
const nameOf = path => path.split(/[\\/]/).at(-1);
const date = value => value ? new Date(value * 1000).toLocaleString(lang === 'zh' ? 'zh-CN' : 'en-US') : '—';
const terminal = $('terminal');
const viewport = terminal.querySelector('.live-output');
const jump = terminal.querySelector('button');
const sizeInput = $('terminal-size');

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

async function request(path, options = {}) {
  const headers = {Authorization: 'Bearer ' + token};
  const method = options.body ? 'POST' : 'GET';
  if (options.body) headers['Content-Type'] = 'application/json';
  const url = path + (options.query ? '?' + new URLSearchParams(options.query) : '');
  const response = await fetch(url, {method, headers, body: options.body && JSON.stringify(options.body)});
  if (options.blob && response.ok) return response.blob();
  const result = await response.json();
  if (!response.ok) throw Error(result.error || response.statusText);
  return result;
}

function message(text = '') {
  $('error').textContent = text;
  $('error').hidden = !text;
}

function labels() {
  monitorNavigation.apply(lang);
  document.querySelector('nav').setAttribute('aria-label', t('title'));
  $('cards').setAttribute('aria-label', t('taskList'));
  viewport.setAttribute('aria-label', lang === 'en' ? 'Task output' : '任务输出');
  $('preview').title = t('report');
  document.title = t('title');
  document.querySelectorAll('[data-label]').forEach(node => {node.textContent = t(node.dataset.label);});
  $('language').value = lang;
  $('style').value = style;
}

function applyTerminalSize() {
  const state = viewport.liveOutputState;
  // Preserve the visible log line when the new font changes wrapping.
  const line = state && !state.following &&
    Array.from(viewport.firstElementChild.children).find(row => row.offsetTop + row.offsetHeight > viewport.scrollTop);
  const offset = line ? (viewport.scrollTop - line.offsetTop) / Math.max(1, line.offsetHeight) : 0;
  const size = sizeInput.value;
  viewport.style.setProperty('--terminal-size', size + 'px');
  const label = size + ' px';
  $('terminal-size-value').textContent = label;
  sizeInput.setAttribute('aria-valuetext', label);
  if (state) {
    if (state.following) viewport.scrollTop = viewport.scrollHeight;
    else if (line) viewport.scrollTop = line.offsetTop + offset * line.offsetHeight;
    state.top = viewport.scrollTop;
  }
}

function duration(row) {
  if (!row.start_time || (remote && row.status === 'stale' && !row.end_time)) return '—';
  const seconds = Math.max(0, Math.floor((row.end_time || Date.now() / 1000) - row.start_time));
  return [Math.floor(seconds / 3600), Math.floor(seconds / 60) % 60, seconds % 60]
    .map(n => String(n).padStart(2, '0')).join(':');
}

function renderPatch(progress) {
  $('patch-status').hidden = !progress;
  if (!progress) return;
  const stale = remote && Date.now() / 1000 - progress.sampled_at > 60;
  const phase = $('patch-phase');
  phase.textContent = t(stale && progress.status === 'running' ? 'patchLastRunning' : 'patch_' + progress.status);
  phase.className = ['failed', 'interrupted', 'unavailable'].includes(progress.status) ? 'error'
    : stale || progress.status !== 'ready' ? 'warning' : '';
  const detail = remote && progress.host ? [t('server') + ' ' + progress.host] : [];
  if (progress.applied) detail.push(t('dataVersion') + ' ' + progress.applied);
  if (progress.target) detail.push(t('targetVersion') + ' ' + progress.target);
  if (progress.patch) detail.push(progress.patch + ' · ' + progress.script);
  if (progress.remaining) detail.push(t('patchRemaining') + ' ' + progress.remaining);
  if (progress.attempt) detail.push(t('attempt') + ' ' + progress.attempt);
  $('patch-detail').textContent = detail.join(' · ');
  const timing = [];
  if (progress.started) {
    timing.push(t('startTime') + ' ' + date(progress.started));
    if (progress.status !== 'unconfirmed') timing.push(t('duration') + ' ' + duration({
      start_time: progress.started, end_time: progress.finished || progress.sampled_at}));
  }
  timing.push(t('sampled') + ' ' + date(progress.sampled_at));
  if (stale) timing.push(t('expired'));
  $('patch-timing').textContent = timing.join(' · ');
  $('patch-error').textContent = progress.error || '';
}

function setOptions(select, values, value) {
  const signature = JSON.stringify(values);
  if (select.dataset.signature !== signature) {
    select.replaceChildren(...values.map(([key, title]) => {
      const option = element('option', title);
      option.value = key;
      return option;
    }));
    select.dataset.signature = signature;
  }
  select.value = value;
}

function renderHealth(health) {
  const footer = $('health');
  if (!health) {
    footer.hidden = true;
    return;
  }
  const labels = {task_lifecycle: 'lifecycle', task_monitor_cache: 'cacheHealth', task_timeouts: 'timeoutProtection'};
  let warned = false;
  const items = Object.entries(health).map(([name, item]) => {
    const status = item.payload.disabled ? t('disabled') : item.stale ? t('heartbeatMissing') : date(item.last_run);
    warned ||= item.stale;
    return t(labels[name]) + ' · ' + status;
  });
  footer.textContent = (warned ? '' : t('watchdogHealthy') + ' · ') + items.join('   |   ');
  footer.classList.toggle('warning', warned);
  footer.hidden = false;
}

function renderCards(rows, active, pinKey) {
  const title = finished ? 'finished' : remote && $('recent').checked ? 'recent' : 'running';
  const queued = title === 'running' ? rows.filter(row => row.phase === 'queued').length : 0;
  $('queue-title').textContent = t(title) + ' (' + (rows.length - queued) + ')'
    + (queued ? ' · ' + t('queued') + ' (' + queued + ')' : '');
  $('follow-task').disabled = !pinned.has(pinKey);
  const signature = JSON.stringify([lang, pinKey, active, rows]);
  if (signature === listSignature) return;
  listSignature = signature;
  const cards = $('cards');
  const scroll = cards.scrollTop;
  cards.replaceChildren(...rows.map(row => {
    const button = element('button', undefined, 'card');
    button.title = row.script;
    button.dataset.taskId = row.task_id;
    button.setAttribute('aria-pressed', String(row.task_id === active));
    const secondary = [taskStatus(row), date(row.start_time)];
    if (finished) secondary.push(duration(row));
    button.append(element('strong', nameOf(row.script)),
                  element('span', secondary.join(' · '), 'muted'));
    button.onclick = () => {
      if (finished) selectedFinished = row.task_id;
      else pinned.set(pinKey, row.task_id);
      updateNow();
    };
    return button;
  }));
  cards.scrollTop = scroll;
}

function renderFinishedSummary(packet) {
  const {counts, total, next_cursor} = packet;
  nextCursor = next_cursor;
  $('prev-page').disabled = cursorStack.length < 2;
  $('next-page').disabled = !nextCursor;
  $('page-info').textContent = `${cursorStack.length} · ${total} ${t('results')}`;
  $('counts').textContent = Object.entries(counts).map(([name, count]) => labelStatus(name) + ' ' + count).join(' · ');
}

function finishedQuery() {
  const query = {
    statuses: [...document.querySelectorAll('[name=status]:checked')].map(node => node.value).join(','),
    window: $('window').value,
    script: $('script-filter').value.trim(),
  };
  for (const range of document.querySelectorAll('[data-date-range]')) {
    const from = range.querySelector('[data-bound=from]').value;
    const through = range.querySelector('[data-bound=before]').value;
    if (from && through && from > through) throw Error(t('invalidRange'));
    if (from) query[range.dataset.dateRange + '_from'] = new Date(from + 'T00:00:00').getTime() / 1000;
    if (through) {
      const day = new Date(through + 'T00:00:00');
      day.setDate(day.getDate() + 1);
      day.setHours(0, 0, 0, 0);
      query[range.dataset.dateRange + '_before'] = day.getTime() / 1000;
    }
  }
  const cursor = cursorStack.at(-1);
  if (cursor) [query.cursor_time, query.cursor_id] = cursor;
  return query;
}

function resetOutput() {
  currentBlock = null;
  older = [];
  earlierSyncing = false;
  outputPage = null;
  $('output-info').textContent = '';
  $('earlier').hidden = true;
  $('report').open = false;
  $('preview').hidden = true;
  $('preview').removeAttribute('srcdoc');
  loadedReport = '';
  viewport.liveOutputState = undefined;
  viewport.firstElementChild.replaceChildren();
  jump.hidden = true;
}

function selectTask(nextIdentity) {
  if (identity === nextIdentity) return;
  identity = nextIdentity;
  task = null;
  source = 0;
  generation = null;
  $('details').open = false;
  $('task').hidden = true;
  resetOutput();
}

function displayTask(row) {
  $('empty').hidden = true;
  $('task').hidden = false;
  $('script').textContent = nameOf(row.script);
  $('task-status').textContent = [taskStatus(row), date(row.start_time), duration(row)].join(' · ');
  const metadata = {
    script: row.script, taskId: row.task_id, startTime: date(row.start_time),
    ended: row.end_time ? date(row.end_time) : null, PID: row.pid, origin: row.source,
    background: finished && row.background ? '✓' : null,
    phase: row.phase, children: row.child_pids?.length ? row.child_pids.join(', ') : null,
    sampled: remote ? date(hosts.find(entry => entry.id === host)?.sampled_at) : null,
    lastOutput: remote ? date(row.output_at) : null,
    availableUntil: row.expires_at ? date(row.expires_at) : null,
  };
  $('metadata').replaceChildren(...Object.entries(metadata).filter(([,value]) => value != null)
    .flatMap(([key,value]) => [element('dt', t(key)), element('dd', value)]));
  $('command').textContent = row.cmd || '';
  $('command').hidden = !row.cmd;
  $('exit-error').textContent = [row.exit_message, row.exit_error].filter(Boolean).join('\n');
  $('exit-error').hidden = !$('exit-error').textContent;
  const stop = $('stop-task');
  stop.hidden = remote || finished || !row.active;
  stop.disabled = Boolean(row.kill_reason);
  stop.title = row.kill_reason || '';
  const files = row.files || [];
  source = Math.min(source, Math.max(0, files.length - 1));
  $('source').hidden = $('source-label').hidden = !files.length;
  setOptions($('source'), files.map((file, index) => [index, file.path]), source);
  const histories = (row.previous || []).map(history => history.generation);
  if (row.generation != null) histories.push(row.generation);
  if (!histories.includes(generation)) generation = null;
  $('generation').hidden = $('generation-label').hidden = !remote || histories.length < 2;
  setOptions($('generation'), histories.map(value => [value, String(value + 1)]), generation ?? row.generation);
  $('reload').hidden = !remote;
  $('report').hidden = !finished || !files[source]?.html;
}

function outputQuery() {
  const query = {task: task.task_id, style, source};
  if (remote) {
    query.host = host;
    if (generation != null) query.generation = generation;
  }
  if (outputPage != null) query.page = outputPage;
  return query;
}

function showTerminal(content, options = {}) {
  renderTerminal({data: {content: content || '', style, ...options}, parentElement: terminal,
    onJump: remote ? () => {
      older = [];
      earlierSyncing = false;
      updateNow();
    } : undefined});
}

function drawRemote(block, {prepend = false} = {}) {
  if (block.content != null) currentBlock = block;
  const content = [...older, currentBlock].filter(Boolean).map(part => part.content).filter(Boolean);
  if (content.length || block.content != null) {
    showTerminal(content.join(style === 'text' ? '\n' : ''), {prepend});
  }
  $('earlier').hidden = !(older[0] || currentBlock)?.start;
  $('output-info').textContent = [
    block.syncing || earlierSyncing ? t('syncing') : '',
    task.gap ? t('missing') : '',
    !task.complete && !task.syncing && task.phase !== 'queued' ? t('collecting') : '',
    !content.length && !block.syncing ? t('noOutput') : '',
  ].filter(Boolean).join(' · ');
}

async function loadOutput(version) {
  if (remote) {
    if (task.syncing) {
      $('output-info').textContent = t('syncing');
      return;
    }
    const query = outputQuery();
    if (currentBlock && viewport.liveOutputState?.following === false && !currentBlock.syncing) {
      query.start = currentBlock.start;
      query.before = currentBlock.end;
    }
    const block = await request('/api/remote/output', {query});
    if (version !== revision) return;
    if (older.length && currentBlock && block.start !== currentBlock.start) older = [];
    drawRemote(block);
    return;
  }
  const result = task.files?.length ? await request('/api/output', {query: outputQuery()}) : {content: ''};
  if (version !== revision) return;
  showTerminal(result.content, {startAtTop: finished});
  $('output-info').textContent = result.unavailable ? t('unavailable') : result.total_pages
    ? `${result.record_count} ${t('records')} · ${result.page + 1}/${result.total_pages}`
    : result.content ? '' : t('noOutput');
  const page = $('output-page');
  page.hidden = $('output-page-label').hidden = !finished || !(result.total_pages > 1);
  if (!page.hidden && document.activeElement !== page) {
    page.max = result.total_pages;
    page.value = result.page + 1;
  }
}

async function refresh() {
  const version = ++revision;
  try {
    let rows;
    if (remote) {
      const response = await request('/api/remote');
      if (version !== revision) return;
      hosts = response;
      if (!hosts.some(entry => entry.id === host)) host = hosts[0]?.id || '';
      setOptions($('host'), hosts.map(entry => [entry.id, entry.name]), host);
      const server = hosts.find(entry => entry.id === host);
      renderPatch(server?.version_patch);
      const expired = server && (!server.sampled_at || Date.now() / 1000 - server.sampled_at > 60);
      $('sample').textContent = server ? t('sampled') + ' ' + date(server.sampled_at) + (expired ? ' · ' + t('expired') : '') : t('noHosts');
      $('sample').classList.toggle('warning', Boolean(expired));
      rows = (server?.tasks || []).filter(row => ['starting', 'running'].includes(row.status) !== $('recent').checked);
    } else {
      const query = finished ? finishedQuery() : {};
      const packet = await request(finished ? '/api/finished' : '/api/running', {query});
      if (version !== revision) return;
      rows = packet.tasks;
      renderPatch(packet.version_patch);
      if (finished) renderFinishedSummary(packet);
      renderHealth(packet.health);
    }
    const pinKey = remote ? host + '/' + $('recent').checked : 'local';
    if (!finished) rows.sort((a, b) => b.start_time - a.start_time);
    if (!rows.some(row => row.task_id === pinned.get(pinKey))) pinned.delete(pinKey);
    if (finished && !rows.some(row => row.task_id === selectedFinished)) selectedFinished = null;
    const active = finished ? selectedFinished : pinned.get(pinKey) || rows[0]?.task_id;
    renderCards(rows, active, pinKey);
    selectTask(active ? JSON.stringify([mode, host, active]) : '');
    if (!active) {
      $('task').hidden = true;
      $('empty').hidden = false;
      $('empty').textContent = t(finished ? rows.length ? 'selectFinished' : 'noFinished' : remote && $('recent').checked ? 'noFinished' : 'empty');
      return;
    }
    const row = await request(remote ? '/api/remote/task' : '/api/task', {query: {host, task: active}});
    if (version !== revision) return;
    if (remote && generation == null && task?.generation != null && row.generation !== task.generation) {
      resetOutput();
    }
    task = row;
    displayTask(row);
    await loadOutput(version);
  } catch (error) {
    if (version === revision) message(error.message);
  }
}

function updateNow() {
  message();
  return refresh();
}

async function manual(action) {
  ++manualRequests;
  try { await action(); }
  catch (error) { message(error.message); }
  finally { --manualRequests; }
}

$('earlier').onclick = () => manual(async () => {
  if (!task || !currentBlock) return;
  const version = ++revision;
  const before = (older[0] || currentBlock).start;
  const button = $('earlier');
  button.disabled = true;
  try {
    const query = {...outputQuery(), before};
    const block = await request('/api/remote/output', {query});
    if (version !== revision) return;
    if (block.content != null) older.unshift(block);
    earlierSyncing = Boolean(block.syncing);
    drawRemote(currentBlock, {prepend: true});
  } finally {
    button.disabled = false;
  }
});

$('host').onchange = event => {
  host = event.target.value;
  localStorage.setItem('learndl-tasks-host', host);
  updateNow();
};
$('recent').onchange = updateNow;
sizeInput.oninput = () => {
  applyTerminalSize();
  localStorage.setItem(sizeKey, sizeInput.value);
};
$('follow-task').onclick = () => {
  pinned.delete(remote ? host + '/' + $('recent').checked : 'local');
  updateNow();
};
$('style').onchange = event => {
  style = event.target.value;
  localStorage.setItem(styleKey, style);
  if (!remote || !currentBlock) return updateNow();
  manual(async () => {
    const version = ++revision;
    const query = outputQuery();
    const blocks = await Promise.all([...older, currentBlock].map(block =>
      request('/api/remote/output', {query: {...query, start: block.start, before: block.end}})));
    if (version !== revision) return;
    if (blocks.every(block => block.content != null)) {
      currentBlock = blocks.pop();
      older = blocks;
      drawRemote(currentBlock);
    } else {
      resetOutput();
      updateNow();
    }
  });
};
$('refresh-page').onclick = updateNow;
$('language').onchange = event => {
  lang = event.target.value;
  localStorage.setItem('learndl-language', lang);
  labels();
  updateNow();
};
$('source').onchange = event => {source = Number(event.target.value); resetOutput(); updateNow();};
$('generation').onchange = event => {generation = Number(event.target.value); resetOutput(); updateNow();};
$('output-page').onchange = event => {
  if (event.target.checkValidity()) {
    outputPage = Number(event.target.value) - 1;
    updateNow();
  }
};
$('reload').onclick = () => manual(async () => {
  if (!task) return;
  const version = ++revision;
  await request('/api/remote/reload', {body: {host, task: task.task_id}});
  if (version !== revision) return;
  resetOutput();
  await updateNow();
});
function changeFinishedPage() {
  selectedFinished = null;
  $('cards').scrollTop = 0;
  updateNow();
}

function applyFinishedFilters() {
  clearTimeout(scriptFilterTimer);
  cursorStack = [null];
  changeFinishedPage();
}

$('filters').onchange = event => {
  if (event.target !== $('script-filter')) applyFinishedFilters();
};
$('script-filter').oninput = () => {
  clearTimeout(scriptFilterTimer);
  scriptFilterTimer = setTimeout(applyFinishedFilters, 250);
};
$('prev-page').onclick = () => {
  if (cursorStack.length > 1) {
    cursorStack.pop();
    changeFinishedPage();
  }
};
$('next-page').onclick = () => {
  if (nextCursor) {
    cursorStack.push(nextCursor);
    changeFinishedPage();
  }
};

$('stop-task').onclick = () => manual(async () => {
  if (!task) return;
  const selected = identity;
  const prepared = await request('/api/kill/prepare', {body: {task: task.task_id}});
  if (identity !== selected) return;
  confirmation = prepared;
  $('confirmed').checked = false;
  $('execute').disabled = true;
  $('kill-info').textContent = [t(prepared.managed ? 'terminateGroup' : 'terminateProcess'),
    prepared.script, 'PID ' + prepared.pid, date(prepared.started)].join(' · ');
  $('confirm').showModal();
});
$('confirmed').onchange = event => {$('execute').disabled = !event.target.checked;};
$('execute').onclick = () => manual(async () => {
  if (!confirmation || !$('confirmed').checked) return;
  const prepared = confirmation;
  confirmation = null;
  $('execute').disabled = true;
  $('confirm').close();
  const result = await request('/api/kill/confirm', {body: {token: prepared.token}});
  await updateNow();
  message(result.message);
});
$('confirm').addEventListener('close', () => {confirmation = null;});

$('report').addEventListener('toggle', () => manual(async () => {
  if (!$('report').open || !task) return;
  const key = JSON.stringify([identity, source]);
  if (key === loadedReport) return;
  $('report-info').textContent = '';
  try {
    const report = await request('/api/report', {query: outputQuery()});
    if (key !== JSON.stringify([identity, source])) return;
    $('preview').srcdoc = '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src data:; style-src \'unsafe-inline\'">' + report.html;
    $('preview').hidden = false;
    loadedReport = key;
  } catch (error) {
    if (key === JSON.stringify([identity, source])) $('report-info').textContent = error.message;
  }
}));
$('download').onclick = () => manual(async () => {
  const filename = task.files[source].name;
  const blob = await request('/api/download', {query: outputQuery(), blob: true});
  const url = URL.createObjectURL(blob);
  const link = element('a');
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

labels();
const savedSize = Number(localStorage.getItem(sizeKey));
if (Number.isInteger(savedSize) && savedSize >= Number(sizeInput.min) && savedSize <= Number(sizeInput.max)) {
  sizeInput.value = savedSize;
}
applyTerminalSize();
document.querySelector(`nav a[href="${location.pathname}"]`)?.setAttribute('aria-current', 'page');
$('remote-tools').hidden = !remote;
$('finished-filters').hidden = !finished;
$('finished-pagination').hidden = !finished;
$('follow-task').hidden = finished;
async function poll() {
  if (!document.hidden && !manualRequests && !$('confirm').open) await refresh();
  setTimeout(poll, remote ? 5000 : finished ? 30000 : 3000);
}
poll();
