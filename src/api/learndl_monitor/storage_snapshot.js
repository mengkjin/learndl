/* Enhance the embedded view of both current and previously saved storage snapshots. */
(() => {
  const words = {
    title: ['存储报告', 'Storage report'], source: ['来源', 'Source'], root: ['扫描目录', 'Scan root'],
    usage: ['项目占用', 'Project usage'], free: ['磁盘可用', 'Disk free'], total: ['磁盘容量', 'Disk capacity'],
    entries: ['扫描条目', 'Entries scanned'], complete: ['扫描完整', 'Complete'], incomplete: ['部分统计', 'Partial scan'],
    started: ['开始', 'Started'], finished: ['完成', 'Finished'], notes: ['统计口径', 'How storage is measured'],
    explanation: [
      '占用量按已分配的磁盘块计算，包含目录自身。父目录已计入所有下级内容，请勿重复相加。符号链接不跟随，硬链接按扫描顺序只计一次占用。文件数按普通文件路径统计，硬链接各路径分别计数，目录和符号链接不计。日期范围取文件名末尾有效的 YYYYMMDD，包含深层文件，无日期文件不参与统计。部分统计条目的文件数和日期范围可能不完整。扫描期间文件可能变化。统计含挂载目录，磁盘可用空间仅指扫描根目录所在文件系统。',
      '',
    ],
    localBrowse: ['展开可读取扫描时保存的下级条目。', 'Expand to browse deeper entries from this scan.'],
    remoteBrowse: ['服务器报告可浏览至已收录的目录层级。', 'Browse the directory levels included in the server report.'],
    pathFilters: ['路径搜索与排序', 'Path search and sorting'], directoryList: ['目录列表', 'Directory listing'],
    search: ['搜索已载入路径', 'Search loaded paths'], placeholder: ['例如 DataBase 或 gru', 'e.g. DataBase or gru'],
    sort: ['排序', 'Sort by'], sizeOrder: ['占用量从大到小', 'Size (largest first)'], nameOrder: ['路径名称', 'Path name'],
    hidden: ['隐藏以 . 或 ~ 开头的名称', 'Hide names starting with . or ~'],
    expand: ['全部展开', 'Expand all'], collapse: ['全部折叠', 'Collapse all'],
    item: ['名称', 'Name'], size: ['占用量', 'Size'], share: ['占上级比例', 'Share of parent'],
    file: ['文件', 'File'], link: ['链接', 'Link'], rowIncomplete: ['统计不完整', 'Incomplete'],
    more: ['加载下一级', 'Load next level'], loading: ['加载中…', 'Loading…'],
    retry: ['加载失败，重试', 'Load failed — retry'],
    errors: ['扫描错误', 'Scan errors'], errorLimit: ['最多显示前 100 条错误。', 'Showing up to the first 100 errors.'],
  };
  let lang = 'en';
  const translated = [];
  const t = key => words[key][lang === 'zh' ? 0 : 1];
  function bind(node, key, transform = value => value) {
    translated.push(() => { node.textContent = transform(t(key)); });
    return node;
  }
  function element(name, className, value) {
    const node = document.createElement(name);
    if (className) node.className = className;
    if (value !== undefined) node.textContent = value;
    return node;
  }
  function labeledValue(label, value, container) {
    if (!value) return;
    const line = element('p');
    const caption = bind(element('span'), label, text => text + ' · ');
    line.append(caption, document.createTextNode(value));
    if (['started', 'finished'].includes(label)) {
      line.title = value;
      const date = new Date(value);
      if (!Number.isNaN(+date)) translated.push(() => {
        line.lastChild.textContent = date.toLocaleString(lang === 'zh' ? 'zh-CN' : 'en-US',
          {dateStyle: 'medium', timeStyle: 'short'});
      });
    }
    container.append(line);
  }

  const header = document.querySelector('header');
  const tree = document.querySelector('.tree');
  const footer = document.querySelector('footer');
  if (!header || !tree || !footer) return;
  // The stable standalone report labels also recover summary fields from legacy HTML.
  const lines = [...header.querySelectorAll(':scope > p')].map(node => node.textContent.trim());
  const field = label => lines.find(line => line.startsWith(label + ': '))?.slice(label.length + 2);
  const [usage, status, entries] = (field('Project usage') || '').split(' · ');
  const [total, free] = (field('Root filesystem') || '').split(' · ');
  const [started, finished] = (field('Started') || '').split(' · Finished: ');
  const heading = element('div', 'report-heading');
  heading.append(bind(element('h1'), 'title'));
  if (status) {
    const partial = status !== 'Status: Complete';
    heading.append(bind(element('span', 'report-status' + (partial ? ' partial' : '')), partial ? 'incomplete' : 'complete'));
  }
  const metrics = element('dl', 'report-metrics');
  for (const [key, value] of Object.entries({
    usage, free: free?.replace(/^Free /, ''), total: total?.replace(/^total /, ''), entries: entries?.replace(/^Scanned entries: /, ''),
  })) {
    if (!value) continue;
    const tile = element('div');
    tile.append(bind(element('dt'), key), element('dd', '', value));
    metrics.append(tile);
  }
  const details = element('div', 'report-meta');
  let source = field('Source');
  if (!source) {
    const metadata = document.getElementById('storage-metadata');
    try { source = JSON.parse(metadata?.textContent || '{}').source_host; } catch { /* Legacy snapshot. */ }
  }
  labeledValue('source', source, details);
  labeledValue('root', field('Root'), details);
  labeledValue('started', started, details);
  labeledValue('finished', finished, details);
  const notes = element('details', 'report-notes');
  const explanation = lines.find(line => line.startsWith('Sizes use allocated disk blocks')) || '';
  notes.append(bind(element('summary'), 'notes'));
  if (explanation) notes.append(bind(element('p'), 'explanation', value => value || explanation));
  const browse = document.getElementById('browse-status');
  if (browse) notes.append(bind(browse, storageBase ? 'localBrowse' : 'remoteBrowse'));
  const actions = element('div', 'report-actions');
  for (const id of ['search', 'sort', 'hide-special']) {
    const input = document.getElementById(id);
    if (!input) continue;
    const label = input.closest('label');
    const caption = bind(element('span'), id === 'hide-special' ? 'hidden' : id);
    if (id === 'search') translated.push(() => { input.placeholder = t('placeholder'); });
    if (id === 'sort') {
      for (const [value, key] of [['size', 'sizeOrder'], ['name', 'nameOrder']]) {
        const option = input.querySelector(`[value="${value}"]`);
        if (option) bind(option, key);
      }
    }
    label.replaceChildren(...(id === 'hide-special' ? [input, caption] : [caption, input]));
    actions.append(label);
  }
  const filters = element('details', 'report-filters');
  filters.append(bind(element('summary'), 'pathFilters'), actions);
  const listing = element('details', 'directory-list');
  listing.open = true;
  tree.before(listing);
  const treeActions = element('div', 'report-actions');
  listing.append(bind(element('summary'), 'directoryList'), treeActions, tree);
  for (const id of ['expand', 'collapse']) {
    const button = document.getElementById(id);
    if (button) treeActions.append(bind(button, id));
  }
  header.replaceChildren(heading, metrics, details, notes, filters);
  const columns = element('div', 'tree-head');
  for (const key of ['item', 'size', 'share']) columns.append(bind(element('span'), key));
  tree.prepend(columns);
  const errors = footer.querySelector('h2')?.textContent.split(': ').at(-1);
  if (errors !== undefined) {
    const errorDetails = element('details');
    errorDetails.append(bind(element('summary'), 'errors', label => label + ' · ' + errors));
    if (errors !== '0') errorDetails.append(bind(element('p'), 'errorLimit'), footer.querySelector('ul'));
    footer.replaceChildren(errorDetails);
  }

  const descriptions = new WeakMap();
  function localizeRows() {
    for (const label of tree.querySelectorAll('.entry-row > span:first-child, summary > span:first-child')) {
      if (label.querySelector('.node-name')) continue;
      const text = [...label.childNodes].filter(node => node.nodeType === Node.TEXT_NODE);
      if (!text.length) continue;
      const name = element('span', 'node-name', text.map(node => node.textContent).join('').trim());
      name.title = name.textContent;
      text[0].replaceWith(name);
      for (const node of text.slice(1)) node.remove();
    }
    for (const badge of tree.querySelectorAll('.kind')) {
      const key = ['Link', '链接'].includes(badge.textContent) ? 'link' : 'file';
      if (badge.textContent !== t(key)) badge.textContent = t(key);
    }
    for (const warning of tree.querySelectorAll('strong')) {
      if (warning.textContent !== t('rowIncomplete')) warning.textContent = t('rowIncomplete');
    }
    for (const description of tree.querySelectorAll('.description')) {
      if (!descriptions.has(description)) descriptions.set(description, description.textContent);
      const original = descriptions.get(description);
      const text = lang === 'zh' ? original.replace(' files', ' 个文件').replace('Dates:', '日期') :
        original.replace(/^1 files\b/, '1 file');
      if (description.textContent !== text) description.textContent = text;
      description.title = text;
    }
    for (const button of tree.querySelectorAll('.load-next')) {
      const reason = button.textContent.match(/^Load failed — retry \((.*)\)$/)?.[1];
      if (reason) button.dataset.reason = reason;
      const failure = button.dataset.reason;
      const label = button.disabled ? t('loading') : failure ? t('retry') + ' (' + failure + ')' : t('more');
      if (button.textContent !== label) button.textContent = label;
      button.title = failure || '';
    }
  }
  // Lazy browsing adds rows and updates its button in the existing report script.
  if (storageBase) {
    const observer = new MutationObserver(() => {
      observer.disconnect();
      localizeRows();
      observer.observe(tree, {childList: true, subtree: true, attributes: true, attributeFilter: ['disabled']});
    });
    observer.observe(tree, {childList: true, subtree: true, attributes: true, attributeFilter: ['disabled']});
  }
  function setLanguage(language) {
    lang = language;
    document.documentElement.lang = lang;
    document.title = t('title');
    translated.forEach(update => update());
    localizeRows();
  }
  window.addEventListener('message', event => {
    if (event.source !== window.parent || event.origin !== window.location.origin) return;
    if (event.data?.type === 'learndl-storage-language' && ['zh', 'en'].includes(event.data.language)) {
      setLanguage(event.data.language);
    }
  });
  setLanguage(lang);
})();
