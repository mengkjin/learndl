// Each service has its own origin (port). Carry the preference across service links.
const monitorNavigation = (() => {
  const key = 'learndl-language';
  const mobile = () => location.protocol === 'https:' && location.hostname.endsWith('.ts.net');
  function language() {
    const url = new URL(location.href);
    const requested = url.searchParams.get('lang');
    if (requested === 'zh' || requested === 'en') {
      localStorage.setItem(key, requested);
      url.searchParams.delete('lang');
      history.replaceState(history.state, '', url);
    }
    return localStorage.getItem(key) === 'en' ? 'en' : 'zh';
  }
  function address(value, lang) {
    const url = new URL(value, location.href);
    if (url.protocol === 'http:' && url.hostname === '127.0.0.1' && mobile()) {
      url.protocol = 'https:';
      url.hostname = location.hostname;
    }
    if (lang) url.searchParams.set('lang', lang);
    return url.href;
  }
  async function waitForRoute(value, lang) {
    if (!mobile()) return;
    const port = new URL(value, location.href).port || '443';
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 20000);
    try {
      while (!controller.signal.aborted) {
        const response = await fetch('/monitor-navigation.js', {cache:'no-store', signal:controller.signal});
        const ports = response.headers.get('X-LearnDL-Mobile-Ports');
        if (response.ok && (!ports || ports.split(',').includes(port))) return;
        await new Promise(resolve => setTimeout(resolve, 500));
      }
    } catch (error) {
      if (!controller.signal.aborted) throw error;
    } finally {
      clearTimeout(timer);
    }
    throw Error(lang === 'en' ? 'Mobile route is not ready yet. Try Open again.' : '手机访问入口尚未就绪，请稍后再点打开。');
  }
  async function open(value, lang) {
    if (!mobile()) { window.open(value, '_blank', 'noopener'); return; }
    const tab = window.open('', '_blank');
    if (!tab) throw Error(lang === 'en' ? 'Allow this site to open a new tab.' : '请允许此页面打开新标签页。');
    tab.opener = null;
    tab.document.body.textContent = lang === 'en' ? 'Connecting…' : '正在连接…';
    try {
      await waitForRoute(value, lang);
      if (!tab.closed) tab.location.replace(value);
    } catch (error) {
      tab.close();
      throw error;
    }
  }
  window.addEventListener?.('storage', event => {
    if (event.key !== key || !['zh', 'en'].includes(event.newValue)) return;
    const selector = document.querySelector('#language, #lang');
    if (selector && selector.value !== event.newValue) {
      selector.value = event.newValue;
      selector.dispatchEvent(new Event('change'));
    }
  });
  const toolbarLabels = {
    language:['页面语言','Page language'],
    settings:['设置','Settings'],
    refresh:['刷新','Refresh'],
    resources:['系统资源','System resources'],
    restart:['重启运行中的子服务','Restart running services'],
  };
  const iconPaths = {
    settings:'<circle cx="12" cy="12" r="3"/><path d="m10 3-.7 2.2-1.7 1-2.3-.5-2 3.5L5 11v2l-1.7 1.8 2 3.5 2.3-.5 1.7 1L10 21h4l.7-2.2 1.7-1 2.3.5 2-3.5L19 13v-2l1.7-1.8-2-3.5-2.3.5-1.7-1L14 3z"/>',
    refresh:'<path d="M20 10a8 8 0 0 0-14-4L3 9m0-5v5h5m-4 5a8 8 0 0 0 14 4l3-3m0 5v-5h-5"/>',
    resources:'<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M6 12h3l2-4 3 8 2-4h2"/>',
    restart:'<path d="M3 11a9 9 0 1 1 2.7 7.3M3 5v6h6m3-4v5"/>',
  };
  function apply(lang) {
    document.documentElement.lang = lang;
    const selector = document.querySelector('#language, #lang');
    const translated = key => toolbarLabels[key][lang === 'en' ? 1 : 0];
    selector.setAttribute('aria-label', translated('language'));
    for (const label of document.querySelectorAll?.('[data-monitor-label]') || []) {
      label.textContent = translated(label.dataset.monitorLabel);
    }
    for (const icon of document.querySelectorAll?.('[data-monitor-icon]') || []) {
      const name = icon.dataset.monitorIcon;
      icon.title = translated(name);
      icon.setAttribute('aria-label', icon.title);
      if (!icon.firstElementChild) {
        icon.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' + iconPaths[name] + '</svg>';
      }
    }
  }
  document.addEventListener?.('click', event => {
    if (event.target.closest('[data-monitor-icon=refresh]')) return;
    for (const menu of document.querySelectorAll('.monitor-menu[open]')) {
      if (!menu.contains(event.target)) menu.open = false;
    }
  });
  document.addEventListener?.('keydown', event => {
    if (event.key !== 'Escape') return;
    const menu = event.target.closest('.monitor-menu[open]') || document.querySelector('.monitor-menu[open]');
    if (menu) { menu.open = false; menu.querySelector('summary').focus(); }
  });
  return {language, address, open, waitForRoute, apply};
})();
