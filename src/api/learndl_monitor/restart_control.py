"""Link child-service pages to the console's authenticated restart flow."""
from __future__ import annotations

import json
import re
from html import escape
from urllib.parse import urlencode, urlsplit

from .services import STATE

__all__ = ['restart_link']


def restart_link(service: str, port: int, *, state=STATE, fixed=False, browser_url=None, language=None) -> str:
    console_port = 8500
    try:
        record = json.loads((state / 'console.json').read_text())
        address = urlsplit(record['url'])
        if address.scheme == 'http' and address.hostname == '127.0.0.1' and address.port:
            console_port = address.port
    except (OSError, ValueError, KeyError, TypeError):
        pass
    fragment = urlencode({'restart-service': service, 'port': port})
    origin = f'http://127.0.0.1:{console_port}'
    if browser_url:
        browser = urlsplit(browser_url)
        if browser.scheme == 'https' and re.fullmatch(r'[a-z0-9-]+\.[a-z0-9.-]+\.ts\.net', browser.hostname or ''):
            origin = f'https://{browser.hostname}:{console_port}'
    query = '?' + urlencode({'lang': language}) if language in ('zh', 'en') else ''
    address = escape(f'{origin}/{query}#{fragment}', quote=True)
    caption = {'en': 'Restart service', 'zh': '重启服务'}.get(language, '重启服务 / Restart service')
    position = 'position:fixed;right:.8rem;top:3.5rem;z-index:999;' if fixed else ''
    return (f'<a class="service-restart" href="{address}" target="_self" '
            f'title="{caption}" aria-label="{caption}" '
            f'style="{position}display:inline-flex;align-items:center;justify-content:center;'
            'width:2.35rem;height:2.35rem;border:1px solid #344355;border-radius:7px;'
            'background:#202c3a;color:#dce8ee;font:22px/1 system-ui;text-decoration:none">↻</a>')
