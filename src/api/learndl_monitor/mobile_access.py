"""Run a private macOS reverse proxy for LearnDL pages behind Tailscale Serve."""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import signal
import subprocess
import time
from urllib.parse import urlsplit

import psutil

from src.automation.console import logger
from .services import SERVICES, STATE

__all__ = ['main']

MOBILE = STATE / 'mobile'
CONFIG = MOBILE / 'httpd.conf'
HTTPD = '/usr/sbin/httpd'
TAILSCALE = '/usr/local/bin/tailscale'
GATEWAY = 'http://127.0.0.1:18590'


def record_port(path):
    """Use a registration only while it belongs to the original live process."""
    try:
        record = json.loads(path.read_text())
        process = psutil.Process(record['pid'])
        if process.create_time() != record['created']:
            return None
        if path.parent.name == 'experiments' and record['kind'] not in ('tensorboard', 'optuna'):
            return None
        port = record.get('port') or urlsplit(record['url']).port
        return port if isinstance(port, int) and 1024 <= port <= 65535 else None
    except (OSError, ValueError, KeyError, TypeError, psutil.Error):
        return None


def ports():
    console = record_port(STATE / 'console.json')
    records = [STATE / f'{key}.json' for key in SERVICES]
    records.extend((STATE / 'experiments').glob('[0-9]*.json'))
    active = {port: port for path in records if (port := record_port(path))}
    if console:
        active.update({443: console, console: console})
    return active


def apache_config(host, phones, routes, published):
    modules = ('mpm_event', 'unixd', 'authz_core', 'headers', 'setenvif', 'proxy',
               'proxy_http')
    lines = [f'ServerRoot "{MOBILE}"', f'DefaultRuntimeDir "{MOBILE}"',
             f'Mutex "file:{MOBILE}" default', f'PidFile "{MOBILE}/httpd.pid"',
             f'ErrorLog "{MOBILE}/httpd.log"', 'LogLevel warn', f'ServerName {host}',
             'Listen 127.0.0.1:18590', 'ServerTokens Prod', 'ServerSignature Off']
    for mod in modules:
        lines.append(f'LoadModule {mod}_module /usr/libexec/apache2/mod_{mod}.so')
    lines += ['ProxyRequests Off', 'ProxyAddHeaders Off', 'ProxyTimeout 180',
              '<Location />', 'Require all denied', '</Location>']
    ips = ', '.join(f"'{ip}'" for ip in phones)
    ready_ports = ','.join(str(port) for port in sorted(routes.keys() & published)) or 'pending'
    for front, backend in sorted(routes.items()):
        authority = host if front == 443 else f'{host}:{front}'
        public = f'https://{authority}'
        local = f'http://127.0.0.1:{backend}'
        prefix = f'/p{front}/'
        options = ' upgrade=websocket retry=0' if front == 443 or backend != routes.get(443) else ''
        lines += [
            f'ProxyPass {prefix} {local}/{options}',
            f'<Location "{prefix}">',
            '  <RequireAll>',
            f'    Require expr "req(\'X-Forwarded-For\') in {{{ips}}}"',
            f'    Require expr "req(\'X-Forwarded-Host\') == \'{authority}\'"',
            '    Require expr "req(\'X-Forwarded-Proto\') == \'https\' && -z req(\'Tailscale-Funnel-Request\')"',
            f'    Require expr "-z req(\'Origin\') || req(\'Origin\') == \'{public}\'"',
            # Requests without Origin are safe only for ordinary reads.
            f'    Require expr "%{{REQUEST_METHOD}} in {{\'GET\', \'HEAD\', \'OPTIONS\'}} || req(\'Origin\') == \'{public}\'"',
            '  </RequireAll>',
            f'  SetEnvIf Origin "^{re.escape(public)}$" mobile_same_origin',
            f'  RequestHeader set Origin "{local}" env=mobile_same_origin',
            f'  Header edit Location "^{re.escape(local)}(/|$)" "{public}$1"',
            '</Location>',
        ]
        # The browser checks the source service before opening a newly registered port.
        # Read this header through the same authenticated route as the page asset.
        lines += [f'<Location "{prefix}monitor-navigation.js">',
                  f'  Header set X-LearnDL-Mobile-Ports "{ready_ports}"', '</Location>']
    return '\n'.join(lines) + '\n'


def serve_config():
    return json.loads(subprocess.check_output([TAILSCALE, 'serve', 'status', '--json'], text=True, timeout=10))


def reconcile(host, routes):
    """Only add or remove our own routes. Never replace an unrelated Serve entry."""
    current = serve_config()
    # Reserved project ports can be published before service startup. Apache
    # only enables forwarding once the project registers the live process.
    published = {443, *(s['port'] for s in SERVICES.values()), *routes}
    ours_before = set()
    for name, web in current.get('Web', {}).items():
        server, _, port = name.rpartition(':')
        proxy = web.get('Handlers', {}).get('/', {}).get('Proxy')
        if server == host and port.isdecimal() and proxy == f'{GATEWAY}/p{port}/':
            ours_before.add(int(port))
    for port in sorted(published | ours_before):
        name = f'{host}:{port}'
        target = f'{GATEWAY}/p{port}/'
        handlers = current.get('Web', {}).get(name, {}).get('Handlers', {})
        ours = handlers.get('/', {}).get('Proxy') == target
        if port not in published:
            if ours:
                subprocess.run([TAILSCALE, 'serve', f'--https={port}', 'off'], check=True, timeout=15)
            continue
        if current.get('AllowFunnel', {}).get(name):
            raise RuntimeError(f'Public Funnel already uses {name}')
        if ours:
            continue
        placeholder = port == 443 and handlers == {'/': {'Text': 'LearnDL'}}
        if not placeholder and (handlers or str(port) in current.get('TCP', {})):
            raise RuntimeError(f'Another Serve listener already uses {name}')
        subprocess.run([TAILSCALE, 'serve', '--bg', f'--https={port}', target],
                       check=True, timeout=15)
    return published


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--phone', action='append', required=True)
    args = parser.parse_args()
    host = args.host.lower().rstrip('.')
    if not re.fullmatch(r'[a-z0-9-]+\.[a-z0-9.-]+\.ts\.net', host):
        parser.error('Expected a Tailscale DNS hostname')
    phones = [str(ipaddress.ip_address(phone)) for phone in args.phone]
    os.umask(0o077)
    MOBILE.mkdir(mode=0o700, parents=True, exist_ok=True)
    worker = None
    last = last_published = None
    published = set()
    next_publish = 0
    config_text = ''
    def shutdown(*_):
        raise SystemExit
    signal.signal(signal.SIGTERM, shutdown)
    try:
        while True:
            routes = ports()
            if routes != last:
                next_publish = 0
            if routes != last or published != last_published:
                updated = apache_config(host, phones, routes, published)
                draft = CONFIG.with_suffix('.new')
                draft.write_text(updated)
                subprocess.run([HTTPD, '-t', '-f', str(draft)], check=True)
                draft.replace(CONFIG)
                if worker is None:
                    worker = subprocess.Popen([HTTPD, '-DFOREGROUND', '-f', str(CONFIG)])
                elif updated != config_text:
                    worker.send_signal(signal.SIGUSR1)
                last, config_text, last_published = routes, updated, published
            if worker is not None and worker.poll() is not None:
                raise RuntimeError(f'Private gateway exited with status {worker.returncode}')
            if time.monotonic() >= next_publish:
                try:
                    status = json.loads(subprocess.check_output(
                        [TAILSCALE, 'status', '--json'], text=True, timeout=10))
                    if host in (status.get('CertDomains') or []):
                        published = reconcile(host, routes)
                        next_publish = time.monotonic() + 300
                    else:
                        published = set()
                        next_publish = time.monotonic() + 8
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                    logger.error('Could not reconcile private mobile access: %s', exc)
                    next_publish = time.monotonic() + 15
            time.sleep(2)
    finally:
        if worker is not None and worker.poll() is None:
            worker.terminate()
            worker.wait(timeout=8)


if __name__ == '__main__':
    main()
