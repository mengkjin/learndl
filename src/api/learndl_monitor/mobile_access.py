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
from typing import NamedTuple
from urllib.parse import urlsplit

import psutil

from src.automation.console import logger
from .access import GUEST_SERVICES, ROLE_HEADER
from .services import SERVICES, STATE

__all__ = ['main']

MOBILE = STATE / 'mobile'
CONFIG = MOBILE / 'httpd.conf'
HTTPD = '/usr/sbin/httpd'
TAILSCALE = '/usr/local/bin/tailscale'
GATEWAY = 'http://127.0.0.1:18590'


class Route(NamedTuple):
    port: int
    service: str


OPTUNA_READ = (r'(?:dashboard(?:/[a-zA-Z0-9_/-]*)?|favicon\.ico|static/[a-zA-Z0-9_./-]+|csv/[0-9]+|'
               r'api/(?:meta|studies(?:/[0-9]+(?:/param_importances|/plot/[a-z_]+)?)?|'
               r'compare-studies/plot/[a-z_]+))?')
# Optuna's rename endpoint copies the study and deletes the original.
OPTUNA_POST = (r'api/(?:studies(?:/[0-9]+/(?:preference(?:/[a-zA-Z0-9_-]+)?|[0-9]+/skip))?|'
               r'trials/[0-9]+/(?:tell|user-attrs))')
OPTUNA_PUT = r'api/studies/[0-9]+/(?:note|preference_feedback_component|[0-9]+/note)'
TENSORBOARD_READ = (r'(?:index\.html|index\.js|chart_worker\.js|icon_bundle\.svg|favicon\.ico|'
                    r'(?:font-roboto|tf-imports)/[a-zA-Z0-9_./-]+|'
                    r'data/(?:plugins_listing|plugin_entry\.html|environment|logdir|runs|experiments|'
                    r'experiment_runs|notifications|window_properties)|'
                    r'data/plugin/(?:scalars/(?:tags|scalars|scalars_multirun)|'
                    r'custom_scalars/(?:layout|scalars|download_data)|histograms/(?:tags|histograms)|'
                    r'distributions/(?:tags|distributions)|graphs/(?:graph|info|run_metadata)|'
                    r'images/(?:tags|images|individualImage)|audio/(?:tags|audio|individualAudio)|'
                    r'text/(?:tags|text)|pr_curves/(?:tags|pr_curves)|timeseries/(?:tags|timeSeries|imageData)|'
                    r'mesh/(?:tags|meshes|data)|hparams/download_data|'
                    r'projector/(?:runs|info|tensor|metadata|bookmarks|sprite_image|'
                    r'index\.js|projector_binary\.(?:js|html))))?')
TENSORBOARD_QUERY = r'data/plugin/(?:hparams/(?:experiment|session_groups|metric_evals)|timeseries/timeSeries)'


def record_route(path, service):
    """Use a registration only while it belongs to the original live process."""
    try:
        record = json.loads(path.read_text())
        process = psutil.Process(record['pid'])
        if process.create_time() != record['created']:
            return None
        if service == 'experiment-instance':
            service = record['kind']
            if service not in ('tensorboard', 'optuna'):
                return None
        port = record.get('port') or urlsplit(record['url']).port
        return Route(port, service) if isinstance(port, int) and 1024 <= port <= 65535 else None
    except (OSError, ValueError, KeyError, TypeError, psutil.Error):
        return None


def ports():
    records = [(path, 'experiment-instance') for path in (STATE / 'experiments').glob('[0-9]*.json')]
    records.extend((STATE / f'{key}.json', key) for key in SERVICES if key in GUEST_SERVICES)
    active = {route.port: route for path, service in records if (route := record_route(path, service))}
    console = record_route(STATE / 'console.json', 'console')
    if console:
        active.update({443: console, console.port: console})
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
              'TraceEnable Off', '<Location />', 'Require all denied', '</Location>']
    ips = ', '.join(f"'{ip}'" for ip in phones)
    ready_ports = ','.join(str(port) for port in sorted(routes.keys() & published)) or 'pending'
    for front, (backend, service) in sorted(routes.items()):
        authority = host if front == 443 else f'{host}:{front}'
        public = f'https://{authority}'
        local = f'http://127.0.0.1:{backend}'
        prefix = f'/p{front}/'
        options = ' upgrade=websocket retry=0' if service == 'compare' else ''
        lines += [
            f'ProxyPass {prefix} {local}/{options}',
            f'<Location "{prefix}">',
            '  <RequireAll>',
            f'    Require expr "req(\'X-Forwarded-For\') in {{{ips}}}"',
            f'    Require expr "req(\'X-Forwarded-Host\') == \'{authority}\'"',
            '    Require expr "req(\'X-Forwarded-Proto\') == \'https\' && -z req(\'Tailscale-Funnel-Request\')"',
            f'    Require expr "-z req(\'Origin\') || req(\'Origin\') == \'{public}\'"',
            '    Require expr "-z req(\'Connection\') || req(\'Connection\') =~ m#^(keep-alive|close|upgrade)$#i"',
            # Requests without Origin are safe only for ordinary reads.
            f'    Require expr "%{{REQUEST_METHOD}} in {{\'GET\', \'HEAD\', \'OPTIONS\'}} || req(\'Origin\') == \'{public}\'"',
        ]
        if service != 'compare':
            lines.append('    Require expr "-z req(\'Upgrade\')"')
        if service in ('tensorboard', 'optuna'):
            read_path = TENSORBOARD_READ if service == 'tensorboard' else OPTUNA_READ
            read = f"(%{{REQUEST_METHOD}} in {{'GET', 'HEAD'}} && %{{REQUEST_URI}} =~ m#^{prefix}{read_path}$#)"
            allowed = [('POST', TENSORBOARD_QUERY)] if service == 'tensorboard' else [
                ('POST', OPTUNA_POST), ('PUT', OPTUNA_PUT)]
            actions = [f"(%{{REQUEST_METHOD}} == '{method}' && %{{REQUEST_URI}} =~ m#^{prefix}{path}$#)"
                       for method, path in allowed]
            lines.append('    Require expr "' + ' || '.join([read, *actions]) + '"')
        lines += [
            '  </RequireAll>',
            f'  RequestHeader set {ROLE_HEADER} "guest"',
            f'  SetEnvIf Origin "^{re.escape(public)}$" mobile_same_origin',
            f'  RequestHeader set Origin "{local}" env=mobile_same_origin',
            f'  Header edit Location "^{re.escape(local)}(/|$)" "{public}$1"',
        ]
        if service == 'compare':
            # A remote Streamlit connection must begin its own session.
            lines.append('  RequestHeader edit Sec-WebSocket-Protocol "^([^,]+,[^,]+),.*$" "$1"')
        lines.append('</Location>')
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
    published = {443, *(SERVICES[key]['port'] for key in GUEST_SERVICES), *routes}
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
