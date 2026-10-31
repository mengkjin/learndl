"""Opt-in real Linux acceptance; only disposable files and transient user units.

LEARNDL_SYSTEMD_ACCEPTANCE=1 python -m unittest <this module> -v
Never starts installed Learndl services or executes the real autofix manifest.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import uuid


@unittest.skipUnless(sys.platform == 'linux' and os.getenv('LEARNDL_SYSTEMD_ACCEPTANCE') == '1',
                     'requires explicit Linux/systemd acceptance opt-in')
class LinuxAcceptance(unittest.TestCase):
    def test_bubblewrap_descendant_and_external_ledger_protection(self):
        from src.automation.scheduling.autofix_sandbox import check_tree, command
        with tempfile.TemporaryDirectory(prefix='.priority-acceptance-', dir=Path.cwd()) as temp:
            base = Path(temp).resolve()
            root = base / 'project'
            root.mkdir()
            (root / 'data').mkdir()
            (root / 'src').mkdir()
            (root / 'src' / 'protected.py').write_text('original')
            outside = base / 'outside'
            ledger = base / 'ledger.sqlite'
            outside.write_text('original')
            ledger.write_text('authority')
            (root / 'alias').symlink_to(outside)
            request = base / 'request.json'
            request.write_text('{}')
            check_tree(root)
            child = '''import sys
from pathlib import Path
for path in sys.argv[1:]:
    try:
        Path(path).write_text('bad')
    except OSError:
        continue
    raise AssertionError('escaped: ' + path)
'''
            parent = '''import subprocess,sys
from pathlib import Path
Path('data/allowed').write_text('ok')
subprocess.run([sys.executable, '-c', sys.argv[1], *sys.argv[2:]], check=True)
'''
            subprocess.run(command(root, request, [sys.executable, '-c', parent, child,
                           str(outside), str(ledger), str(root / 'alias'), str(root / 'src' / 'protected.py')]), check=True, timeout=30)
            self.assertEqual((root / 'data' / 'allowed').read_text(), 'ok')
            self.assertEqual((root / 'src' / 'protected.py').read_text(), 'original')
            self.assertEqual(outside.read_text(), 'original')
            self.assertEqual(ledger.read_text(), 'authority')

    def test_independent_oneshot_units_run_concurrently(self):
        subprocess.run(['systemctl', '--user', 'show-environment'], check=True, stdout=subprocess.DEVNULL, timeout=10)
        with tempfile.TemporaryDirectory(prefix='learndl-helper-acceptance-') as temp:
            root = Path(temp)
            units, processes = [], []
            code = '''import sys,time
from pathlib import Path
root = Path(sys.argv[1])
(root / sys.argv[2]).touch()
until = time.monotonic() + 10
while not (root / 'release').exists() and time.monotonic() < until:
    time.sleep(.05)
'''
            try:
                for name in ('mail', 'cli', 'cache'):
                    unit = 'learndl-acceptance-' + uuid.uuid4().hex
                    units.append(unit)
                    processes.append(subprocess.Popen(['systemd-run', '--user', '--quiet', '--wait', '--collect',
                        '--unit=' + unit, '--property=Type=oneshot', '--property=TimeoutStartSec=15s',
                        '--property=TimeoutStopSec=1s', '--property=KillMode=control-group',
                        '--property=NoNewPrivileges=yes', sys.executable, '-c', code, str(root), name]))
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline and not all((root / name).exists() for name in ('mail', 'cli', 'cache')):
                    time.sleep(.05)
                self.assertTrue(all((root / name).exists() for name in ('mail', 'cli', 'cache')))
                self.assertTrue(all(proc.poll() is None for proc in processes))
                (root / 'release').touch()
                for proc in processes:
                    self.assertEqual(proc.wait(timeout=10), 0)
            finally:
                for unit in units:
                    subprocess.run(['systemctl', '--user', 'stop', unit], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, check=False, timeout=10)
                for proc in processes:
                    if proc.poll() is None:
                        proc.kill()
                    proc.wait(timeout=10)

    def test_oneshot_timeout_stops_descendants(self):
        import psutil
        with tempfile.TemporaryDirectory(prefix='learndl-timeout-acceptance-') as temp:
            marker = Path(temp) / 'child.json'
            code = '''import subprocess,sys,time,json,signal
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])
Path(sys.argv[1]).write_text(json.dumps({'pid': child.pid}))
time.sleep(60)
'''
            result = subprocess.run(['systemd-run', '--user', '--quiet', '--wait', '--collect',
                '--unit=learndl-acceptance-' + uuid.uuid4().hex, '--property=Type=oneshot',
                '--property=TimeoutStartSec=2s', '--property=TimeoutStopSec=1s', '--property=KillMode=control-group',
                sys.executable, '-c', code, str(marker)], timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(marker.exists())
            pid = json.loads(marker.read_text())['pid']
            try:
                process = psutil.Process(pid)
            except psutil.NoSuchProcess:
                return
            self.assertEqual(process.status(), psutil.STATUS_ZOMBIE)


if __name__ == '__main__':
    unittest.main()
