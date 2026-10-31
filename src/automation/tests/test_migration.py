"""Upgrade compatibility without production state, services, SMTP or training."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from src.automation.paths import ROOT, config_directory
from src.automation.scheduling import channels, priority
from src.automation.scheduling.autofix_store import Store, initialize, state_dir
from src.automation.scheduling.config import load_config
from src.automation.scheduling.installer import build_plan, service, validate_unit


class MigrationTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name).resolve()

    def test_aliases_share_module_and_lock_context_identity(self):
        names = ['watchdog', 'cli_recovery'] + [
            'scheduling.' + p.stem for p in (ROOT/'src/automation/scheduling').glob('*.py')
            if p.stem != '__init__']
        for name in names:
            with self.subTest(module=name):
                old = importlib.import_module('src.api.task_monitor.' + name)
                new = importlib.import_module('src.automation.' + name)
                self.assertIs(old, new)
        old = importlib.import_module('src.api.task_monitor.scheduling.maintenance_gate')
        new = importlib.import_module('src.automation.scheduling.maintenance_gate')
        self.assertIs(old._depth, new._depth)
        self.assertIs(old._lane_fds, new._lane_fds)

    def test_builtin_alias_and_custom_configuration(self):
        expected = load_config(ROOT/'configs/automation', 'mengkjin-server')
        self.assertEqual(load_config(ROOT/'runs/scheduling', 'mengkjin-server'), expected)
        custom = self.base/'custom'
        shutil.copytree(ROOT/'configs/automation', custom)
        config = custom/'maintenance.yaml'
        config.write_text(config.read_text().replace('max_sources: 20', 'max_sources: 21'))
        self.assertEqual(config_directory(custom), custom)
        self.assertEqual(load_config(custom, 'mengkjin-server')['watchdog']['jobs']['task_monitor_cache']['max_sources'], 21)
        self.assertEqual(config_directory(), ROOT/'configs/automation')

    def test_control_directory_and_successful_request_survive_rename(self):
        old = importlib.import_module('src.api.task_monitor.scheduling.autofix_store')
        root = self.base/'project'
        key = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16]
        self.assertEqual(state_dir(root), Path.home()/'.local/state/learndl'/key/'autofix')
        self.assertEqual(old.ROOT, ROOT)
        directory = self.base/'control'
        old.initialize(directory, root)
        store = old.Store(directory)
        item = {'id': 'completed-repair', 'script': 'scripts/repair.py', 'params': {}}
        store.register([item], 'before')
        store.update(item['id'], attempted=True, phase='success')
        store.enqueue('daily_update', 100)
        inode = (directory/'gate.lock').stat().st_ino
        with store.connect() as db:
            fingerprint = db.execute('SELECT fingerprint FROM fixes').fetchone()[0]
        initialize(directory, root)
        new = Store(directory)
        new.register([item], 'after')
        saved = new.get(item['id'])
        if not isinstance(saved, dict):
            raise AssertionError('missing migrated fix')
        self.assertEqual(saved['phase'], 'success')
        self.assertFalse(saved['conflict'])
        self.assertEqual(new.scheduled(), [('daily_update:100', 'daily_update')])
        self.assertEqual((directory/'gate.lock').stat().st_ino, inode)
        with new.connect() as db:
            self.assertEqual(db.execute('SELECT fingerprint FROM fixes').fetchone()[0], fingerprint)

    def test_new_and_legacy_service_snapshots_validate(self):
        config = load_config(config_directory(), 'mengkjin-server')
        plan = build_plan(config, '', ROOT, Path(sys.executable), config['timezone'])
        for name, content in plan['units'].items():
            validate_unit(name, content)
            validate_unit(name, content.replace('src.automation.', 'src.api.task_monitor.'))
        # Older direct-module startup commands are also valid rollback targets.
        for name, module, args in (
            ('learndl-watchdog.service', 'watchdog', []),
            ('learndl-schedule-daily_update.service', 'scheduling.runner', ['run', 'daily_update']),
            ('learndl-idle-worklist.service', 'scheduling.idle_worklist', ['worker']),
            ('learndl-git-update.service', 'scheduling.git_update', []),
        ):
            content = service(ROOT, Path(sys.executable), 'src.api.task_monitor.' + module, args)
            validate_unit(name, content)
            with self.assertRaises(ValueError):
                validate_unit(name, content.replace('src.api.task_monitor.', 'untrusted.'))

    def test_old_and_new_cli_entrypoints_remain_executable(self):
        root = self.base/'offline-project'
        (root/'configs/automation').mkdir(parents=True)
        (root/'configs/automation/autofix.yaml').write_text('schema_version: 1\nfixes: []\n')
        for prefix in ('src.automation', 'src.api.task_monitor'):
            for module in ('scheduling.installer', 'scheduling.autofix', 'cli_recovery'):
                result = subprocess.run([sys.executable, '-m', prefix+'.'+module, '--help'],
                                        cwd=ROOT, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([sys.executable, '-m', prefix+'.scheduling.autofix',
                                     'plan', '--root', str(root)], cwd=ROOT,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
        for path in ('src/automation/cli_recovery.py', 'src/api/task_monitor/cli_recovery.py'):
            result = subprocess.run([sys.executable, str(ROOT/path), '--help'],
                                    cwd=self.base, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_copied_legacy_bootstrap_dispatches_through_compatibility_module(self):
        from src.automation.scheduling import worker_bootstrap
        root = self.base/'project'
        runtime = self.base/'runtime'
        runtime.mkdir()
        directory = self.base/'control'
        initialize(directory, root)
        store = Store(directory)
        priority.initialize(store)
        channels.initialize(store)
        (runtime/'installed.json').write_text(json.dumps({'scheduler_version': 3}))
        (directory/'worker-settings.json').write_text(json.dumps({'root': str(root), 'runtime': str(runtime)}))
        # Reproduce the old installed target spelling without invoking production workers.
        (directory/'worker_bootstrap.py').write_text(Path(worker_bootstrap.__file__).read_text().replace(
            'src.automation.', 'src.api.task_monitor.'))
        old = root/'src/api/task_monitor/scheduling'
        new = root/'src/automation/scheduling'
        old.mkdir(parents=True)
        new.mkdir(parents=True)
        (root/'src/__init__.py').touch()
        shutil.copyfile(ROOT/'src/api/task_monitor/scheduling/auxiliary.py', old/'auxiliary.py')
        marker = root/'executed'
        (new/'auxiliary.py').write_text(f'from pathlib import Path\nPath({str(marker)!r}).touch()\n')
        subprocess.run([sys.executable, str(directory/'worker_bootstrap.py'), 'email_outbox'],
                       cwd=self.base, check=True, timeout=30)
        self.assertTrue(marker.exists())

    def test_offline_import_does_not_initialize_project_or_home(self):
        home = self.base/'untouched-home'
        code = ('import sys; import src.automation; from src.automation.paths import config_directory; '
                'from src.automation.scheduling.config import load_config; '
                'load_config(config_directory(), "mengkjin-server"); assert "src.proj" not in sys.modules')
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True,
                                env=dict(os.environ, HOME=str(home), PYTHONDONTWRITEBYTECODE='1'), timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(home.exists())
