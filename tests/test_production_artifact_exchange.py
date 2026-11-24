"""Production deliveries use the shared outbox and the receiver's native layout."""
import io
import json
import os
import tempfile
import unittest
import zipfile
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

from src.automation.artifact_exchange import core, production


class ProductionArtifactExchangeTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.sender_root = self.root / 'server'
        self.receiver_root = self.root / 'mac'
        self.shared = self.root / 'shared'
        (self.shared / '.stfolder').mkdir(parents=True)
        self.sender = self.config(self.sender_root, 'publisher', 'server')
        self.receiver = self.config(self.receiver_root, 'receiver', 'mac')

    def config(self, root, role, host):
        return core.Config(enabled=True, role=role, shared_root=self.shared,
                           state=root / 'state', results=root / 'results', source_host=host)

    def file(self, name, data, stamp):
        path = self.sender_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        os.utime(path, ns=(stamp, stamp))
        return path

    def test_mixed_outputs_restore_to_native_paths_once(self):
        stamp = 1_700_000_000_123_456_789
        names = [
            'models/nn/gru@daily/configs/model.yaml',
            'models/nn/gru@daily/archive/0/20261123/best/state_dict.pt',
            'data/Export/model_prediction/gru_V1/2026/gru_V1.20261124.feather',
            'data/Export/trading_portfolio/use_daily/use_daily.20261124.feather',
            'data/Export/factor_model_port/gru_V1/gru_V1.20261124.feather',
            'data/Export/factor_model_account/gru_V1/top.csi500.tar',
            'results/trade/backtest/use_daily.backtest/portfolio/use_daily.backtest.20261124.feather',
            'results/trade/tracking/use_daily/use_daily_analytic_data.xlsx',
            'results/trade/trading_ports/trading_ports.20261124.csv',
        ]
        workbook = io.BytesIO()
        with zipfile.ZipFile(workbook, 'w') as archive:
            archive.writestr('xl/workbook.xml', '<workbook/>')
        paths = [self.file(name, workbook.getvalue() if name.endswith('.xlsx') else name.encode(), stamp + i)
                 for i, name in enumerate(names)]
        [record] = production.publish(paths, self.sender, self.sender_root)
        self.assertEqual(record['status'], 'published')
        self.assertEqual(record['route'], 'production/server')
        self.assertEqual(production.publish(paths, self.sender, self.sender_root), [])

        [received] = core.receive(self.receiver)
        self.assertEqual(received['installed'], len(names))
        self.assertEqual(list(self.sender.inbox.rglob('*.transfer.zip')), [])
        for i, name in enumerate(names):
            with self.subTest(name=name):
                target = self.receiver_root / name
                self.assertEqual(target.read_bytes(), paths[i].read_bytes())
                self.assertEqual(target.stat().st_mtime_ns, stamp + i)

    def test_late_package_and_local_newer_file_cannot_roll_back(self):
        name = 'data/Export/trading_portfolio/use_daily/use_daily.20261124.feather'
        stamp = 1_700_000_000_000_000_000
        old = self.file(name, b'old', stamp)
        [first] = production.publish([old], self.sender, self.sender_root)
        new = self.file(name, b'new', stamp + 1)
        [second] = production.publish([new], self.sender, self.sender_root)
        folder = self.sender.inbox / first['route']
        core._receive(folder / second['filename'], self.receiver)
        outcome = core._receive(folder / first['filename'], self.receiver)
        self.assertEqual(outcome['retained'], 1)
        target = self.receiver_root / name
        self.assertEqual(target.read_bytes(), b'new')

        target.write_bytes(b'local')
        os.utime(target, ns=(stamp + 10, stamp + 10))
        latest = self.file(name, b'remote', stamp + 2)
        production.publish([latest], self.sender, self.sender_root)
        [result] = core.receive(self.receiver)
        self.assertEqual(result['retained'], 1)
        self.assertEqual(target.read_bytes(), b'local')

    def test_unpublished_snapshot_retries_after_missing_share_marker(self):
        name = 'data/Export/model_prediction/gru_V1/2026/gru_V1.20261124.feather'
        path = self.file(name, b'complete', 1_700_000_000_000_000_000)
        (self.shared / '.stfolder').rmdir()
        [failed] = production.publish([path], self.sender, self.sender_root)
        self.assertEqual(failed['status'], 'failed')
        self.assertFalse((self.sender.state / 'production_published.json').exists())
        (self.shared / '.stfolder').mkdir()
        [retried] = production.publish([path], self.sender, self.sender_root)
        self.assertEqual(retried['status'], 'published')
        core.receive(self.receiver)
        self.assertEqual((self.receiver_root / name).read_bytes(), b'complete')

    def test_receiver_rejects_unsafe_but_well_formed_production_member(self):
        name = 'data/Export/model_prediction/gru_V1/2026/gru_V1.20261124.feather'
        path = self.file(name, b'payload', 1_700_000_000_000_000_000)
        [record] = production.publish([path], self.sender, self.sender_root)
        archive = self.sender.inbox / record['route'] / record['filename']
        with zipfile.ZipFile(archive) as source:
            manifest = json.loads(source.read('manifest.json'))
            content = source.read('payload/production/' + name)
        forbidden = 'production/configs/strategy/trading_port.yaml'
        manifest['artifacts'][0]['paths'] = [forbidden]
        manifest['files'][0]['path'] = forbidden
        with zipfile.ZipFile(archive, 'w') as output:
            output.writestr('manifest.json', json.dumps(manifest))
            output.writestr('payload/' + forbidden, content)
        [result] = core.receive(self.receiver)
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(archive.is_file())
        self.assertFalse((self.receiver_root / 'configs').exists())

    def test_only_completed_saves_and_final_model_files_are_selected(self):
        name = 'data/Export/factor_model_account/gru/top.tar'
        pending = Future()
        failed = Future()
        failed.set_result(False)
        with patch.object(production, '_publisher_safe', return_value=(self.sender, self.sender_root)):
            @production.batched
            def operation():
                path = self.sender_root / name
                production.saved(path, pending)
                production.saved(path.with_name('failed.tar'), failed)
                self.assertFalse(list(self.sender.inbox.rglob('*.transfer.zip')))
                self.file(name, b'closed', 1_700_000_000_000_000_000)
                pending.set_result(True)
            operation()
        core.receive(self.receiver)
        self.assertEqual((self.receiver_root / name).read_bytes(), b'closed')

        base = self.sender_root / 'models/nn/gru@daily'
        files = ['archive/0/20261123/best/state_dict.pt', 'archive/0/20261123/best/boost_head.pt',
                 'configs/model.yaml', 'configs/algo.gru.yaml']
        for path in [*files, 'archive/0/20261123/best/attempt1/state_dict.pt', 'checkpoint/state_dict.pt']:
            self.file((base / path).relative_to(self.sender_root).as_posix(), b'file', 1_700_000_000_000_000_000)
        self.assertCountEqual([path.relative_to(base).as_posix() for path in production.model_files(base)], files)
        with patch.object(production, 'sandboxed', return_value=True):
            self.assertEqual(production.publish(production.model_files(base), self.sender, self.sender_root), [])

    def test_unclosed_report_is_left_for_the_next_reconciliation(self):
        name = 'results/trade/tracking/use_daily/use_daily_analytic_plot.pdf'
        stamp = 1_700_000_000_000_000_000
        path = self.file(name, b'%PDF-1.7\nincomplete', stamp)
        self.assertEqual(production.publish([path], self.sender, self.sender_root), [])
        path = self.file(name, b'%PDF-1.7\nbody\n%%EOF\n', stamp + 1)
        self.assertEqual(production.publish([path], self.sender, self.sender_root)[0]['status'], 'published')
