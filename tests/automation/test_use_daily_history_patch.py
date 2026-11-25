import sqlite3
import tempfile
import unittest
from email.message import EmailMessage
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.automation.version_patch import load_patch_support

history = load_patch_support('0.7.14', 'use_daily_history')
old = load_patch_support('0.7.11', 'sellside_portfolios')


class UseDailyHistoryTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.ctx = SimpleNamespace(root=self.root, backup_dir=self.root / 'attempts/new/backup')
        self.ctx.backup_dir.mkdir(parents=True)
        self.log = self.root / 'attempts/old/patch.log'
        with sqlite3.connect(self.root / 'ledger.sqlite') as db:
            db.execute('create table attempts(log_path, started, patch, attempt)')
            db.execute('insert into attempts values(?,?,?,?)', (str(self.log), 1800000000, '0.7.11/001', 1))
        for attr, path in [('production', self.root), ('rslt_trade', self.root/'results/trade'),
                           ('trade_port', self.root/'data/Export/trading_portfolio'),
                           ('email_outbox', self.root/'outbox'), ('rslt_external', self.root/'external')]:
            self.enterContext(patch.object(history.PATH, attr, path))
        self.enterContext(patch.object(history, 'load_patch_support', return_value=old))
        self.enterContext(patch.object(history.MACHINE, 'platform_server', True))
        self.enterContext(patch.object(old.CALENDAR, 'updated', return_value=20261125))
        self.enterContext(patch.object(old, '_check_vendor_history'))
        self.enterContext(patch.object(old, '_publish'))
        self.port = old.TrackingPort.load('use_daily')
        self.port.portfolio_dir.mkdir(parents=True)
        for date in [20260105, 20261123, 20261124, 20261125]:
            self.frame(99).to_feather(self.port.export_path(date))
        saved = self.log.parent/'backup/original'/self.port.export_path(20260105).relative_to(self.root)
        saved.parent.mkdir(parents=True)
        self.frame(7).to_feather(saved)

    def frame(self, secid):
        return pd.DataFrame(dict(secid=[secid], weight=[1.], value=[100.]))

    def test_missing_seed_aborts_before_install(self):
        with self.assertRaisesRegex(ValueError, '20261123'):
            history.apply(self.ctx)
        self.assertEqual(pd.read_feather(self.port.export_path(20260105)).secid.tolist(), [99])

    def test_seed_from_mail_and_backup_then_chain(self):
        message = EmailMessage()
        message['Subject'] = 'Learndl [Ubuntu]: Success - Daily Update - 20261123183004'
        message.set_content('holdings')
        rows = self.frame(11).assign(name='use_daily', date=20261123)
        message.add_attachment(rows.to_csv().encode(), maintype='text', subtype='csv',
                               filename='trading_ports.20261123.csv')
        history.PATH.email_outbox.mkdir()
        (history.PATH.email_outbox/'message.eml').write_bytes(message.as_bytes())

        def build(port, date, *, last_port, export):
            frame = self.frame(int(last_port.to_dataframe().secid.iloc[0]) + 1).assign(name=port.name, date=date)
            port.save_port(frame, date)
            return frame

        def analyze(port, **kwargs):
            port.result_dir.mkdir(parents=True)
            for path in (port.result_path_account, port.result_path_data, port.result_path_plot):
                path.write_bytes(b'report')

        with patch.object(old.StagedTrackingPort, 'build_portfolio', build), \
                patch.object(old.StagedTrackingPort, 'analyze', analyze), \
                patch.object(old.production, 'finished_report', return_value=True), \
                patch('src.res.factor.util.classes.alpha_model.AlphaComposite.get_alpha_models'):
            history.apply(self.ctx)
            history.apply(self.ctx)
        self.assertEqual([pd.read_feather(self.port.export_path(date)).secid.item()
                          for date in [20260105, 20261123, 20261124, 20261125]], [7, 11, 12, 13])
        self.assertTrue(history.verify(self.ctx))
