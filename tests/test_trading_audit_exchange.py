import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

from src.automation.artifact_exchange.core import Config
from src.automation.artifact_exchange.trading_audit import publish


class TradingAuditExchangeTest(unittest.TestCase):
    def test_publication_is_bounded_and_keeps_originals(self):
        import src.proj
        from src.proj.db.interface.db_path import DBPath

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config = Config(enabled=True, role='publisher', shared_root=root / 'shared',
                            state=root / 'state', results=root / 'results', source_host='server')
            (config.shared_root / '.stfolder').mkdir(parents=True)
            portfolios = root / 'portfolio'
            recent = portfolios / 'use_daily' / 'use_daily.20261124.feather'
            recent.parent.mkdir(parents=True)
            recent.write_bytes(b'current portfolio')
            recent.with_name('use_daily.20260101.feather').write_bytes(b'old portfolio')
            prediction = root / 'gru_day_V1.20261120.feather'
            prediction.write_bytes(b'prediction')
            days = [20261123, 20261124]
            calendar = SimpleNamespace(updated=lambda: days[-1],
                                       td=lambda date, offset: SimpleNamespace(td=days[0]),
                                       range=lambda *args, **kwargs: days)
            with patch.object(src.proj, 'CALENDAR', calendar), \
                    patch.object(src.proj, 'PATH', SimpleNamespace(trade_port=portfolios)), \
                    patch.object(DBPath, 'get_paths', return_value={20261120: prediction}) as get_paths, \
                    patch('src.automation.artifact_exchange.trading_audit.sandboxed', return_value=False):
                self.assertIn('published through 20261124', publish(config=config))
                get_paths.assert_called_once_with(days, closest=True)
                target = config.shared_root / 'persistent' / 'trading_audit' / 'server' / 'latest.zip'
                with ZipFile(target) as archive:
                    self.assertIsNone(archive.testzip())
                    self.assertEqual(set(archive.namelist()), {
                        'manifest.json',
                        'portfolio/use_daily/use_daily.20261124.feather',
                        'pred/gru_day_V1/gru_day_V1.20261120.feather',
                    })
                    manifest = json.loads(archive.read('manifest.json'))
                    self.assertEqual(manifest['predictions'], {'gru_day_V1': [20261120]})
                    self.assertEqual(manifest['portfolios']['use_daily'], [20261124])
                self.assertEqual(recent.read_bytes(), b'current portfolio')
                self.assertEqual(prediction.read_bytes(), b'prediction')
                original = target.read_bytes()
                recent.unlink()
                self.assertIn('no use_daily portfolio', publish(config=config))
                self.assertEqual(target.read_bytes(), original)
