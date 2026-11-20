"""Regression checks for directory migration, protocol compatibility and UI state."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.automation.artifact_exchange import core
from src.automation.version_patch import load_patch_support


class ResultsLayoutTest(unittest.TestCase):
    def test_old_model_route_migration_and_interrupted_commit(self):
        layout = load_patch_support('0.7.4', 'results')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            shared = root/'share'
            (shared/'.stfolder').mkdir(parents=True)
            pub = core.Config(True, 'publisher', shared, root/'pub', root/'server-results', source_host='server')
            rec = core.Config(True, 'receiver', shared, root/'rec', root/'results')
            source = root/'report.txt'
            source.write_text('test')
            row, = core.publish_artifacts('train', 'run', [
                core.ArtifactSpec(source, category='model_training', experiment='nn@gru')], config=pub)
            archive, = pub.inbox.rglob('*.transfer.zip')
            self.assertEqual(archive.parent.relative_to(pub.inbox), Path('model/nn@gru'))
            old_route = pub.inbox/'model_training/nn@gru'/archive.name
            old_route.parent.mkdir(parents=True)
            archive.rename(old_route)
            replay = old_route.read_bytes()
            outcome, = core.receive(rec)
            new = Path(outcome['target'])
            self.assertEqual(new.parent, rec.results/'external/models/server/nn@gru')
            old = core._target(row['manifest'], rec)
            old.parent.mkdir(parents=True)
            new.rename(old)
            receipt = rec.state/'received'/(row['id']+'.json')
            old_receipt = json.loads(receipt.read_text()) | {'target': str(old)}
            core.atomic_json(receipt, old_receipt)
            monitor = root/'monitor'
            marks = monitor/'model_preview/archived.json'
            def key(path):
                return hashlib.sha256(str(path).encode()).hexdigest()
            core.atomic_json(marks, [key(old/'results/report.txt')])
            trash = monitor/'model_preview/trash/one/item.json'
            core.atomic_json(trash, {'original': str(old/'results/deleted.txt')})
            legacy = rec.results/'reports/storage/legacy/storage_reports'
            legacy.mkdir(parents=True)
            (legacy/'storage.html').write_text('html')
            (legacy/'storage.sqlite3').write_bytes(b'index')
            (rec.results/'storage_reports').symlink_to(legacy)

            preflight = layout.discover(rec, monitor)
            self.assertTrue(old.exists() and not new.exists())
            self.assertIn(str(old), preflight['redirects'])
            journal = rec.state/'migrations/results-0.7.4.json'
            def interrupted(path, value):
                if path == journal and any(r.get('done') for r in value['moves']):
                    raise OSError('interrupted after rename')
                core.atomic_json(path, value)
            with patch.object(layout, 'atomic_json', side_effect=interrupted), self.assertRaises(OSError):
                layout.apply(rec, monitor, root/'attempt-1.json')
            layout.apply(rec, monitor, root/'attempt-2.json')
            layout.apply(rec, monitor, root/'attempt-3.json')
            self.assertEqual({p.name for p in rec.results.iterdir()}, layout.KEEP)
            self.assertEqual(json.loads(receipt.read_text()), old_receipt | {'target': str(new)})
            self.assertEqual(json.loads(marks.read_text()), [key(new/'results/report.txt')])
            self.assertEqual(json.loads(trash.read_text())['original'], str(new/'results/deleted.txt'))
            self.assertEqual((rec.results/'custom/storage_reports/storage.sqlite3').read_bytes(), b'index')
            old_route.parent.mkdir(parents=True, exist_ok=True)
            old_route.write_bytes(replay)
            self.assertEqual(core.receive(rec)[0]['target'], str(new))

    def test_server_conflict_keeps_both_files(self):
        layout = load_patch_support('0.7.4', 'results')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            config = core.Config(results=root/'results', state=root/'state', role='publisher')
            src = config.results/'rl_experimental/checkpoint'
            dst = config.results/'custom/rl_experimental/checkpoint'
            for path, content in [(src, 'original'), (dst, 'occupied')]:
                path.parent.mkdir(parents=True)
                path.write_text(content)
            with self.assertRaises(FileExistsError):
                layout.apply(config, root/'monitor', root/'attempt.json')
            self.assertEqual((src.read_text(), dst.read_text()), ('original', 'occupied'))
            dst.rename(dst.with_suffix('.saved'))
            layout.apply(config, root/'monitor', root/'attempt.json')
            self.assertEqual(dst.read_text(), 'original')
