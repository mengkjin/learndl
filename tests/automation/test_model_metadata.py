from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.api.learndl_monitor.artifact_library import Library
from src.automation.artifact_exchange.core import ArtifactSpec, Config, atomic_json, publish_artifacts, receive
from src.automation.artifact_exchange.model_metadata import BACKFILL_TASK, metadata_specs
from src.model_provenance import read_properties
from src.res.model.util.model_properties import Phase
from src.automation.version_patch import load_patch_support


MODEL = 'nn@gru@metadata'


class ModelMetadataTest(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        shared = self.root / 'share'
        (shared / '.stfolder').mkdir(parents=True)
        self.pub = Config(True, 'publisher', shared, self.root / 'pub', self.root / 'server', source_host='server')
        self.rec = Config(True, 'receiver', shared, self.root / 'rec', self.root / 'results', source_host='client')
        self.base = self.root / 'models/nn/gru@metadata'
        (self.base / 'configs').mkdir(parents=True)
        self.config_file = self.base / 'configs/model.yaml'
        self.config_file.write_text('model.module: gru\n')
        (self.base / 'configs/lifecycle.yaml').write_text('stage: exploration\n')
        self.library = Library('model_preview', self.rec, self.root / 'library')

    def publish_metadata(self, run):
        with metadata_specs(self.pub, self.base, MODEL) as (specs, baseline, inventory):
            if not specs:
                return None
            outcome, = publish_artifacts('model_training', run, [
                replace(spec, experiment=MODEL) for spec in specs
            ], config=self.pub)
            self.assertEqual(outcome['status'], 'published', outcome)
            atomic_json(baseline, inventory)
            return outcome

    def test_latest_snapshot_resolves_after_out_of_order_arrival(self):
        before = self.publish_metadata('first')
        self.assertIsNotNone(before)
        assert before is not None
        archive = self.pub.inbox / before['route'] / before['filename']
        saved = self.root / archive.name
        archive.rename(saved)

        (self.base / 'configs/lifecycle.yaml').write_text('stage: production\n')
        after = self.publish_metadata('second')
        assert after is not None
        self.assertNotIn('configs/model.yaml', {f['path'] for f in after['manifest']['files']})
        self.assertIsNone(self.publish_metadata('unchanged'))

        with patch('src.res.model.util.core.model_path.source_models', return_value=[]):
            self.assertEqual(receive(self.rec)[0]['status'], 'received')
            model = next(iter(self.library.model_listing()['models'].values()))
            self.assertEqual(model['versions'][0]['missing'], 1)
            archive.parent.mkdir(parents=True, exist_ok=True)
            saved.rename(archive)
            self.assertEqual(receive(self.rec)[0]['status'], 'received')
            listing = self.library.model_listing()
            model = next(iter(listing['models'].values()))
            self.assertEqual(model['stage'], 'production')
            self.assertEqual(len(model['versions']), 2)
            self.assertEqual(len(model['versions'][0]['files']), 2)
            self.assertEqual(model['versions'][0]['missing'], 0)
            self.assertEqual(self.library.model_folder(model['id']), self.rec.results / 'external/models/server' / MODEL)
            selection = next(f for f in listing['files'] if f['label'] == 'model.yaml')
            self.assertIn('read-only', self.library.manage('delete', [selection['id']])[0]['error'])

            self.config_file.unlink()
            self.publish_metadata('deletion')
            self.assertEqual(receive(self.rec)[0]['status'], 'received')
            model = next(iter(self.library.model_listing()['models'].values()))
            self.assertEqual(len(model['versions'][0]['files']), 1)

    def test_phase_summary_only_advances_for_effective_work(self):
        trainer = SimpleNamespace(base_path=SimpleNamespace(base=self.base),
                                  config=SimpleNamespace(is_resuming=False))
        def phase(run_id, kind, dates=None, status='success'):
            trainer.training_run = SimpleNamespace(run_id=run_id, data={'git': {'head': 'head-' + run_id}})
            run = Phase(trainer, kind)
            if dates is not None:
                run.worked(*dates)
            run.finish(status)
            return read_properties(run.path)

        first = phase('first', 'initial_training', [20250101, 20250201])['initial_training']
        trainer.config.is_resuming = True
        phase('resume1', 'initial_training', [20250115, 20250401])
        phase('resume2', 'initial_training', [20250402, 20250501])
        phase('resume2', 'initial_training', [20250402, 20250501])
        records = phase('no-work', 'initial_training')
        self.assertEqual(records['initial_training'], first)
        self.assertEqual(records['resume_training']['count'], 2)
        self.assertEqual(records['resume_training']['cumulative_date_range'], [20250115, 20250501])
        phase('test', 'last_test', [20250401, 20250430])
        phase('no-dates', 'last_test')
        records = phase('failed-test', 'last_test', [20250501, 20250531], 'failed')
        self.assertEqual(records['last_test']['git_head'], 'head-test')
        self.assertEqual(len(records), 3)

    def test_backfill_pending_and_repeated_run(self):
        from src.proj import PATH

        data = self.base / 'archive/0/20250101/best/state_dict.pt'
        data.parent.mkdir(parents=True)
        data.write_bytes(b'weights stay on the publisher')
        original, = publish_artifacts('model_training', 'old-run', [
            ArtifactSpec(self.config_file, category='model_training', experiment=MODEL)
        ], config=self.pub)
        self.assertEqual(original['status'], 'published')
        self.assertEqual(receive(self.rec)[0]['status'], 'received')
        (self.pub.shared_root / '.stfolder').rmdir()
        migration = load_patch_support('0.7.6', 'model_metadata')
        model = SimpleNamespace(full_name=MODEL, base=self.base, requires_fit=True)
        context = SimpleNamespace(backup_dir=self.root / 'backup')
        with patch.object(migration, 'source_models', side_effect=lambda _: [(model, self.base, '')]), \
                patch.object(migration, 'operation_history', return_value={}), \
                patch.object(Config, 'load', return_value=self.pub), \
                patch.object(PATH, 'lc_machine', self.root / 'machine'):
            migration.apply(context)
            self.assertTrue(migration.verify(context))
            records = read_properties(self.base / 'logs/property.jsonl')
            self.assertEqual(records['initial_training']['status'], 'estimated')
            self.assertIsNone(records['initial_training'].get('git_head'))
            ledger, = [(path, row) for path, row in migration.delivery_records(self.pub)
                       if row['manifest']['task'] == BACKFILL_TASK]
            self.assertEqual(migration.durable(ledger[0], ledger[1]), 'pending')
            with metadata_specs(self.pub, self.base, MODEL) as (specs, _, _):
                self.assertEqual(specs, [])
            self.config_file.write_text('model.module: changed after the frozen snapshot\n')
            (self.pub.shared_root / '.stfolder').mkdir()
            migration.apply(context)
            self.assertTrue(migration.verify(context))
        result, = receive(self.rec)
        batch = Path(result['target'])
        self.assertEqual(batch.parent, self.rec.results / 'external/models/server' / MODEL)
        self.assertEqual((batch / 'configs/model.yaml').read_text(), 'model.module: gru\n')
        self.assertTrue((batch / 'logs/property.jsonl').exists())
        self.assertFalse(list(batch.rglob('*.pt')))
        self.assertEqual(receive(self.rec), [])
