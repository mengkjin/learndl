from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.api.learndl_monitor.artifact_library import Library
from src.automation.artifact_exchange.core import ArtifactSpec, Config, atomic_json, publish_artifacts, receive
from src.automation.artifact_exchange.model_builds import archived_builds, load, INDEX, PENDING
from src.res.model.util.model_identity import build_identity

MODEL = 'nn@gru@build-test'


class ModelBuildTest(unittest.TestCase):
    def setUp(self):
        work = TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.root = Path(work.name).resolve()
        self.enterContext(patch('src.automation.artifact_exchange.core.ROOT', self.root))
        share = self.root / 'share'
        (share / '.stfolder').mkdir(parents=True)
        self.pub = Config(True, 'publisher', share, self.root / 'pub', self.root / 'pub-results', source_host='server')
        self.rec = Config(True, 'receiver', share, self.root / 'rec', self.root / 'rec-results', source_host='client')
        self.base = self.root / 'models/nn/gru@build-test'
        (self.base / 'results').mkdir(parents=True)
        (self.base / 'logs').mkdir()

    def publish(self, run, paths, build=None, *, adopt=False):
        specs = [ArtifactSpec(self.base / path, category='model_training', experiment=MODEL,
                              destination=str(Path(path).parent)) for path in paths]
        result, = publish_artifacts('model_training', run, specs, config=self.pub, build=build, adopt_legacy=adopt)
        return result

    def transfer(self, record):
        return self.pub.inbox / record['route'] / record['filename']

    def test_two_restarts_and_two_retests_keep_one_build_including_replayed_updates(self):
        from src.proj import PATH

        with patch.object(PATH, 'lc_machine', self.root / 'machine'):
            build = build_identity(self.base, fresh=True)
        summary = self.base / 'results/summary.txt'
        summary.write_text('initial')
        unchanged = self.base / 'results/initial.txt'
        unchanged.write_text('only in the first update')
        first = self.publish('initial', ['results/summary.txt', 'results/initial.txt'], build)
        delayed = self.root / first['filename']
        self.transfer(first).rename(delayed)
        for run, phase in [('resume-fit-1', 'train'), ('resume-fit-2', 'train'),
                           ('resume-test-1', 'test'), ('resume-test-2', 'test')]:
            summary.write_text(run)
            log = self.base / 'logs' / (phase + '.md')
            with log.open('a') as stream:
                stream.write(f'## {phase} · {run}\n\n- completed\n')
            record = self.publish(run, ['results/summary.txt', 'results/initial.txt', f'logs/{phase}.md'], build)
            self.assertEqual(record['status'], 'published', record)
            self.assertNotIn('results/initial.txt', {f['path'] for f in record['manifest']['files']})
        self.assertEqual(len(receive(self.rec)), 4)
        self.transfer(first).parent.mkdir(parents=True)
        delayed.rename(self.transfer(first))
        packet = self.transfer(first).read_bytes()
        self.assertEqual(receive(self.rec)[0]['status'], 'received')
        self.transfer(first).parent.mkdir(parents=True)
        self.transfer(first).write_bytes(packet)
        self.assertEqual(receive(self.rec)[0]['status'], 'received')

        (target, index), = archived_builds(self.rec)
        self.assertEqual(target.name, build['name'])
        self.assertEqual(len(index['runs']), 5)
        self.assertEqual((target / 'results/summary.txt').read_text(), 'resume-test-2')
        self.assertEqual((target / 'results/initial.txt').read_text(), 'only in the first update')
        self.assertEqual((target / 'logs/test.md').read_text().count('## '), 2)

        library = Library('model_preview', self.rec, self.root / 'monitor')
        with patch('src.res.model.util.core.model_path.source_models', return_value=[]):
            initial = library.listing('initial')
            self.assertEqual([(row['relative'], row['run']) for row in initial], [('results/initial.txt', 'initial')])
            resumed = {row['relative'] for row in library.listing('resume-test-1')}
            self.assertEqual(resumed, {'logs/test.md'})

    def test_migrated_trash_restores_after_build_adoption_and_replay(self):
        from src.automation.version_patch import load_patch_support
        from src.proj import PATH

        self.enterContext(patch('src.res.model.util.core.model_path.source_models', return_value=[]))
        library = Library('model_preview', self.rec, self.root / '.local_resources/learndl_monitor')
        source = self.base / 'results/report.txt'
        source.write_text('legacy')
        record = self.publish('old', ['results/report.txt'])
        packet = self.transfer(record).read_bytes()
        old = Path(receive(self.rec)[0]['target'])
        migration = load_patch_support('0.7.7', 'model_builds')
        old_file = old / 'results/report.txt'
        old_file.write_text('modified')
        with self.assertRaisesRegex(ValueError, 'differs from its manifest'):
            migration.migrate_receiver(self.rec, self.root, self.root / 'backup')
        self.assertTrue(old_file.is_file())
        old_file.write_text('legacy')
        original, = library.listing()
        self.assertTrue(library.manage('delete', [original['id']])[0]['ok'])
        migration.migrate_receiver(self.rec, self.root, self.root / 'backup')
        atomic_json(self.rec.state / 'model_builds_ready.json', {})
        with patch.object(PATH, 'lc_machine', self.root / 'machine'):
            build = build_identity(self.base)
        self.publish('adopt', ['logs/property.jsonl'], build, adopt=True)
        target = Path(receive(self.rec)[0]['target'])
        self.transfer(record).parent.mkdir(parents=True, exist_ok=True)
        self.transfer(record).write_bytes(packet)
        self.assertEqual(receive(self.rec)[0]['target'], str(target))
        trashed, = [row for row in library.listing() if row.get('trashed')]
        self.assertTrue(library.manage('restore', [trashed['id']])[0]['ok'])
        restored, = [row for row in library.listing() if row['label'] == 'report.txt']
        self.assertEqual((target / restored['relative']).read_text(), 'legacy')
        self.assertEqual(restored['run'], 'old')

    def test_old_packets_attach_to_history_and_rebuild_stays_separate(self):
        from src.proj import PATH

        atomic_json(self.rec.state / 'model_builds_ready.json', {})
        summary = self.base / 'results/summary.txt'
        summary.write_text('old')
        self.publish('legacy', ['results/summary.txt'])
        self.assertEqual(receive(self.rec)[0]['status'], 'received')
        with patch.object(PATH, 'lc_machine', self.root / 'machine'):
            first = build_identity(self.base)
            props = 'logs/property.jsonl'
            summary.write_text('new')
            outcome = self.publish('adopt', ['results/summary.txt', props], first, adopt=True)
            self.assertEqual(outcome['status'], 'published', outcome)
            self.assertEqual(receive(self.rec)[0]['status'], 'received')
            self.assertEqual(build_identity(self.base), first)
            original = self.base / props
            original.unlink()
            second = build_identity(self.base)
        self.assertNotEqual(first['fingerprint'], second['fingerprint'])
        prefix, _, suffix = first['name'].rpartition('.')
        self.assertEqual(second['name'], prefix + '.' + str(int(suffix) + 1))
        summary.write_text('retrained')
        self.publish('retrained', ['results/summary.txt', props], second)
        self.assertEqual(receive(self.rec)[0]['status'], 'received')
        rows = dict((folder.name, index) for folder, index in archived_builds(self.rec))
        self.assertEqual(set(rows), {first['name'], second['name']})
        self.assertIn('legacy', rows[first['name']]['runs'])
        library = Library('model_preview', self.rec, self.root / 'monitor')
        with patch('src.res.model.util.core.model_path.source_models', return_value=[]):
            model, = library.model_listing()['models'].values()
            self.assertEqual(len(model['builds']), 2)
            self.assertEqual(library.model_folder(model['id'], first['name']).name, first['name'])

    def test_interrupted_file_replacement_retries_before_older_packet(self):
        from src.proj import PATH

        with patch.object(PATH, 'lc_machine', self.root / 'machine'):
            build = build_identity(self.base)
        summary = self.base / 'results/summary.txt'
        summary.write_text('first')
        first = self.publish('first', ['results/summary.txt'], build)
        self.assertEqual(receive(self.rec)[0]['status'], 'received')
        summary.write_text('second')
        second = self.publish('second', ['results/summary.txt'], build)
        target = self.rec.results / 'external/models/server' / MODEL / build['name']
        with patch('src.automation.artifact_exchange.model_builds.replace_file', side_effect=OSError('interrupted')):
            self.assertEqual(receive(self.rec)[0]['status'], 'failed')
        self.assertEqual(load(target / PENDING), second['id'])
        committed = load(target / INDEX)
        assert committed is not None
        self.assertEqual(committed['files']['results/summary.txt']['package'], first['id'])
        self.assertEqual(receive(self.rec)[0]['status'], 'received')
        self.assertEqual((target / 'results/summary.txt').read_text(), 'second')

    def test_original_results_and_status_log_survive_explicit_reset(self):
        from src.proj import PATH
        from src.res.model.util.core import ModelPath

        short_root = self.root / 'models/st'
        archive = self.root / 'models/archive'
        with patch.object(PATH, 'lc_machine', self.root / 'machine'), \
                patch.object(PATH, 'model_st', short_root), patch.object(PATH, 'model_archive', archive), \
                patch.object(ModelPath, 'log_operation'):
            model = ModelPath('st@' + MODEL)
            model.mkdir(exist_ok=True)
            before = build_identity(model.base)
            model('logs', 'train.md').write_text('original training log')
            model.rslt('summary.txt').write_text('original report')
            model.clear_model_path()
            model.mkdir(exist_ok=True)
            after = build_identity(model.base, reset=True)
            preserved, = archive.iterdir()
            self.assertEqual(build_identity(preserved), before)
            self.assertNotEqual(before['fingerprint'], after['fingerprint'])
            self.assertFalse(model('logs', 'train.md').exists())
            self.assertEqual((preserved / 'logs/train.md').read_text(), 'original training log')
            self.assertEqual((preserved / 'results/summary.txt').read_text(), 'original report')
            self.assertIsNone(ModelPath.resume_from_archive(model.full_name))
