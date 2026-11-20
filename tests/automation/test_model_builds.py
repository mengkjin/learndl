from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.api.learndl_monitor.artifact_library import Library
from src.automation.artifact_exchange.core import ArtifactSpec, Config, atomic_json, publish_artifacts, receive
from src.automation.artifact_exchange.model_builds import archived_builds, load, INDEX, PENDING
from src.automation.artifact_exchange.model_identity import build_identity

MODEL = 'nn@gru@build-test'


class ModelBuildTest(unittest.TestCase):
    def setUp(self):
        work = TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.root = Path(work.name).resolve()
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
        self.assertNotEqual(first['name'], second['name'])
        summary.write_text('retrained')
        self.publish('retrained', ['results/summary.txt', props], second)
        self.assertEqual(receive(self.rec)[0]['status'], 'received')
        rows = dict((folder.name, index) for folder, index in archived_builds(self.rec))
        self.assertEqual(set(rows), {first['name'], second['name']})
        self.assertIn('legacy', rows[first['name']]['runs'])
        library = Library('model_preview', self.rec, self.root / 'monitor')
        with patch('src.api.learndl_monitor.artifact_library.source_models', return_value=[]):
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
        self.assertEqual(load(target / INDEX)['files']['results/summary.txt']['package'], first['id'])
        self.assertEqual(receive(self.rec)[0]['status'], 'received')
        self.assertEqual((target / 'results/summary.txt').read_text(), 'second')
