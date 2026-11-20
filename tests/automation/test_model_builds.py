from dataclasses import replace
from pathlib import Path
import sqlite3
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

    def test_model_preview_moves_only_selected_build_and_routes_new_deliveries(self):
        from src.api.learndl_monitor.model_management import inventory, manage
        from src.proj import PATH

        self.enterContext(patch.object(PATH, 'lc_machine', self.root / 'machine'))
        self.enterContext(patch('src.res.model.util.core.model_path.source_models', return_value=[]))
        original = build_identity(self.base, fresh=True)
        report = self.base / 'results/report.txt'
        report.write_text('initial')
        events = self.base / 'events'
        events.mkdir()
        (events / 'events.out.tfevents.test').write_bytes(b'events')
        db = self.base / 'study.sqlite3'
        with sqlite3.connect(db) as connection:
            connection.execute('CREATE TABLE sample (x integer)')
        publish_artifacts('model_training', 'original', [
            ArtifactSpec(path, kind=kind, category='model_training', experiment=MODEL)
            for path, kind in ((report, 'reports'), (events, 'tensorboard'), (db, 'optuna'))
        ], config=self.pub, build=original)
        self.assertEqual(receive(self.rec)[0]['status'], 'received')

        library = Library('model_preview', self.rec, self.root / 'monitor')
        self.enterContext(patch('src.api.learndl_monitor.experiments.Instances.discover', return_value=[]))

        def execute(action, build, value=None):
            entry = next(row for row in inventory(library).values() if row['build'] == build['name'])
            item = dict(id=entry['id'], value=value)
            planned, = manage(library, action, [item])
            self.assertTrue(planned.get('ok'), planned)
            applied, = manage(library, action, [item | {'target': planned['target']}], apply=True)
            self.assertTrue(applied.get('ok'), applied)
            return Path(applied['target'])

        entry, = inventory(library).values()
        atomic_json(library.state / 'model_views.json', {entry['model_id']: {entry['build']: 'archived_trash'}})
        self.assertTrue(next(iter(inventory(library).values()))['hidden'])
        execute('archive', original)
        archived = execute('rename', original, 'renamed')
        self.assertEqual(archived.relative_to(self.rec.results).parts,
                         ('external', 'models', 'archive', 'server', 'nn@gru@renamed', original['name']))
        report.write_text('update')
        updated = self.publish('update', ['results/report.txt'], original)
        self.assertEqual(receive(self.rec)[0]['target'], str(archived))
        self.assertEqual((archived / 'results/report.txt').read_text(), 'update')
        receipt = load(self.rec.state / 'received' / (updated['manifest']['id'] + '.json'), {})
        self.assertEqual(receipt['target'], str(archived))
        self.assertTrue(next(iter(inventory(library).values()))['hidden'])

        self.base.rename(self.root / 'original')
        report.parent.mkdir(parents=True)
        report.write_text('new build')
        newer = build_identity(self.base, fresh=True)
        self.publish('newer', ['results/report.txt'], newer)
        result, = receive(self.rec)
        self.assertEqual(Path(result['target']).parent.name, MODEL)
        self.assertNotEqual(Path(result['target']).parent, archived.parent)
        self.assertEqual({(row['name'], row['hidden']) for row in inventory(library).values()},
                         {('nn@gru@renamed', True), (MODEL, False)})
        execute('show', original)
        restored = execute('restore_archive', original)
        numbered = execute('renumber', original, '2')
        self.assertFalse(restored.exists())
        self.assertEqual(numbered.parent.name, 'nn@gru@renamed@2')
        self.assertEqual((numbered / 'results/report.txt').read_text(), 'update')
        self.assertEqual(report.read_text(), 'new build')

        from src.api.learndl_monitor.experiments import synced_sources

        for rows in synced_sources(self.rec.results).values():
            source, = rows
            self.assertTrue(Path(source['path']).is_relative_to(numbered))
        other = replace(self.pub, source_host='another-server')
        publish_artifacts('model_training', 'other-host', [
            ArtifactSpec(report, category='model_training', experiment=MODEL)
        ], config=other, build=original)
        received, = receive(self.rec)
        self.assertEqual(Path(received['target']).parent.parent.name, 'another-server')
        self.assertEqual((numbered / 'results/report.txt').read_text(), 'update')

    def test_manage_local_empty_directory_and_restore_with_number_conflict(self):
        from src.api.learndl_monitor.model_management import inventory, manage
        from src.proj import PATH

        base = self.root / 'models'
        archive = base / 'archive'
        archive.mkdir()
        self.enterContext(patch.object(PATH, 'model', base))
        self.enterContext(patch.object(PATH, 'model_nn', base / 'nn'))
        self.enterContext(patch.object(PATH, 'model_archive', archive))
        self.enterContext(patch('src.res.model.util.core.model_path.ModelPath.log_file'))
        self.enterContext(patch('src.api.learndl_monitor.experiments.Instances.discover', return_value=[]))
        library = Library('model_preview', self.rec, self.root / 'monitor')

        def plan(operation, entry, value=None):
            result, = manage(library, operation, [dict(id=entry['id'], value=value)])
            self.assertTrue(result.get('ok'), result)
            return result

        def apply(operation, planned, value=None):
            result, = manage(library, operation, [planned | {'value': value}], apply=True)
            self.assertTrue(result.get('ok'), result)
            return Path(result['target'])

        entry, = inventory(library).values()
        self.assertEqual(library.model_listing()['files'], [])
        moved = apply('archive', plan('archive', entry))
        self.assertEqual(moved.parent, archive)
        self.base.mkdir()
        archived = next(row for row in inventory(library).values() if row['archived'])
        restore = plan('restore_archive', archived)
        self.assertTrue(restore['name'].endswith('@2'))
        restored = apply('restore_archive', restore)
        row = next(row for row in inventory(library).values() if row['path'] == str(restored))
        conflict, = manage(library, 'renumber', [{'id': row['id'], 'value': '1'}])
        self.assertIn('destination already exists', conflict['error'])
        renamed = apply('rename', plan('rename', row, 'another'), 'another')
        self.assertTrue(renamed.is_dir())
        self.assertTrue(self.base.is_dir())

    def test_archived_legacy_build_adoption_keeps_location_and_hidden_state(self):
        from src.api.learndl_monitor.model_management import inventory, manage
        from src.proj import PATH

        self.enterContext(patch.object(PATH, 'lc_machine', self.root / 'machine'))
        self.enterContext(patch('src.res.model.util.core.model_path.source_models', return_value=[]))
        self.enterContext(patch('src.api.learndl_monitor.experiments.Instances.discover', return_value=[]))
        atomic_json(self.rec.state / 'model_builds_ready.json', {})
        (self.base / 'results/report.txt').write_text('legacy')
        self.publish('legacy', ['results/report.txt'])
        self.assertEqual(receive(self.rec)[0]['status'], 'received')

        library = Library('model_preview', self.rec, self.root / '.local_resources/learndl_monitor')
        entry, = inventory(library).values()
        selection = {'id': entry['id']}
        manage(library, 'hide', [selection | {'target': entry['path']}], apply=True)
        planned, = manage(library, 'archive', [selection])
        with patch('src.api.learndl_monitor.model_management.update_state', side_effect=OSError('interrupted')):
            outcome, = manage(library, 'archive', [planned], apply=True)
            self.assertEqual(outcome['error'], 'interrupted')
        self.assertEqual(manage(library, 'recover', [], apply=True), [{'ok': True}])
        archived = Path(planned['target'])
        self.assertTrue(archived.is_dir())

        formal = build_identity(self.base)
        self.publish('adopt', ['logs/property.jsonl'], formal, adopt=True)
        outcome, = receive(self.rec)
        self.assertEqual(outcome['status'], 'received', outcome)
        target = archived.with_name(formal['name'])
        self.assertEqual(outcome['target'], str(target))
        self.assertEqual((target / 'results/report.txt').read_text(), 'legacy')
        self.assertTrue(next(iter(inventory(library).values()))['hidden'])

    def test_history_undo_retains_updates_and_restores_visibility(self):
        from src.api.learndl_monitor.model_management import inventory, manage, operation_history, undo_latest
        from src.proj import PATH
        self.enterContext(patch.object(PATH, 'lc_machine', self.root / 'machine'))
        self.enterContext(patch('src.res.model.util.core.model_path.source_models', return_value=[]))
        self.enterContext(patch('src.api.learndl_monitor.experiments.Instances.discover', return_value=[]))
        build = build_identity(self.base, fresh=True)
        report = self.base / 'results/report.txt'
        report.write_text('before')
        self.publish('before', ['results/report.txt'], build)
        receive(self.rec)
        library = Library('model_preview', self.rec, self.root / 'monitor')
        original, = inventory(library).values()
        for action, value in [('rename', 'new-name@2'), ('archive', None), ('hide', None)]:
            row, = inventory(library).values()
            item = dict(id=row['id'], value=value)
            planned, = manage(library, action, [item])
            self.assertTrue(planned.get('ok'), planned)
            result, = manage(library, action, [item | {'target': planned['target']}], apply=True,
                             entry='cli' if action == 'archive' else 'web')
            self.assertTrue(result.get('ok'), result)
        report.write_text('updated')
        self.publish('updated', ['results/report.txt'], build)
        receive(self.rec)
        self.assertEqual(len(operation_history(library)), 3)
        for _ in range(3):
            self.assertTrue(undo_latest(library)['ok'])
            self.assertTrue(undo_latest(library, apply=True)['ok'])
        row, = inventory(library).values()
        self.assertEqual(row['path'], original['path'])
        self.assertFalse(row['hidden'])
        self.assertEqual((Path(row['path']) / 'results/report.txt').read_text(), 'updated')
        self.assertTrue(all(batch['undone'] for batch in operation_history(library)))

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
