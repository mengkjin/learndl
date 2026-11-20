"""Local catalog and file operations for synchronized mail and model outputs."""
from __future__ import annotations

from contextlib import ExitStack
from hashlib import sha256
import json
import mimetypes
from pathlib import Path
import shutil
from uuid import uuid4

import portalocker

from src.automation.artifact_exchange.core import (
    Config, archived_batches, atomic_json, experiment_members, fsync_dir, real_tree, received_member, safe_path,
)
from src.automation.artifact_exchange.model_builds import archived_builds
from src.shared.model_provenance import LOG_PHASES, PROPERTY_FILE, read_properties
from src.automation.artifact_exchange.model_metadata import BACKFILL_TASK, INVENTORY
from .experiments import Instances
from .services import ROOT, STATE


__all__ = ['Library', 'open_folder', 'open_local']


def read_json(path):
    return json.loads(real_tree(path).read_text())


def existing(root, relative):
    path = real_tree(root / safe_path(relative))
    if path.is_file():
        return path
    return None


class Library:
    def __init__(self, kind, config=None, state=STATE):
        self.kind = kind
        self.config = config or Config.load()
        self.state = real_tree(state / kind)
        self.state.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.instances = Instances(state) if kind == 'model_preview' else None

    def _batches(self, category):
        for folder, manifest in archived_batches(self.config, category):
            if (self.config.state / 'received' / (manifest['id'] + '.json')).is_file():
                yield folder, manifest

    def rows(self):
        catalog = {}
        marks_path = self.state / 'archived.json'
        marks = set(read_json(marks_path)) if marks_path.exists() else set()

        def add(path, label, **data):
            key = sha256(str(path).encode()).hexdigest()
            catalog[key] = dict(id=key, path=path, label=label, archived=key in marks, **data)

        if self.kind == 'mail_archive':
            for folder, manifest in self._batches('mail_archive'):
                try:
                    meta = read_json(folder / 'mail/mail.json')
                    allowed = {item['path'] for item in manifest['files']}
                    files = []
                    for name, relative in [('正文', 'body.txt'), ('原始邮件', 'message.eml'),
                                           *((a['name'], a['file']) for a in meta['attachments'])]:
                        item = 'mail/' + relative
                        if item in allowed and (path := existing(folder, item)):
                            files.append((name, path))
                    add(folder, meta['subject'], metadata=meta, files=files, type='mail',
                        stamp=meta['created_at'], host=manifest['host'])
                except (OSError, ValueError, KeyError, TypeError):
                    continue
        else:
            for folder, index in archived_builds(self.config):
                common = dict(host=index['host'], model=index['model'], model_root=folder,
                              display_name=folder.parent.name,
                              physically_archived=len(folder.relative_to(self.config.results / 'external/models').parts) == 4,
                              lock=None, build=folder.name,
                              historical=index['build'].get('estimated', False))
                package_runs = {run['package']: run_id for run_id, run in index['runs'].items()}
                for item in index['files'].values():
                    name = item['path']
                    if name.startswith('snapshot/'):
                        continue
                    try:
                        path = existing(folder, name)
                    except (OSError, ValueError):
                        continue
                    if path:
                        kind = ('process_log' if name in ('logs/train.md', 'logs/test.md') else
                                'file' if item['kind'] == 'reports' else item['kind'])
                        run = item.get('run') or package_runs.get(item['package'], '')
                        if name.startswith('html/') and path.suffix == '.html':
                            run = path.stem
                        add(path, path.name, type=kind, relative=name, stamp=item['stamp'],
                            run='' if kind == 'process_log' else run,
                            readonly=name.startswith(('configs/', 'logs/')), **common)

            for folder, manifest in self._batches('model_training'):
                host = manifest.get('host') or 'unknown'
                model = manifest.get('experiment') or manifest['task']
                experiments = experiment_members(manifest)
                common = dict(host=host, model=model, model_root=folder.parent, lock=None,
                              stamp=manifest.get('created_at', ''), batch=folder.name,
                              run=manifest['run_id'], historical=manifest.get('task') == BACKFILL_TASK)
                for item in manifest['files']:
                    relative = item['path']
                    if relative.startswith(('run-log/', 'snapshot/', 'results/factor_residual/')):
                        continue
                    try:
                        path = existing(folder, received_member(manifest, relative, experiments, folder=folder).as_posix())
                    except (OSError, ValueError):
                        continue
                    if path:
                        metadata = relative.startswith(('configs/', 'logs/'))
                        add(path, path.name, type=('metadata_index' if relative == INVENTORY
                                                   else experiments.get(relative, 'file')),
                            relative=path.relative_to(folder).as_posix(), sha=item['sha256'],
                            readonly=metadata or relative == INVENTORY, **common)

            for identity in self.config.results.glob('external/models/*/*/run_logs/model.json'):
                try:
                    meta = read_json(identity)
                    for phase in LOG_PHASES:
                        if path := existing(identity.parent, f'{phase}.md'):
                            add(path, path.name, type='process_log', lock=None, model_root=identity.parent.parent,
                                relative=f'run_logs/{phase}.md', host=meta['host'], model=meta['model'])
                except (OSError, ValueError, KeyError, TypeError):
                    continue

            from src.proj import PATH
            from src.res.model.util.core.model_path import source_models
            from src.api.pkgs.dashboard import model_tensorboard

            studies = list(PATH.optuna.glob('*/*.sqlite3'))
            for model, base, archived in source_models(PATH.model):
                try:
                    build = read_properties(base / PROPERTY_FILE).get('build', {}).get('name', '')
                    common = dict(host=self.config.source_host, model=model.full_name, lock=base,
                                  model_root=base, archive_stamp=archived, build=build or archived,
                                  physically_archived=bool(archived))
                    for relative in ('configs', 'logs', 'run_logs', 'results'):
                        for path in (base / relative).rglob('*'):
                            try:
                                real_tree(path)
                                if relative == 'results' and path.relative_to(base / relative).parts[0] in (
                                        'tensorboard', 'optuna', 'factor_residual'):
                                    continue
                                if not path.is_file() or any(part.startswith('.') for part in path.relative_to(base).parts):
                                    continue
                                phase_log = path.parent.name in ('run_logs', 'logs') and path.name in ('train.md', 'test.md')
                                html = path.parent == base / 'run_logs/html' and path.suffix == '.html'
                                add(path, path.name, type='process_log' if phase_log else 'file',
                                    run=path.stem if html else '', relative=path.relative_to(base).as_posix(),
                                    readonly=relative in ('configs', 'logs'), **common)
                            except (OSError, ValueError):
                                continue
                    events = real_tree(model_tensorboard(base))
                    if events.is_dir() and any(events.rglob('*tfevents*')):
                        add(events, 'TensorBoard', type='tensorboard', **common)
                    for study in (base / 'results/optuna').glob('*.sqlite3'):
                        add(real_tree(study), study.name, type='optuna', **common)
                    for study in studies:
                        if not (base / 'results/optuna' / study.name).exists() and study.stem.rpartition('@')[0] == model.full_name and not archived:
                            add(real_tree(study), study.name, type='optuna', **common)
                except (OSError, ValueError):
                    continue

        trash = self.state / 'trash'
        for meta_path in trash.glob('*/item.json'):
            try:
                record = read_json(meta_path)
                stored = real_tree(meta_path.parent / 'payload')
                if stored.exists():
                    key = meta_path.parent.name
                    catalog[key] = dict(record['row'], id=key, path=stored, trashed=True, original=Path(record['original']))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return catalog

    def listing(self, run='', *, catalog=None):
        rows = []
        for row in (self.rows() if catalog is None else catalog).values():
            if row['type'] == 'metadata_index':
                continue
            if run:
                if self.kind == 'mail_archive':
                    if run not in row['metadata'].get('model_runs', []):
                        continue
                elif row.get('run') != run and (
                        row['type'] != 'process_log' or row.get('trashed') or not self.chapters(row, 0, run)['pages']):
                    continue
            public = {k: v for k, v in row.items()
                      if k not in ('path', 'files', 'lock', 'original', 'model_root', 'sha', 'runs')}
            if row.get('model'):
                public['model_id'] = self.model_id(row)
            files = row.get('files', [])
            public['files'] = [name for name, _ in files]
            if row['type'] == 'mail':
                public['mail_parts'] = [{'body.txt': 'body', 'message.eml': 'original'}.get(
                    path.name, '') if path.parent == row['path'] / 'mail' else '' for _, path in files]
            if row.get('type') == 'mail' and not row.get('trashed'):
                public['bytes'] = sum(path.stat().st_size for _, path in files)
            elif row['path'].is_file():
                public['bytes'] = row['path'].stat().st_size
            else:
                public['bytes'] = 0
            rows.append(public)
        return sorted(rows, key=lambda r: (r.get('stamp', ''), r['label']), reverse=True)

    @staticmethod
    def model_id(row):
        return sha256(json.dumps([row.get('host'), row.get('model')])
                      .encode()).hexdigest()

    def model_folder(self, key, build=None):
        for row in self.rows().values():
            if (row.get('model_root') and not row.get('trashed') and self.model_id(row) == key
                    and (build is None or row.get('build', '') == build)):
                return real_tree(row['model_root'])
        raise ValueError('Model folder unavailable')

    def model_listing(self, run=''):
        from .model_catalog import describe_models

        catalog = self.rows()
        files = self.listing(run, catalog=catalog)
        models = describe_models(catalog.values(), self.model_id)
        from .model_management import inventory
        for entry in inventory(self).values():
            key = entry['model_id']
            if key not in models:
                models[key] = dict(id=key, name=entry['name'], host=entry['host'], builds=[])
            if not any(build['build'] == entry['build'] for build in models[key]['builds']):
                models[key]['builds'].append(dict(id=key, name=entry['name'], host=entry['host'],
                    build=entry['build'], source_name=entry['source_name'], module_type=entry['module_type'],
                    module=entry['module'], stage=entry['stage'], short_test=entry['short']=='yes',
                    physically_archived=entry['archived'], experiments=[], versions=[], properties={}))
        views = self._model_views()
        for key, model in models.items():
            for build in model['builds']:
                status = views.get(key, {}).get(build['build'], 'active')
                build['hidden'] = status != 'active'
        return {'files': files, 'models': models}

    def _model_views(self):
        path = self.state / 'model_views.json'
        return read_json(path) if path.exists() else {}

    def available_row(self, key):
        return self.rows()[key]

    def chapters(self, row, page, run=''):
        if row['type'] != 'process_log' or row.get('trashed'):
            raise ValueError('Select a model process log')
        path = real_tree(row['path'])
        offsets = []
        with path.open('rb') as stream:
            while line := stream.readline():
                if line.startswith(b'## '):
                    offsets.append((stream.tell() - len(line), line.decode(errors='replace').strip()[3:]))
            end = stream.tell()
            if not offsets:
                return {'chapters': [], 'pages': 0}
            order = sorted(range(len(offsets)), key=lambda i: offsets[i][1], reverse=True)
            if run:
                order = [i for i in order if offsets[i][1].endswith(' · ' + run)]
            result = []
            for index in order[page * 10:(page + 1) * 10]:
                offset, title = offsets[index]
                length = (offsets[index + 1][0] if index + 1 < len(offsets) else end) - offset
                stream.seek(offset)
                result.append({'title': title, 'content': stream.read(min(length, 256 * 1024)).decode(errors='replace'),
                               'truncated': length > 256 * 1024})
        return {'chapters': result, 'pages': (len(order) + 9) // 10}

    def file(self, row, index):
        if row.get('trashed'):
            raise ValueError('Restore the item to open it')
        if row['type'] == 'mail':
            if index < 0:
                raise ValueError('Invalid attachment index')
            name, path = row['files'][index]
        else:
            path = row['path']
            name = path.name
        path = real_tree(path)
        if not path.is_file():
            raise ValueError('This item is not a file')
        mime = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
        if path.suffix.lower() in ('.yaml', '.yml', '.log', '.jsonl', '.toml', '.eml'):
            mime = 'text/plain'
        if mime not in ('text/html', 'text/plain', 'text/markdown', 'application/json', 'application/pdf'):
            mime = 'application/octet-stream'
        return path, name, mime

    def spreadsheet(self, row, index, sheet=''):
        """Return a bounded, read-only worksheet preview."""
        from openpyxl import load_workbook
        from zipfile import ZipFile

        path, name, _ = self.file(row, index)
        if Path(name).suffix.lower() != '.xlsx':
            raise ValueError('请选择 XLSX 文件')
        fallback = '，请打开文件查看' if self.kind == 'model_preview' else '，请下载查看'
        if path.stat().st_size > 32 * 1024**2:
            raise ValueError('工作簿超过预览大小限制' + fallback)
        with ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist()) > 128 * 1024**2:
                raise ValueError('工作簿解压后超过预览大小限制' + fallback)
        book = load_workbook(path, read_only=True, data_only=True, keep_links=False)
        try:
            names = [worksheet.title for worksheet in book.worksheets]
            if not names:
                raise ValueError('工作簿没有可预览的工作表')
            sheet = sheet or names[0]
            if sheet not in names:
                raise ValueError('工作表不存在')
            worksheet = book[sheet]
            worksheet.reset_dimensions()
            rows = [[str(value)[:1024] if value is not None else '' for value in values]
                    for values in worksheet.iter_rows(max_row=40, max_col=20, values_only=True)]
            while rows and not any(rows[-1]):
                rows.pop()
            width = max((i + 1 for values in rows for i, value in enumerate(values) if value), default=0)
            return {'sheets': names, 'sheet': sheet, 'rows': [values[:width] for values in rows]}
        finally:
            book.close()

    def manage(self, action, keys):
        with portalocker.Lock(str(self.state / 'operations.lock'), timeout=2):
            catalog = self.rows()
            marks_path = self.state / 'archived.json'
            marks = set(read_json(marks_path)) if marks_path.exists() else set()
            results = []
            for key in dict.fromkeys(keys):
                try:
                    row = catalog[key]
                    if row.get('readonly'):
                        raise ValueError('Model configuration and provenance are read-only')
                    if action in ('archive', 'unarchive'):
                        if row.get('trashed'):
                            raise ValueError('Restore the item before archiving it')
                        (marks.add if action == 'archive' else marks.discard)(key)
                        atomic_json(marks_path, sorted(marks))
                    else:
                        self._move(action, row, marks, marks_path)
                    results.append({'id': key, 'ok': True})
                except (OSError, ValueError, KeyError, portalocker.exceptions.LockException) as exc:
                    results.append({'id': key, 'error': str(exc)})
            return results

    def _move(self, action, row, marks, marks_path):
        path = real_tree(row['path'])
        if row.get('trashed'):
            if action == 'purge':
                shutil.rmtree(path.parent)
                return
            if action != 'restore':
                raise ValueError('Choose restore or permanently delete')
            original = real_tree(row['original'])
            with self._write_guards(row):
                if original.exists():
                    raise ValueError('A newer file already occupies the original path')
                original.parent.mkdir(parents=True, exist_ok=True)
                path.rename(original)
                shutil.rmtree(path.parent)
                if row.get('archived'):
                    marks.add(sha256(str(original).encode()).hexdigest())
                    atomic_json(marks_path, sorted(marks))
            return
        if action != 'delete':
            raise ValueError('Move the item to Trash first')
        with self._write_guards(row):
            folder = self.state / 'trash' / uuid4().hex
            folder.mkdir(parents=True)
            saved = {k: str(v) if isinstance(v, Path) else v for k, v in row.items()
                     if k not in ('path', 'files')}
            atomic_json(folder / 'item.json', {'original': str(path), 'row': saved})
            try:
                path.rename(folder / 'payload')
            except OSError:
                shutil.rmtree(folder)
                raise
            fsync_dir(path.parent)
            fsync_dir(folder)
            marks.discard(row['id'])
            atomic_json(marks_path, sorted(marks))

    def _write_guards(self, row):
        from src.automation.artifact_exchange.core import locked

        with ExitStack() as setup:
            if base := row.get('lock'):
                setup.enter_context(portalocker.Lock(str(Path(base) / '.lifecycle.lock'), timeout=0))
            if self.config.role == 'receiver':
                setup.enter_context(locked(self.config, 'receive'))
            if self.instances and row['type'] in ('tensorboard', 'optuna'):
                setup.enter_context(portalocker.Lock(str(self.instances.state / 'operations.lock'), timeout=1))
                candidate = str(row.get('original', row['path']))
                if any(candidate in (item.get('source'), item.get('archive_source'))
                       for item in self.instances.discover()):
                    raise ValueError('Stop the experiment instance before moving its source')
            return setup.pop_all()

    def experiment(self, row, action):
        if self.instances is None or row.get('trashed') or row['type'] not in ('tensorboard', 'optuna'):
            raise ValueError('Choose an experiment source')
        if action == 'start':
            return self.instances.act({'action': 'start', 'kind': row['type'], 'source': str(row['path'])})
        if action != 'stop':
            raise ValueError('Unknown experiment action')
        candidates = [r for r in self.instances.discover()
                      if str(row['path']) in (r.get('source'), r.get('archive_source'))]
        if not candidates:
            raise ValueError('This source has no running instance')
        instance = max(candidates, key=lambda item: item['created'])
        return self.instances.act({'action': 'stop', 'kind': instance['kind'],
                                   'pid': instance['pid'], 'created': instance['created']})


def open_local(path):
    import subprocess
    import sys

    path = real_tree(path)
    command = ['open' if sys.platform == 'darwin' else 'xdg-open', str(path)]
    subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)
    return str(path)


def open_folder(path):
    path = real_tree(path)
    return open_local(path if path.is_dir() else path.parent)
