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
    Config, _target, atomic_json, experiment_members, fsync_dir, real_tree, received_member, safe_path,
)
from src.automation.artifact_exchange.model_logs import STAGES
from .experiments import Instances
from .services import ROOT, STATE


__all__ = ['Library', 'open_folder']


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
        for path in sorted((self.config.results / category).glob('*/*/source.json'), reverse=True):
            try:
                manifest = read_json(path)
                if (manifest['schema_version'] == 2 and manifest['category'] == category
                        and _target(manifest, self.config) == path.parent
                        and (self.config.state / 'received' / (manifest['id'] + '.json')).is_file()):
                    yield path.parent, manifest
            except (OSError, ValueError, KeyError, TypeError):
                continue

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
            for folder, manifest in self._batches('model_training'):
                host, model = manifest['host'], manifest['experiment']
                experiments = experiment_members(manifest)
                for item in manifest['files']:
                    relative = item['path']
                    if relative.startswith(('run-log/', 'snapshot/', 'results/factor_residual/')):
                        continue
                    try:
                        path = existing(folder, received_member(manifest, relative, experiments, folder=folder).as_posix())
                    except (OSError, ValueError):
                        continue
                    if path:
                        add(path, path.name, type=experiments.get(relative, 'file'), host=host, model=model,
                            run=manifest['run_id'], stamp=manifest['created_at'], batch=folder.name,
                            relative=path.relative_to(folder).as_posix(), lock=None)

            for identity in self.config.results.glob('model_run_logs/*/*/model.json'):
                try:
                    meta = read_json(identity)
                    for phase in STAGES:
                        if path := existing(identity.parent, f'{phase}.md'):
                            add(path, path.name, type='process_log', lock=None, **meta)
                except (OSError, ValueError, KeyError, TypeError):
                    continue

            from src.proj import PATH
            from src.res.model.util import ModelPath

            from src.api.pkgs.dashboard import model_tensorboard

            studies = list(PATH.optuna.glob('*/*.sqlite3'))
            for base in PATH.model.glob('*/*'):
                try:
                    real_tree(base)
                    if not base.is_dir() or base.name.startswith('.'):
                        continue
                    try:
                        if base.parent == PATH.model_archive:
                            name, _, stamp = base.name.rpartition('.')
                            if len(stamp) != 14 or not stamp.isdigit():
                                continue
                            model = f'{ModelPath(name).full_name} · 归档 {stamp}'
                        else:
                            model = ModelPath(base).full_name
                    except (AssertionError, ValueError):
                        continue
                    common = dict(host=self.config.source_host, model=model, lock=base)
                    for relative in ('run_logs', 'results'):
                        for path in (base / relative).rglob('*'):
                            try:
                                real_tree(path)
                                if relative == 'results' and path.relative_to(base / relative).parts[0] in (
                                        'tensorboard', 'optuna', 'factor_residual'):
                                    continue
                                if not path.is_file() or any(part.startswith('.') for part in path.relative_to(base).parts):
                                    continue
                                phase_log = path in (base / 'run_logs/train.md', base / 'run_logs/test.md')
                                html = path.parent == base / 'run_logs/html' and path.suffix == '.html'
                                add(path, path.name, type='process_log' if phase_log else 'file',
                                    run=path.stem if html else '', relative=path.relative_to(base).as_posix(), **common)
                            except (OSError, ValueError):
                                continue
                    events = real_tree(model_tensorboard(base))
                    if events.is_dir() and any(events.rglob('*tfevents*')):
                        add(events, 'TensorBoard', type='tensorboard', **common)
                    for study in (base / 'results/optuna').glob('*.sqlite3'):
                        add(real_tree(study), study.name, type='optuna', **common)
                    for study in studies:
                        if not (base / 'results/optuna' / study.name).exists() and study.stem.rpartition('@')[0] == model:
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

    def listing(self, run=''):
        rows = []
        for row in self.rows().values():
            if run:
                if self.kind == 'mail_archive':
                    if run not in row['metadata'].get('model_runs', []):
                        continue
                elif row.get('run') != run and (
                        row['type'] != 'process_log' or row.get('trashed') or not self.chapters(row, 0, run)['pages']):
                    continue
            public = {k: v for k, v in row.items() if k not in ('path', 'files', 'lock', 'original')}
            files = row.get('files', [])
            public['files'] = [name for name, _ in files]
            if row.get('type') == 'mail' and not row.get('trashed'):
                public['bytes'] = sum(path.stat().st_size for _, path in files)
            elif row['path'].is_file():
                public['bytes'] = row['path'].stat().st_size
            else:
                public['bytes'] = 0
            rows.append(public)
        return sorted(rows, key=lambda r: (r.get('stamp', ''), r['label']), reverse=True)

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
        if path.stat().st_size > 32 * 1024**2:
            raise ValueError('工作簿超过预览大小限制，请下载查看')
        with ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist()) > 128 * 1024**2:
                raise ValueError('工作簿解压后超过预览大小限制，请下载查看')
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
        instance = candidates[0]
        return self.instances.act({'action': 'stop', 'kind': instance['kind'],
                                   'pid': instance['pid'], 'created': instance['created']})


def open_folder(path):
    import subprocess
    import sys

    path = real_tree(path)
    folder = path if path.is_dir() else path.parent
    command = ['open', str(folder)] if sys.platform == 'darwin' else ['xdg-open', str(folder)]
    subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)
    return str(folder)
