"""Independent worklist documents, ordered by Git committer time."""
from __future__ import annotations
import subprocess
from collections import Counter
from src.proj import PATH
from src.res.model.util.training_history import file_revision


def git_time(path):
    result = subprocess.run(['git', '--no-optional-locks', '-C', str(PATH.main), 'log', '-1',
                             '--format=%ct', '--', str(path.resolve())],
                            capture_output=True, text=True, timeout=10, check=True)
    return int(result.stdout.strip()) if result.stdout.strip() else None


def load(*, automatic=False):
    import yaml
    from .worklist_state import WorklistState
    docs, errors, sources = [], [], {}
    for path in PATH.sched_worklists.glob('*.yaml'):
        if path.name.startswith('.'):
            continue
        try:
            revision = file_revision(path)
            timestamp = git_time(path)
            doc = yaml.safe_load(revision['content'])
            entries = WorklistState.entries(doc)
            WorklistState.rerun_mark(doc)
            docs.append(dict(path=str(path), revision=revision, time=timestamp, doc=doc, entries=entries))
        except (ValueError, OSError, subprocess.SubprocessError, yaml.YAMLError) as exc:
            errors.append(f'{path}: {exc}')
    docs.sort(key=lambda d: (d['time'] if d['time'] is not None else float('inf'), d['path']))
    counts = Counter(name for d in docs for name in d['entries'])
    duplicates = {name for name, count in counts.items() if count > 1}
    for name in sorted(duplicates):
        errors.append(f'Duplicate schedule {name}: ' + ', '.join(d['path'] for d in docs if name in d['entries']))
    main = next((d for d in docs if d['path'] == str(PATH.sched_worklists / 'main.yaml')), None)
    merged = dict(main['doc']) if main else {}
    merged['fit'] = []
    for d in docs:
        if automatic and d['time'] is None:
            errors.append(f'{d["path"]}: waiting for Git modification history')
        for name, options in d['entries'].items():
            if name in duplicates or (automatic and d['time'] is None):
                continue
            merged['fit'].append(dict(name=name, **options))
            sources[name] = d
    merged['_sources'] = sources
    merged['_fallback'] = main
    merged['_blocked'] = duplicates | {name for d in docs if automatic and d['time'] is None for name in d['entries']}
    return merged, docs, errors


def revision_for(worklist, name):
    if name in worklist['_blocked']:
        raise ValueError(f'Worklist entry is blocked: {name}')
    source = worklist['_sources'].get(name) or worklist['_fallback']
    return source['revision'] if source else {'path': '', 'content': '', 'sha256': '', 'git_commit': None}
