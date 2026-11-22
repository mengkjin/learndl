"""Independent worklist documents, ordered by Git committer time."""
from __future__ import annotations
import json
import subprocess
from collections import Counter
from src.proj import PATH
from src.proj.env.schedules import find_schedule_path, schedule_stem
from src.res.model.util.training_history import file_revision

__all__ = ['git_time', 'load', 'revision_for', 'reference_for']


def git_time(path):
    result = subprocess.run(['git', '--no-optional-locks', '-C', str(PATH.main), 'log', '-1',
                             '--format=%ct', '--', str(path.resolve())],
                            capture_output=True, text=True, timeout=10, check=True)
    return int(result.stdout.strip()) if result.stdout.strip() else None


def load(*, automatic=False):
    import yaml
    from src.res.model.workflow.worklist_state import WorklistState
    docs, errors, sources = [], [], {}
    for path in PATH.sched_worklists.glob('*.yaml'):
        if path.name.startswith('.'):
            continue
        try:
            revision = file_revision(path)
            timestamp = git_time(path)
            doc = yaml.safe_load(revision['content'])
            entries = WorklistState.entries(doc)
            for reference in entries:
                schedule_stem(reference)
            WorklistState.rerun_mark(doc)
            docs.append(dict(path=str(path), revision=revision, time=timestamp, doc=doc, entries=entries))
        except (ValueError, OSError, subprocess.SubprocessError, yaml.YAMLError) as exc:
            errors.append(f'{path}: {exc}')
    docs.sort(key=lambda d: (d['time'] if d['time'] is not None else float('inf'), d['path']))
    counts = Counter(schedule_stem(ref) for d in docs for ref in d['entries'])
    duplicates = {name for name, count in counts.items() if count > 1}
    for name in sorted(duplicates):
        occurrences = [f"{doc['path']} ({ref})" for doc in docs
                       for ref in doc['entries'] if schedule_stem(ref) == name]
        errors.append(f'Duplicate schedule {name}: ' + ', '.join(occurrences))
    main = next((d for d in docs if d['path'] == str(PATH.sched_worklists / 'main.yaml')), None)
    merged = dict(main['doc']) if main else {}
    merged['fit'] = []
    for d in docs:
        if automatic and d['time'] is None:
            errors.append(f'{d["path"]}: waiting for Git modification history')
        for reference, options in d['entries'].items():
            name = schedule_stem(reference)
            if name in duplicates or (automatic and d['time'] is None):
                continue
            merged['fit'].append(dict(name=name, **options))
            sources[name] = d | {'reference': reference, 'options': options}
    merged['_sources'] = sources
    merged['_fallback'] = main
    merged['_blocked'] = duplicates | {schedule_stem(ref) for d in docs if automatic and d['time'] is None for ref in d['entries']}
    return merged, docs, errors


def revision_for(worklist, name):
    if name in worklist['_blocked']:
        raise ValueError(f'Worklist entry is blocked: {name}')
    source = worklist['_sources'].get(name) or worklist['_fallback']
    return source['revision'] if source else {'path': '', 'content': '', 'sha256': '', 'git_commit': None}


def reference_for(worklist, name):
    """Recover the selected reference from the stable stem used in the ledger."""
    from .worklist_state import WorklistState

    stem = schedule_stem(name)
    source = worklist['_sources'].get(stem)
    request = WorklistState(stem).request_path
    requested = json.loads(request.read_text()).get('schedule_name') if request.is_file() else None
    reference = source['reference'] if source else requested or name
    for explicit in (name, requested):
        if explicit and '/' in explicit and explicit != reference:
            selected = find_schedule_path(explicit)
            if selected is None or selected != find_schedule_path(reference):
                raise ValueError(f'Schedule reference {explicit} conflicts with {reference}')
    return reference
