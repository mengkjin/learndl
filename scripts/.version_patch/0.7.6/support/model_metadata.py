"""One-time model provenance reconstruction and publisher backfill."""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import suppress
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from src.automation.artifact_exchange.core import (
    Config, atomic_json, digest, publish_artifacts, real_tree, send_prepared,
)
from src.automation.artifact_exchange.model_metadata import (
    BACKFILL_TASK, metadata_specs, source_models,
)
from src.automation.artifact_exchange.model_properties import (
    PROPERTY_FILE, coverage, read_properties, write_properties,
)


def read(path):
    return json.loads(real_tree(path).read_text())


def timestamp(value):
    return datetime.fromisoformat(value).astimezone(timezone.utc).isoformat()


def history(root, bases):
    runs = defaultdict(list)
    for path in root.glob('*/run.json'):
        with suppress(OSError, ValueError, KeyError, TypeError):
            run = read(path)
            if run.get('model_path') in bases and run.get('started_at'):
                run['started_at'] = timestamp(run['started_at'])
                if run.get('finished_at'):
                    run['finished_at'] = timestamp(run['finished_at'])
                runs[run['model_path']].append(run)
    return runs


def artifact_times(paths):
    estimates = []
    for path in paths:
        with suppress(OSError, ValueError):
            stat = real_tree(path).stat()
            birth = getattr(stat, 'st_birthtime', None)
            created = datetime.fromtimestamp(birth or stat.st_mtime, timezone.utc).isoformat()
            modified = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
            estimates.append((created, modified, 'file_created' if birth else 'file_modified'))
    return sorted(estimates)


def operation_history(model, base, archived):
    from src.proj import PATH
    from src.proj.log.logfile import LogEntry

    log = PATH.logs.joinpath('model', 'operation', model.full_name).with_suffix('.log')
    if not log.is_file():
        return {}
    cutoff = timestamp(datetime.strptime(archived, '%Y%m%d%H%M%S').isoformat()) if archived else ''
    # Archived and live models share the operation log. Use it only when the
    # recorded operation falls within the original directory's lifetime.
    born = getattr(base.stat(), 'st_birthtime', None)
    if archived and not born:
        return {}
    lower = datetime.fromtimestamp(born, timezone.utc).isoformat() if born else ''
    stages = defaultdict(list)
    open_ops = {}
    for entry in LogEntry.from_lines(real_tree(log).read_text().splitlines(), from_latest=False):
        phase, _, suffix = entry.title.partition(' >> ')
        action, _, resume = suffix.partition(' >> ')
        value = timestamp(entry.timestamp.isoformat())
        if phase not in ('fit_model', 'test_model') or value < lower or (cutoff and value > cutoff):
            continue
        if action == 'start':
            open_ops[phase] = (value, resume == 'is_resuming=True')
        elif action == 'end' and phase in open_ops:
            start, is_resume = open_ops.pop(phase)
            stages[phase].append((start, value, is_resume))
    return stages


def historical_properties(model, base, archived, runs):
    weights = [p for p in (base / 'archive').glob('*/*/*/*')
               if p.name in ('state_dict.pt', 'boost_dict.pt')]
    preds = list((base / 'snapshot/pred_recorder/preds').glob('*.feather'))
    train_times = artifact_times(weights) if model.requires_fit else []
    test_times = artifact_times(preds or (p for p in (base / 'results').glob('*.html')
                                         if 'testing' in p.name))
    nodes = defaultdict(list)
    with suppress(OSError, ValueError, KeyError, TypeError):
        for node in read(base / 'snapshot/model_manifest.json').values():
            if node.get('run_id') and node.get('data_start') and node.get('data_end'):
                nodes[node['run_id']].append([node['data_start'], node['data_end']])

    if archived:
        cutoff = timestamp(datetime.strptime(archived, '%Y%m%d%H%M%S').isoformat())
        born = getattr(base.stat(), 'st_birthtime', None)
        lower = datetime.fromtimestamp(born, timezone.utc).isoformat() if born else None
        runs = [r for r in runs if lower and lower <= r['started_at'] <= cutoff]
    operations = operation_history(model, base, archived)
    records = {}
    for phase, kind, times in [('fit', 'initial_training', train_times), ('test', 'last_test', test_times)]:
        matches = []
        for start, end, resume in operations.get(phase + '_model', []):
            matches.append({'kind': kind, 'started_at': start, 'finished_at': end,
                            'status': 'success', 'source': 'operation_log', 'resume': resume,
                            'date_range': None})
        for run in sorted(runs, key=lambda r: r['started_at']):
            if phase not in run.get('stages', ()) or run.get('status') != 'success':
                continue
            span = coverage(*nodes.get(run['run_id'], [])) if phase == 'fit' else None
            # A recorded successful invocation may contain a no-op stage.
            operation = next((item for item in matches
                              if run['started_at'] <= item['started_at'] <= (run['finished_at'] or '')), None)
            if operation is None and not span:
                continue
            facts = {'run_id': run['run_id'], 'git_head': (run.get('git') or {}).get('head'),
                     'date_range': span, 'resume': bool(run.get('resume'))}
            if operation is not None:
                operation.update(facts)
            else:
                matches.append({'kind': kind, 'started_at': run['started_at'], 'finished_at': run['finished_at'],
                                'source': 'training_history', 'status': 'success', **facts})
        # Legacy start/end entries also exist for no-op stages. Require a
        # matching node or a file written during the stage to count it.
        matches = [item for item in matches if item['date_range'] or any(
            item['started_at'] <= modified <= (
                datetime.fromisoformat(item['finished_at']) + timedelta(seconds=1)).isoformat()
            for _, modified, _ in times
        )]
        matches.sort(key=lambda item: item['started_at'])
        if matches:
            chosen = matches[0] if phase == 'fit' else matches[-1]
            records[kind] = chosen
            if phase == 'fit':
                resumed = [item for item in matches[1:] if item['resume']]
                if resumed:
                    records['resume_training'] = resumed[-1] | {
                        'kind': 'resume_training', 'count': len(resumed),
                        'first_started_at': resumed[0]['started_at'],
                        'cumulative_date_range': coverage(*(item['date_range'] for item in resumed)),
                    }
                full = [item for item in matches[1:] if not item['resume']]
                if full:
                    chosen['latest_full_training'] = full[-1]
        elif times:
            created, _, source = times[0] if phase == 'fit' else max(times, key=lambda time: time[1])
            records[kind] = {'kind': kind, 'started_at': created,
                             'finished_at': max(time[1] for time in times), 'status': 'estimated',
                             'source': source, 'finish_source': 'file_modified', 'date_range': None}
    return records


def journal_path(config):
    return config.state / 'migrations/model-metadata-0.7.6.json'


def delivery_records(config):
    for path in sorted((config.state / 'outbox').glob('*/delivery.json')):
        with suppress(OSError, ValueError, TypeError):
            yield path, read(path)


def preflight(config, root):
    models = list(source_models(root))
    candidates = defaultdict(list)
    for model, base, _ in models:
        candidates[model.full_name].append(str(base))
    published = set()
    if config.enabled and config.role == 'publisher':
        for _, record in delivery_records(config):
            manifest = record.get('manifest', {})
            if (record.get('status') == 'published' and manifest.get('category') == 'model_training'
                    and manifest.get('task') != BACKFILL_TASK and manifest.get('host') == config.source_host
                    and manifest.get('experiment')):
                published.add(manifest['experiment'])
    targets = {name: {'paths': candidates[name]} for name in sorted(published)}
    return {'models': [str(base) for _, base, _ in models], 'targets': targets}


def durable(path, record):
    if record.get('status') == 'published':
        return 'published'
    package = path.parent / 'package.zip'
    if record.get('status') == 'prepared' and package.is_file() and digest(real_tree(package)) == record['sha256']:
        return 'pending'
    return 'failed'


def apply(ctx):
    from src.proj import PATH
    from src.res.model.util.lifecycle import ModelLifecycle

    config = Config.load()
    path = journal_path(config)
    report = read(path) if path.exists() else preflight(config, PATH.model)
    atomic_json(path, report)
    models = {str(base): (model, base, archived) for model, base, archived in source_models(PATH.model)}
    runs = history(PATH.lc_machine / 'training_history', {str(m.base) for m, _, _ in models.values()})
    for key in report['models']:
        if key not in models:
            raise ValueError(f'Model left the preflight inventory: {key}')
        model, base, archived = models[key]
        properties = base / PROPERTY_FILE
        if not properties.exists():
            with ModelLifecycle(base).lock():
                if not properties.exists():
                    write_properties(properties, historical_properties(model, base, archived, runs[str(model.base)]))

    existing = {record['manifest']['experiment']: (ledger, record) for ledger, record in delivery_records(config)
                if record.get('manifest', {}).get('task') == BACKFILL_TASK
                and record['manifest'].get('host') == config.source_host}
    for name, item in report['targets'].items():
        try:
            if len(item['paths']) != 1:
                item['status'] = 'ambiguous' if item['paths'] else 'missing'
                continue
            if item['paths'][0] not in models:
                item['status'] = 'missing'
                continue
            _, base, _ = models[item['paths'][0]]
            if name in existing:
                ledger, frozen = existing[name]
            else:
                with ModelLifecycle(base).lock(), metadata_specs(config, base, name, full=True) as (specs, _, _):
                    if not specs:
                        item['status'] = 'missing'
                        continue
                    records = publish_artifacts(BACKFILL_TASK, '0.7.6-model-metadata', [
                        replace(spec, experiment=name) for spec in specs
                    ], config=config, prepare_only=True)
                    frozen = records[0]
                    if frozen.get('status') not in ('prepared', 'published'):
                        raise ValueError(frozen.get('error') or frozen.get('reason') or 'Preparation failed')
                    ledger = config.state / 'outbox' / frozen['outbox_key'] / 'delivery.json'
            outcome = send_prepared([dict(frozen)], config)[0]
            item.update(status=durable(ledger, read(ledger)), id=frozen['id'], ledger=str(ledger))
            if 'error' in outcome:
                item['error'] = outcome['error']
            else:
                item.pop('error', None)
        except (OSError, ValueError, KeyError, IndexError) as exc:
            item.update(status='failed', error=str(exc))
        finally:
            atomic_json(path, report)
    counts = Counter(item['status'] for item in report['targets'].values())
    report['counts'] = {'source_models': len(report['models']), 'candidates': len(report['targets']),
                        **{status: counts[status] for status in ('missing', 'ambiguous', 'published', 'pending', 'failed')}}
    atomic_json(path, report)
    atomic_json(ctx.backup_dir / 'model-metadata.json', report)


def verify(ctx):
    config = Config.load()
    report = read(journal_path(config))
    for base in report['models']:
        path = real_tree(Path(base) / PROPERTY_FILE)
        if not path.is_file():
            return False
        read_properties(path)
    for item in report['targets'].values():
        if item['status'] in ('missing', 'ambiguous'):
            continue
        if item['status'] not in ('published', 'pending'):
            return False
        ledger = Path(item['ledger'])
        if durable(ledger, read(ledger)) not in ('published', 'pending'):
            return False
    return True


if __name__ == '__main__':
    from src.proj import Logger, PATH

    report = preflight(Config.load(), PATH.model)
    Logger.stdout(json.dumps(report, ensure_ascii=False, indent=2), vb_level=0)
