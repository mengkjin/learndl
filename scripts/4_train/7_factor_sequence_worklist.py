#!/usr/bin/env python3
# description: Prepare six factor sequence schedules and enqueue them in worklist
# mode: shell
"""Reuse a completed data audit and split plan without launching training."""
from pathlib import Path
import argparse
import json
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def prepare(suite):
    from src.proj import PATH, CALENDAR
    from src.data.preprocess.factor_sequence.bundle import load
    from src.data.preprocess.factor_sequence.schedule_training import schedule_config, variants
    suite = Path(suite).resolve()
    report = json.loads((suite / 'suite.json').read_text())
    dataset = Path(report['dataset'])
    manifest = load(dataset)  # streamed integrity checks, no tensor loading
    audit = json.loads((dataset / 'audit.json').read_text())
    if audit.get('status') != 'passed' or audit.get('version') != manifest['version']:
        raise ValueError('Dataset requires a matching passed audit')
    plan = json.loads((suite / 'walk_forward.json').read_text())
    folds = [{**f, 'model_date': int(CALENDAR.td(f['test'][0], -1))} for f in plan['folds']]
    if not folds:
        raise ValueError('Empty historical plan')
    if report.get('data_version') != manifest['version']:
        raise ValueError('Suite and dataset versions differ')
    # Freeze identities by source suite: rerunning preparation cannot create @2 roots.
    names, pending = [], []
    for model, variant in variants(False):
        name = f'fs_worklist_{suite.name}_{model}_17'
        config = schedule_config(dataset, name, model, folds, False, 17, variant)
        target = PATH.sched_shared / f'{name}.yaml'
        content = yaml.safe_dump(config, sort_keys=False)
        if target.exists() and target.read_text() != content:
            raise ValueError(f'Existing schedule differs: {target}')
        names.append(name)
        pending.append((target, content))
    worklist = PATH.sched_worklist
    original = worklist.read_text()
    config = yaml.safe_load(original)
    if config.get('resume') is not True or config.get('force') is not False:
        raise ValueError('Worklist must use resume: true, force: false to reuse model roots safely')
    for target, content in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with target.open('x') as stream:
                stream.write(content)
    config['fit'] = list(dict.fromkeys([*names, *config['fit']]))
    # Publish the queue only after all six schedules exist. Preserve other settings.
    if worklist.read_text() != original:
        raise RuntimeError('Worklist changed during preparation; rerun preparation')
    temporary = worklist.with_suffix('.factorseq.tmp')
    with temporary.open('x') as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    temporary.replace(worklist)
    print(f'Dataset: {dataset}\nTest range: {folds[0]["test"][0]} .. {folds[-1]["test"][-1]}')
    print('Queued ordinary schedules:\n' + '\n'.join(names))
    print('Training starts through the idle worklist worker; preparation itself does not train.')
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True,
                        help='Existing results/factor_sequence/<run_id> with a passed audit and walk_forward.json')
    args = parser.parse_args()
    prepare(args.suite)


if __name__ == '__main__':
    main()
