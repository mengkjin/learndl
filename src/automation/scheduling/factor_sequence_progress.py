"""Bounded, read-only evidence collection; never reconciles or creates run ledgers."""
from pathlib import Path
import json
import os
import re
import sqlite3
import time


def read_json(path):
    try:
        if path.stat().st_size > 4_000_000:
            return {'read_error': 'JSON exceeds diagnostic size limit'}
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return {'read_error': str(exc)}


def log_tail(path, now):
    try:
        stat = path.stat()
        with path.open('rb') as stream:
            stream.seek(max(0, stat.st_size - 24000))
            tail = stream.read(24000).decode('utf-8', errors='replace')
        return {'path': str(path), 'bytes': stat.st_size, 'modified_at': stat.st_mtime,
                'age_seconds': max(0, now-stat.st_mtime), 'tail': tail}
    except OSError as exc:
        return {'path': str(path), 'read_error': str(exc)}


def collect(results_root, runtime, run_id, isolated=None):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('Invalid run ID')
    now = time.time()
    folder = Path(results_root)/run_id
    runtime = Path(runtime)
    report = {'run_id': run_id, 'collected_at': now, 'result_directory': str(folder),
              'suite': read_json(folder/'suite.json'),
              'mail': read_json(folder/'mail_status.json'), 'scheduler': [], 'folds': [], 'logs': []}
    # Do not instantiate RunStore: its constructor writes schema even for status queries.
    database = runtime/'runs.sqlite'
    try:
        with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True, timeout=3) as conn:
            rows = conn.execute("SELECT data FROM runs WHERE json_extract(data,'$.request_id')=? "
                                "AND json_extract(data,'$.task_kind')='factor_sequence'", (run_id,)).fetchall()
        evidence = {}
        try:
            with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True, timeout=3) as conn:
                for raw in rows:
                    identity = json.loads(raw[0]).get('id')
                    value = conn.execute('SELECT data FROM run_evidence WHERE run_id=?', (identity,)).fetchone()
                    if value:
                        evidence[identity] = json.loads(value[0])
        except sqlite3.Error as exc:
            report['evidence_read_error'] = str(exc)
        fields = ('id','phase','pid','process_created','started_at','finished_at','returncode',
                  'blocked_by','log_path','progress_timeout_seconds')
        for row in rows:
            doc = json.loads(row[0])
            report['scheduler'].append({**{key: doc[key] for key in fields if key in doc},
                                        'evidence': evidence.get(doc.get('id'), {})})
    except (OSError, sqlite3.Error, ValueError) as exc:
        report['scheduler_read_error'] = str(exc)
    isolated = bool(os.environ.get('LEARNDL_AUTOFIX_ID')) if isolated is None else isolated
    for row in report['scheduler']:
        row['process_check'] = 'unknown: autofix PID namespace cannot establish host liveness'
        if not isolated and row.get('pid') and row.get('process_created'):
            import psutil
            try:
                p = psutil.Process(row['pid'])
                row['process_check'] = ('alive_matching_identity' if
                    abs(p.create_time()-row['process_created']) < .01 and p.is_running()
                    else 'identity_mismatch')
                if row['process_check'] == 'alive_matching_identity':
                    row['process_status'] = p.status()
            except psutil.NoSuchProcess:
                row['process_check'] = 'absent'
            except psutil.Error as exc:
                row['process_check'] = 'unknown: '+str(exc)
    # Derive paths from the selected run, not paths supplied by an untrusted JSON document.
    for child in sorted(Path(results_root).glob(run_id+'_f*'))[:200]:
        runs = read_json(child/'training/runs.json')
        entries = runs if isinstance(runs, list) else []
        report['folds'].append({'directory': str(child), 'smoke': read_json(child/'smoke.json'),
            'models': [{k: r[k] for k in ('model','name','status','seconds','error','model_path') if k in r}
                       for r in entries], 'runs_read_error': runs.get('read_error') if isinstance(runs,dict) else None})
    logs = sorted(folder.glob('*.log'), key=lambda p: p.stat().st_mtime, reverse=True)[:4]
    for row in report['scheduler']:
        identity = row.get('id', '')
        if re.fullmatch(r'[A-Za-z0-9_-]+', identity):
            logs.append(runtime/'idle_worklist'/f'{identity}.log')
    report['logs'] = [log_tail(p, now) for p in logs]
    suite = report['suite']
    report['current_stage'] = next((s.get('name') for s in reversed(suite.get('stages', []))
                                    if s.get('status') in ('running','failed')), None)
    report['completed_stages'] = [s.get('name') for s in suite.get('stages', []) if s.get('status') == 'passed']
    status = suite.get('status')
    phases = [r.get('phase') for r in report['scheduler']]
    if status in ('passed','failed'):
        report['assessment'] = 'suite_'+status
    elif any(p in ('error','failed','terminated','timed_out') for p in phases):
        report['assessment'] = 'scheduler_ended_without_final_suite'
    elif status == 'running':
        report['assessment'] = 'unfinished; scheduler/file evidence does not prove current progress'
    else:
        report['assessment'] = 'suite_missing_or_unreadable; inspect queue and scheduler evidence'
    report['notes'] = [
        'A running suite is not proof of a live process. Old logs alone do not prove a hang.',
        'Task Monitor may omit build/audit and unregistered wrapper processes.',
        'Legacy runs only sent suite mail; newer runs also store per-model training_mail receipts.',
        'No training restart, state reconciliation, label rebuild or result overwrite was performed.']
    return report
