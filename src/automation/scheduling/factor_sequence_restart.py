"""Delete only recorded legacy per-fold model roots, then run six ordinary schedules."""
from pathlib import Path
import json
import re
import shutil
import sys

LEGACY_RUNS = ('20261031_factor_sequence_verify_01', '20261031_factor_sequence_verify_02',
               '20261101_factor_sequence_recover_03')
MODELS = ('factorseq_mlp','factorseq_gru','factorseq_m1','factorseq_m2','factorseq_mask','factorseq_gate')
MODEL_PATTERN = '|'.join(MODELS)
LEGACY_NAME = re.compile(r'fs_('+'|'.join(LEGACY_RUNS)+r')_f\d{3}_\d{8}_('
                         +MODEL_PATTERN+r')_(?:full|use_performance|use_market|modulation)_\d+')


def plan_cleanup(model_root, results_root):
    """Use both immutable run naming and recorded model identity; never a broad fs_* glob delete."""
    model_root, results_root = Path(model_root), Path(results_root)
    if model_root.is_symlink() or not model_root.is_dir():
        raise ValueError('Expected a real existing NN model directory')
    records={}
    for run_id in LEGACY_RUNS:
        for path in sorted(results_root.glob(run_id+'_f*/training/runs.json')):
            for row in json.loads(path.read_text()):
                name=row.get('name','')
                match=LEGACY_NAME.fullmatch(name)
                if not match or row.get('model')!=match[2]:
                    raise ValueError(f'Unexpected legacy model record: {path}: {name}')
                records[(row['model'],name)]=str(path)
    targets=[]
    for path in sorted(model_root.iterdir()):
        parts=path.name.split('@')
        if len(parts) not in (2,3) or parts[0] not in MODELS:
            continue
        match=LEGACY_NAME.fullmatch(parts[1])
        if not match:
            continue
        if len(parts)==3 and (not parts[2].isdigit() or int(parts[2])<2):
            raise ValueError(f'Unexpected legacy directory index: {path}')
        if parts[0]!=match[2] or (parts[0],parts[1]) not in records:
            raise ValueError(f'Legacy directory lacks matching training evidence: {path}')
        if path.is_symlink() or not path.is_dir() or path.resolve().parent!=model_root.resolve():
            raise ValueError(f'Unsafe legacy model path: {path}')
        if sys.platform=='linux':
            from .autofix_sandbox import check_mounts
            check_mounts(model_root.resolve(),Path('/proc/self/mountinfo').read_text())
        # Internal links are also rejected rather than following a mount/link outside this model.
        for sub in path.rglob('*'):
            if sub.is_symlink() or sub.is_mount():
                raise ValueError(f'Linked/mounted legacy content requires inspection: {sub}')
        stat=path.stat()
        targets.append(dict(path=str(path),run_id=match[1],model=parts[0],
                            evidence=records[(parts[0],parts[1])],device=stat.st_dev,inode=stat.st_ino))
    return targets


def delete_planned(targets, journal):
    """Write the exact list before irreversible deletion; keep progress on partial failure."""
    from src.data.preprocess.factor_sequence.bundle import write_json
    report={'status':'running','backup':False,'targets':targets,'deleted':[]}
    write_json(journal,report)
    try:
        for target in targets:
            path=Path(target['path'])
            stat=path.stat()
            if path.is_symlink() or (stat.st_dev,stat.st_ino)!=(target['device'],target['inode']):
                raise ValueError(f'Deletion target changed: {path}')
            shutil.rmtree(path)
            if path.exists():
                raise RuntimeError(f'Deletion did not finish: {path}')
            report['deleted'].append(str(path))
            write_json(journal,report)
        report['status']='passed'
    except BaseException:
        import traceback
        report.update(status='failed',error=traceback.format_exc())
        raise
    finally:
        write_json(journal,report)
    return report


def check_managed_scopes(run_id, runtime):
    """A dead parent is insufficient: its supervised session/cgroup must also be empty."""
    import sqlite3
    from .runtime import members
    database=Path(runtime)/'runs.sqlite'
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True,timeout=3) as conn:
        rows=conn.execute("SELECT data FROM runs WHERE json_extract(data,'$.request_id')=? "
                          "AND json_extract(data,'$.task_kind')='factor_sequence'",(run_id,)).fetchall()
    if not rows:
        raise RuntimeError(f'Missing managed scope identity for {run_id}')
    for raw, in rows:
        row=json.loads(raw)
        # idle_worklist starts each task with start_new_session=True, but its
        # service cgroup is reused by later invocations. Historical checks must
        # not mistake this cleanup worker for a surviving original child.
        scope=dict(row)
        if (scope.get('cgroup') or '').endswith('/learndl-idle-worklist.service'):
            scope.pop('cgroup')
        remaining=members(scope) if scope.get('pid') else []
        if remaining:
            raise RuntimeError(f'Original managed process group still has members: {run_id}')


def execute(request_id, module):
    from src.proj import PATH
    from src.proj.util.script import FitLockNN
    from .config import runtime_dir
    from .factor_sequence_progress import collect
    from .factor_sequence_recovery import original_inactive
    from src.data.preprocess.factor_sequence.bundle import write_json
    from src.proj.util.web.task_mail import TaskResultMail
    destination=PATH.result/'factor_sequence_diagnostics'/request_id
    destination.mkdir(parents=True,exist_ok=False)
    journal=destination/'cleanup.json'
    try:
        with FitLockNN.guard():
            targets=plan_cleanup(PATH.model_nn,PATH.result/'factor_sequence')
            evidence={}
            for run_id in LEGACY_RUNS:
                if (PATH.result/'factor_sequence'/run_id).exists() or any(t['run_id']==run_id for t in targets):
                    evidence[run_id]=collect(PATH.result/'factor_sequence',runtime_dir(),run_id,isolated=False)
                    write_json(destination/'progress.json',evidence)
                    if not original_inactive(evidence[run_id]):
                        raise RuntimeError(f'Original task is still alive: {run_id}; no folders deleted')
                    check_managed_scopes(run_id,runtime_dir())
                    if any(row.get('phase') in ('queued','starting') for row in evidence[run_id]['scheduler']):
                        raise RuntimeError(f'Original task can still start: {run_id}; no folders deleted')
            delete_planned(targets,journal)
            if plan_cleanup(PATH.model_nn,PATH.result/'factor_sequence'):
                raise RuntimeError('Legacy model directories remain; training will not start')
        # Same one-shot worker, after exclusive NN cleanup lock is released.
        # The suite itself sequentially invokes the normal schedule script for the six models.
    except BaseException:
        import traceback
        failed=json.loads(journal.read_text()) if journal.exists() else {'backup':False,'deleted':[]}
        failed.update(status='failed',error=traceback.format_exc())
        write_json(journal,failed)
        raise
    finally:
        # A single cleanup receipt, never one mail per old model/fold.
        attachments=[p for p in (journal,destination/'progress.json') if p.exists()]
        try:
            ok=TaskResultMail(destination/'mail')(f'Factor sequence cleanup {request_id}',
                'Legacy per-fold model directories deleted without backups only when checks passed. '
                'See cleanup.json for the exact outcome. New training uses the ordinary schedule interface.',
                attachments=attachments)
            write_json(destination/'mail_status.json',{'accepted':bool(ok)})
        except Exception as exc:
            write_json(destination/'mail_status.json',{'accepted':False,'error':str(exc)})
    return module.main(run_id=request_id,test_start=20170101,research=False,ablations=False)
