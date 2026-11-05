"""Host-side recovery admission; isolated autofix only submits this work."""
from __future__ import annotations

__all__ = [
    'original_inactive',
    'recover',
]
import os
import traceback


def original_inactive(report):
    """Missing/stale suite status alone never establishes process death."""
    import psutil
    if report.get('scheduler_read_error'):
        raise RuntimeError('Cannot read original scheduler ledger')
    rows = report['scheduler']
    if not rows:
        raise RuntimeError('Original task has no scheduler identity; inspect manually')
    for row in rows:
        check = row.get('process_check','unknown')
        if check == 'alive_matching_identity':
            return False
        if check.startswith('unknown') and row.get('phase') not in ('killed','cancelled','complete','error','failed','done','completed','success','terminated','timed_out'):
            raise RuntimeError('Original task liveness is unresolved')
    # Parent exit does not prove its pipeline subprocesses have exited.
    for proc in psutil.process_iter(['pid','uids','cmdline']):
        try:
            if proc.pid == os.getpid() or proc.uids().real != os.getuid():
                continue
            args = proc.cmdline()
            if any(arg == report['run_id'] or arg.startswith(report['run_id']+'_f') for arg in args):
                return False
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as exc:
            raise RuntimeError('Cannot exclude surviving original child process') from exc
    return True


def recover(run_id, resume_from, module):
    from src.proj import PATH
    from .config import runtime_dir
    from .factor_sequence_progress import collect
    from src.data.preprocess.factor_sequence.bundle import write_json
    from src.data.preprocess.factor_sequence.server_suite import send_result
    from src.proj.util.web.task_mail import TaskResultMail, retry_delivery
    diagnostic = PATH.result/'factor_sequence_diagnostics'/run_id/'host'
    diagnostic.mkdir(parents=True,exist_ok=True)
    report = collect(PATH.result/'factor_sequence',runtime_dir(),resume_from,isolated=False)
    write_json(diagnostic/'progress.json',report)
    if not original_inactive(report):
        return 75
    source = PATH.result/'factor_sequence'/resume_from
    # Send whatever the original attempt actually produced before trying recovery.
    # Old attempts do not contain isolated model logs; label the shared fold log honestly.
    mails=[]  # Legacy per-fold messages are never backfilled.
    summary=send_result(source)
    write_json(diagnostic/'mail_recovery.json',{'original_summary':summary,'training':mails})
    # Send progress regardless of whether the original suite can be resumed.
    mail=diagnostic/'mail'
    try:
        if (mail/'delivery.json').exists():
            delivered=retry_delivery(mail)
        else:
            delivered=TaskResultMail(mail)(f'Factor recovery {run_id}',
                f"Original: {resume_from}; {report['assessment']}. Schedule models resume through the normal training interface; legacy per-fold runs require explicit migration.\n"
                'Attached: scheduler/log evidence and original-result delivery receipts.',
                attachments=[diagnostic/'progress.json',diagnostic/'mail_recovery.json'])
        diagnostic_mail={'status':'sent' if delivered else 'failed'}
    except Exception:
        diagnostic_mail={'status':'failed','error':traceback.format_exc()}
    write_json(diagnostic/'mail_status.json',diagnostic_mail)
    if report['suite'].get('status')=='passed':
        # A complete suite needs delivery only, never another training run.
        if diagnostic_mail['status']=='failed' or summary['status']=='failed' or any(m['status']=='failed' for m in mails):
            raise RuntimeError(f'Results complete but delivery failed; inspect {diagnostic}')
        return 0
    return module.main(run_id=run_id,resume_from=resume_from,
        research=report['suite'].get('research',False),ablations=report['suite'].get('ablations',False),
        data_config=str(source/'data_config.json'))
