# description: Audit and recover a completed RL walk-forward result email
# content: 只读核验指定 RL 实验的训练与测试产物；通过后重新打包发送，不读取数据库、不训练、不修改原结果。
# autofix: true
# email: true
# mode: shell
# parameters:
#   run_id:
#     type: str
#     default: 5d17935c80694717a8df5fbd055791d4
#     desc: 需要核验的自动 RL 运行 ID
#   run_directory:
#     type: str
#     default: results/rl_experimental/walk_forward/gru_v1_low_turnover_v1/automatic/5d17935c80694717a8df5fbd055791d4
#     desc: 相对项目根目录的已有实验目录
#   dry_run:
#     type: bool
#     default: false
#     desc: 只读核验，不创建结果包和补发结果附件

from pathlib import Path
from uuid import uuid4

from src.proj import PATH, Proj
from src.proj.util.script import ScriptTool, AutoRunTask


@ScriptTool('autofix_recover_rl_results')
def main(run_id: str = '5d17935c80694717a8df5fbd055791d4',
         run_directory: str = 'results/rl_experimental/walk_forward/gru_v1_low_turnover_v1/automatic/5d17935c80694717a8df5fbd055791d4',
         dry_run: bool = False, **kwargs):
    from src.res.rl_experimental.bundle import bundle_attachments
    from src.res.rl_experimental.recover_results import audit_suite, recover_suite, read_json
    from src.res.rl_experimental.task_mail import TaskResultMail

    root = Path(PATH.main).resolve()
    run = (root/run_directory).resolve()
    allowed = (root/'results/rl_experimental').resolve()
    if allowed not in run.parents or run.name != run_id or not run.is_dir():
        raise ValueError(f'Invalid or missing original RL run: {run}')
    logs = list((root/'.local_resources').glob(f'*/runtime/scheduling/idle_worklist/{run_id}.log'))
    log = logs[0] if len(logs) == 1 else None
    print(f'[RL/recovery] Audit {run_id}; no training, no model loading, no database access.', flush=True)
    if dry_run:
        audit = audit_suite(run, project_root=root, log_path=log)
        print(audit)
        return
    destination = run/'recovery'/('20261028-audit-'+uuid4().hex[:12])
    # The surrounding ScriptTool has already created the active task.
    task = AutoRunTask._instances[-1]
    task.email_sender = TaskResultMail(destination/'mail')
    # Honor the original experiment's recipient rather than silently redirecting.
    try:
        task.kwargs['email_recipient'] = read_json(run/'config.json').get('recipient')
    except (OSError, ValueError):
        # Still produce a diagnostic audit when the source config is missing.
        # The normal project recipient receives this failure report.
        pass
    audit, metadata = recover_suite(run, destination, project_root=root, log_path=log)
    Proj.email_attachments.append(destination/'audit.json')
    print(f'[RL/recovery] complete={audit["complete"]}; verified training={audit["training_tasks_verified"]}, '
          f'test paths={audit["test_paths_verified"]}; audit: {destination}/audit.json', flush=True)
    if metadata is None:
        raise ValueError('Result audit failed; no success bundle or retraining. ' + '; '.join(audit['errors'][:5]))
    for attachment in bundle_attachments(metadata):
        Proj.email_attachments.append(attachment)
    print(f'[RL/recovery] Result bundle registered for task mail: {metadata}', flush=True)
    return {'original_run': run_id, 'tests_verified': True, 'audit': str(destination/'audit.json'),
            'bundle': str(metadata), 'mail_receipt': str(destination/'mail/delivery.json')}


if __name__ == '__main__':
    main()
