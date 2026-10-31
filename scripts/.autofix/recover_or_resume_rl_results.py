# description: Audit RL results and request missing quarterly training
# content: 完整则打包补发；未完成且已完成产物校验通过则提交共享训练插槽恢复请求，不在 autofix 内训练。
# autofix: true
# email: true
# mode: shell
# parameters:
#   run_directory:
#     type: str
#     default: results/rl_experimental/walk_forward/gru_v1_low_turnover_v1/automatic/5d17935c80694717a8df5fbd055791d4
#     desc: 原季度实验目录
#   dry_run:
#     type: bool
#     default: false
#     desc: 只诊断，不打包或提交恢复请求

from pathlib import Path
from uuid import uuid4

from src.proj import PATH, Proj
from src.proj.util.script import ScriptTool, AutoRunTask


@ScriptTool('autofix_recover_or_resume_rl_results')
def main(run_directory: str = 'results/rl_experimental/walk_forward/gru_v1_low_turnover_v1/automatic/5d17935c80694717a8df5fbd055791d4',
         dry_run: bool = False, **kwargs):
    from src.res.rl_experimental.recover_results import recover_suite, audit_suite, read_json
    from src.res.rl_experimental.resume import inspect_resume
    from src.proj.util.web.task_mail import TaskResultMail, save_json
    from src.res.rl_experimental.bundle import bundle_attachments
    from src.automation.scheduling.rl_resume import submit
    from src.automation.scheduling.maintenance_gate import sandbox_request
    root = Path(PATH.main).resolve()
    run = (root/run_directory).resolve()
    if not run.is_relative_to(root/'results/rl_experimental') or not run.is_dir():
        raise ValueError('Invalid existing RL run directory')
    if dry_run:
        audit = audit_suite(run, project_root=root)
        print(audit)
        if not audit['complete']:
            plan = inspect_resume(run, root)
            print({'reusable': len(plan['completed']), 'missing': plan['pending'], 'source': plan['source_commit']})
        return
    destination = run/'recovery'/('audit-or-resume-'+uuid4().hex)
    task = AutoRunTask._instances[-1]
    task.email_sender = TaskResultMail(destination/'mail')
    task.kwargs['email_recipient'] = read_json(run/'config.json').get('recipient')
    logs = list((root/'.local_resources').glob(f'*/runtime/scheduling/idle_worklist/{run.name}.log'))
    audit, metadata = recover_suite(run, destination, project_root=root, log_path=logs[0] if len(logs) == 1 else None)
    Proj.email_attachments.append(destination/'audit.json')
    if metadata:
        for attachment in bundle_attachments(metadata):
            Proj.email_attachments.append(attachment)
        print('[RL/recovery] All training/testing verified; result bundle attached, no resume needed.', flush=True)
        return
    try:
        plan = inspect_resume(run, root)
    except Exception as exc:
        save_json(destination/'resume_rejected.json', {'error': f'{type(exc).__name__}: {exc}'})
        Proj.email_attachments.append(destination/'resume_rejected.json')
        raise
    save_json(destination/'resume_plan.json', plan)
    Proj.email_attachments.append(destination/'resume_plan.json')
    context = sandbox_request()
    request_id = context['id'] if context else '20261028_resume_rl_5d17935c_02'
    request = submit(root, request_id, plan)
    print(f'[RL/recovery] Reusable={len(plan["completed"])}, pending={len(plan["pending"])}. '
          f'Request queued for shared GPU slot: {request}. No training inside autofix.', flush=True)
    return {'status': 'resume_requested', 'request': str(request), 'tests_complete': False}


if __name__ == '__main__':
    main()
