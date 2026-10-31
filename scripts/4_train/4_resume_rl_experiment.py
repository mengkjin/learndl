# description: Resume RL Experiment (shared idle worker only)
# content: 按显式恢复请求复用旧模型，补齐季度训练和测试；由共享训练插槽执行。
# email: true
# mode: shell
# parameters:
#   request_path:
#     type: str
#     required: true
#     desc: 由 autofix 生成的不可变恢复请求路径

import os
from pathlib import Path
from uuid import uuid4

from src.proj import PATH, Proj
from src.proj.util.script import ScriptTool, AutoRunTask


@ScriptTool('resume_rl_experiment')
def main(request_path: str, **kwargs):
    from src.automation.scheduling.rl_resume import validate_request, request_dir
    from src.res.rl_experimental.recover_results import audit_suite, recover_suite, read_json
    from src.res.rl_experimental.resume import resume_suite
    from src.res.rl_experimental.bundle import bundle_attachments
    from src.proj.util.web.task_mail import TaskResultMail, save_json
    if not os.environ.get('LEARNDL_MANAGED_RUN'):
        raise RuntimeError('Use a shared idle-worker resume request; do not run GPU recovery inside autofix or directly')
    path = Path(request_path).resolve()
    if path.parent != request_dir(PATH.main).resolve():
        raise ValueError('Unexpected resume request location')
    doc = validate_request(read_json(path), PATH.main)
    run = Path(doc['run_directory'])
    attempt = run/'recovery'/('resume-'+uuid4().hex)
    task = AutoRunTask._instances[-1]
    task.kwargs['email'] = not doc['no_email']
    task.kwargs['email_recipient'] = doc['recipient']
    task.email_sender = TaskResultMail(attempt/'mail')
    try:
        audit = audit_suite(run, project_root=PATH.main)
        if not audit['complete']:
            resume_suite(run, PATH.main, attempt)
        else:
            attempt.mkdir(parents=True)
            print('[RL/resume] Results already complete; skip all training and testing.', flush=True)
        audit, metadata = recover_suite(run, attempt/'verified_results', project_root=PATH.main)
        if not audit['complete']:
            raise ValueError('Resumed results failed final audit: ' + '; '.join(audit['errors'][:5]))
        if metadata is None:
            raise ValueError('Recovered suite did not produce bundle metadata')
        for attachment in bundle_attachments(metadata):
            Proj.email_attachments.append(attachment)
        return {'status': 'success', 'original_run': str(run), 'bundle': str(metadata)}
    finally:
        attempt.mkdir(parents=True, exist_ok=True)
        save_json(attempt/'final_audit.json', audit_suite(run, project_root=PATH.main))
        Proj.email_attachments.append(attempt/'final_audit.json')


if __name__ == '__main__':
    main()
