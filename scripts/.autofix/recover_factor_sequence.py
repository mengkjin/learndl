# description: Inspect factor sequence progress, resend results and queue safe recovery
# content: 非阻塞检查旧结果并提交宿主恢复；确认无原进程后补发结果，复用完整折并重跑未完成折，每次训练单独邮件。
# autofix: true
# email: true
# mode: shell
# parameters:
#   run_id:
#     type: str
#     default: 20261031_factor_sequence_verify_02
#     desc: Existing run ID
#   dry_run:
#     type: bool
#     default: false
#     desc: Inspect only; do not submit recovery

from src.proj.util.script import ScriptTool


@ScriptTool('autofix_recover_factor_sequence')
def main(run_id='20261031_factor_sequence_verify_02', dry_run=False, **kwargs):
    import json
    from uuid import uuid4
    from src.proj import PATH, Proj
    from src.proj.util.script import AutoRunTask
    from src.proj.util.web.task_mail import TaskResultMail
    from src.automation.scheduling.config import runtime_dir
    from src.automation.scheduling.factor_sequence_progress import collect
    from src.automation.scheduling.factor_sequence import submit
    from src.automation.scheduling.maintenance_gate import sandbox_request
    context=sandbox_request()
    request_id=context['id'] if context else 'manual_recover_'+uuid4().hex
    destination=PATH.result/'factor_sequence_diagnostics'/request_id
    destination.mkdir(parents=True,exist_ok=False)
    AutoRunTask._instances[-1].email_sender=TaskResultMail(destination/'mail')
    report=collect(PATH.result/'factor_sequence',runtime_dir(),run_id,isolated=bool(context))
    path=destination/'progress.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    Proj.email_attachments.append(path)
    print(f"Original {run_id}: {report['assessment']}; evidence: {path}",flush=True)
    if not dry_run:
        request=submit(request_id,resume_from=run_id)
        print(f'Recovery queued: {request}. Host will check original process/children, resend available results, '
              'and resume incomplete folds in the shared training slot. No training in autofix.',flush=True)


if __name__=='__main__':
    main()
