# description: Read factor sequence progress and email diagnostic evidence
# content: 只读检查阶段、调度账本、日志和模型状态并邮件报告，不启动或重启训练。
# autofix: true
# email: true
# mode: shell
# parameters:
#   run_id:
#     type: str
#     default: 20261031_factor_sequence_verify_02
#     desc: Existing factor sequence run to inspect

from src.proj.util.script import ScriptTool


@ScriptTool('autofix_check_factor_sequence_progress')
def main(run_id: str = '20261031_factor_sequence_verify_02', **kwargs):
    import json
    from uuid import uuid4
    from src.proj import PATH, Proj
    from src.proj.util.script import AutoRunTask
    from src.proj.util.web.task_mail import TaskResultMail
    from src.automation.scheduling.config import runtime_dir
    from src.automation.scheduling.factor_sequence_progress import collect
    from src.automation.scheduling.maintenance_gate import sandbox_request
    context = sandbox_request()
    check_id = context['id'] if context else 'manual_'+uuid4().hex
    destination = PATH.result/'factor_sequence_diagnostics'/check_id
    destination.mkdir(parents=True, exist_ok=False)
    AutoRunTask._instances[-1].email_sender = TaskResultMail(destination/'mail')
    report = collect(PATH.result/'factor_sequence', runtime_dir(), run_id, isolated=bool(context))
    path = destination/'progress.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    Proj.email_attachments.append(path)
    summary = {key: report[key] for key in ('run_id','assessment','current_stage','completed_stages','mail','notes')}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Fold records: {len(report["folds"])}; diagnostics: {path}')
    return summary


if __name__ == '__main__':
    main()
