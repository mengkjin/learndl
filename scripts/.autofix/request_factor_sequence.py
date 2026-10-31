# description: Queue factor sequence validation from 2017 through latest mature labels
# content: 提交一次性因子时序验证请求；由共享 GPU worker 空闲时执行并邮件发送结果。
# autofix: true
# email: true
# mode: shell
# parameters:
#   dry_run:
#     type: bool
#     default: false
#     desc: 只显示计划，不提交请求

from src.proj.util.script import ScriptTool


@ScriptTool('autofix_request_factor_sequence')
def main(dry_run: bool = False, **kwargs):
    from src.automation.scheduling.maintenance_gate import sandbox_request
    from src.automation.scheduling.factor_sequence import submit
    context = sandbox_request()
    request_id = context['id'] if context else '20261031_factor_sequence_verify_01'
    print(f'Factor sequence: {request_id}; OOS 2017 to latest; six models, three epochs each')
    if not dry_run:
        print(f'Request queued: {submit(request_id)}. Training and result mail follow after maintenance.')


if __name__ == '__main__':
    main()
