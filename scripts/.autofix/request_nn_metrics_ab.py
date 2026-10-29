# description: Queue four default NN short tests and email the analysis bundle
# content: 提交旧新新旧四次 short test 请求；维护结束后由共享 GPU worker 执行并发送结果邮件。
# autofix: true
# email: true
# mode: shell
# parameters:
#   dry_run:
#     type: bool
#     default: false
#     desc: 只显示计划，不提交请求

from src.proj.util.script import ScriptTool


@ScriptTool('autofix_request_nn_metrics_ab')
def main(dry_run: bool = False, **kwargs):
    from src.api.task_monitor.scheduling.maintenance_gate import sandbox_request
    from src.api.task_monitor.scheduling.nn_metrics_ab import submit
    context = sandbox_request()
    request_id = context['id'] if context else '20261029_nn_metrics_ab_01'
    print(f'NN A/B: {request_id}; legacy1 -> fast1 -> fast2 -> legacy2; seed=20261029')
    if not dry_run:
        print(f'Request queued: {submit(request_id)}. Training and result mail follow after maintenance.')


if __name__ == '__main__':
    main()
