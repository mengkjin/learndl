# description: Delete recorded legacy fold model directories and train six ordinary schedules
# content: 宿主确认旧任务退出后永久删除旧版逐折模型目录，不备份，再按顺序运行六个普通 schedule 训练任务。
# autofix: true
# email: true
# mode: shell
# parameters:
#   dry_run:
#     type: bool
#     default: false
#     desc: Show scope without submitting the destructive cleanup and training request

from src.proj.util.script import ScriptTool


@ScriptTool('autofix_restart_factor_sequence_schedules')
def main(dry_run=False, **kwargs):
    from src.automation.scheduling.maintenance_gate import sandbox_request
    from src.automation.scheduling.factor_sequence import submit
    from src.automation.scheduling.factor_sequence_restart import LEGACY_RUNS, MODELS
    context=sandbox_request()
    request_id=context['id'] if context else '20261103_factor_sequence_schedules_04'
    print(f'Permanent deletion, no backup, restricted to recorded per-fold NN models from: {LEGACY_RUNS}')
    print(f'Then train sequentially through the normal schedule interface: {MODELS}')
    if not dry_run:
        print(f'Request queued: {submit(request_id,cleanup_legacy=True)}. Host performs cleanup and training in the shared worker slot.')


if __name__=='__main__':
    main()
