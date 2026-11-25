#!/usr/bin/env python
# description: Continue registered production models that are due for training
# email: False
# mode: shell

import os

from src.proj import Logger, MACHINE
from src.proj.util.script import ScriptTool


@ScriptTool('production_model_update', lock_name='daily_update', email=False)
def main():
    import psutil

    from src.api.pkgs.data import DataAPI
    from src.automation.scheduling.runtime import RunStore, members
    from src.res.model.model_module.application import ModelTrainer

    if not (MACHINE.updatable and MACHINE.cuda_server):
        Logger.info('Production training requires an updatable CUDA server.')
        return
    if not DataAPI.is_updated():
        Logger.info('Production training data is no longer up to date. Returning the request to the queue.')
        return 75

    store = RunStore()
    run = store.get(os.environ.get('LEARNDL_MANAGED_RUN', ''))

    def notify(kind, message):
        if kind == 'waiting':
            candidates = []
            for active in store.active():
                if active.get('owner') != 'idle_worklist' or not active.get('pid'):
                    continue
                try:
                    if members(active):
                        candidates.append(f"{active['task']} (PID {active['pid']})")
                except (OSError, RuntimeError, psutil.Error):
                    continue
            message += '\nActive managed research processes: ' + (', '.join(candidates) or 'none identified')
        if run.get('task') == 'production_model_update' and run.get('pid') == os.getpid():
            store.event(run, f'production_{kind}', message)
        else:
            from src.proj.util.web.emailer import Email
            Email.send(f'Production Model Update - {kind}', message)

    ModelTrainer.update_models(production_notice=notify)


if __name__ == '__main__':
    result = main()
    raise SystemExit((result.func_return or 0) if result.execution_success else 1)
