# coding: utf-8
# description: Move model experiments into their results directories.
# email: False
# mode: os

import json


def apply(ctx):
    from src.automation.artifact_exchange.core import Config
    from src.automation.artifact_exchange.result_layout import run
    from src.proj import PATH

    run(Config.load(), PATH.model, PATH.model_archive, PATH.optuna,
        report=ctx.backup_dir / 'experiment-results.json')


def verify(ctx):
    summary = json.loads((ctx.backup_dir / 'experiment-results.json').read_text())
    return all(item['state'] in ('migrated', 'already-migrated') for item in summary['moves'])
