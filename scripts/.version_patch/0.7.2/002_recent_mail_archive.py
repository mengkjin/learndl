# coding: utf-8
# description: Backfill seven days of confirmed task mail to Syncthing.
# email: False
# mode: os

import json


def apply(ctx):
    from src.automation.artifact_exchange.core import Config, atomic_json
    from src.automation.artifact_exchange.mail_backfill import host_inputs, run

    report = ctx.backup_dir / 'recent-mail.json'
    config = Config.load()
    summary = run(config, *host_inputs(), report=report)
    if config.role != 'publisher':
        atomic_json(report, summary)


def verify(ctx):
    summary = json.loads((ctx.backup_dir / 'recent-mail.json').read_text())
    return summary['role'] == 'receiver' or summary['counts']['failed'] == 0
