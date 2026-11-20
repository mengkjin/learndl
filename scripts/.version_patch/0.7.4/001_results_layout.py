# coding: utf-8
# description: Migrate project results and received research batches to the 0.7.4 layout.
# email: False
# mode: os
from src.automation.artifact_exchange.core import Config
from src.automation.version_patch import load_patch_support


def apply(ctx):
    layout = load_patch_support('0.7.4', 'results')
    layout.apply(Config.load(), ctx.root / '.local_resources/learndl_monitor', ctx.backup_dir / 'results.json')


def verify(ctx):
    layout = load_patch_support('0.7.4', 'results')
    config = Config.load()
    pending = layout.discover(config, ctx.root / '.local_resources/learndl_monitor')
    return (not pending['moves'] and not pending['aliases'] and
            layout.read(config.state / 'migrations/results-0.7.4.json').get('complete') is True)
