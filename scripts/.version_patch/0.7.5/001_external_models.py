# coding: utf-8
# description: Rename the synchronized model archive and update stored references.
# email: False
# mode: os
from contextlib import ExitStack

import portalocker

from src.automation.artifact_exchange.core import Config, fsync_dir, locked, real_tree
from src.automation.version_patch import load_patch_support


def apply(ctx):
    config = Config.load()
    monitor = ctx.root / '.local_resources/learndl_monitor'
    external = config.results / 'external'
    old, new = real_tree(external / 'model'), real_tree(external / 'models')
    with ExitStack() as stack:
        for kind in ('mail_archive', 'model_preview'):
            if (monitor / kind).is_dir():
                stack.enter_context(portalocker.Lock(monitor / kind / 'operations.lock', timeout=0))
        stack.enter_context(locked(config, 'receive'))
        if (monitor / 'experiments').is_dir():
            stack.enter_context(portalocker.Lock(monitor / 'experiments/operations.lock', timeout=0))
        if old.exists():
            if not old.is_dir():
                raise ValueError(f'Expected a directory: {old}')
            if new.exists():
                # A direct upgrade from 0.7.3 can leave the old scaffold empty.
                old.rmdir()
            else:
                old.rename(new)
            fsync_dir(external)
        layout = load_patch_support('0.7.4', 'results')
        layout.update_state(config, monitor, {old: new})


def verify(ctx):
    return not real_tree(Config.load().results / 'external/model').exists()
