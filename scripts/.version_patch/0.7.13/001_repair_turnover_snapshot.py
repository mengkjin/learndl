# coding: utf-8
# description: Repair the legacy 20260622 turn_1m stock factor snapshot.
# content: Back up the Feather file and verify exact preservation of securities and signals.
# email: False
# mode: os

from src.automation.version_patch import load_patch_support


def apply(ctx):
    load_patch_support('0.7.13', 'turnover_snapshot', root=ctx.root).apply(ctx)


def verify(ctx):
    return load_patch_support('0.7.13', 'turnover_snapshot', root=ctx.root).verify(ctx)
