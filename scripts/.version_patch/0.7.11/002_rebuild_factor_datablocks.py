# coding: utf-8
# description: Rebuild stock-factor DataBlocks after fixing factor selection.
# content: Back up affected fit and predict dumps and verify coverage against stored factors.
# email: False
# mode: os

from src.automation.version_patch import load_patch_support


def apply(ctx):
    load_patch_support('0.7.11', 'factor_datablocks', root=ctx.root).apply(ctx)


def verify(ctx):
    return load_patch_support('0.7.11', 'factor_datablocks', root=ctx.root).verify(ctx)
