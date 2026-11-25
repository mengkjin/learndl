# coding: utf-8
# description: Restore original use_daily history and replay from 20261124.
# content: Seed the production replay from the original November 23 holdings.
# email: False
# mode: os

from src.automation.version_patch import load_patch_support


def apply(ctx):
    load_patch_support('0.7.14', 'use_daily_history', root=ctx.root).apply(ctx)


def verify(ctx):
    return load_patch_support('0.7.14', 'use_daily_history', root=ctx.root).verify(ctx)
