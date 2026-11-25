# description: Rebuild affected sellside model tests after the production holdings repair.
# content: Replace stale prediction records, test portfolios, accounts and reports.
# email: False
# mode: os
from src.automation.version_patch import load_patch_support


def apply(ctx):
    load_patch_support('0.7.14', 'sellside_tests', root=ctx.root).apply(ctx)


def verify(ctx):
    return load_patch_support('0.7.14', 'sellside_tests', root=ctx.root).verify(ctx)
