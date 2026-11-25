# coding: utf-8
# description: Replay affected portfolios after the Dongfang sellside cutoff.
# content: Back up recorded outputs and rebuild the portfolio chain with corrected sellside reads.
# email: False
# mode: os

from src.automation.version_patch import load_patch_support


def apply(ctx):
    load_patch_support('0.7.11', 'sellside_portfolios', root=ctx.root).apply(ctx)


def verify(ctx):
    return load_patch_support('0.7.11', 'sellside_portfolios', root=ctx.root).verify(ctx)
