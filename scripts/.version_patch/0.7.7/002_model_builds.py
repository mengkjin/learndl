# coding: utf-8
# description: Index source model builds and consolidate previously received runs.
# email: False
# mode: os
from src.automation.version_patch import load_patch_support


def apply(ctx):
    load_patch_support('0.7.7', 'model_builds').apply(ctx)


def verify(ctx):
    return load_patch_support('0.7.7', 'model_builds').verify()
