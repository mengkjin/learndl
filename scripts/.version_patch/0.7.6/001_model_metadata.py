# coding: utf-8
# description: Initialize model properties and backfill previously published models.
# email: False
# mode: os
from src.automation.version_patch import load_patch_support


def apply(ctx):
    load_patch_support('0.7.6', 'model_metadata').apply(ctx)


def verify(ctx):
    return load_patch_support('0.7.6', 'model_metadata').verify(ctx)
