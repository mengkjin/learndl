"""Persisted model properties and build identity shared with lightweight receivers."""
from __future__ import annotations
from contextlib import suppress
import json
import re

from .file_ops import real_tree

__all__ = ['KINDS', 'PROPERTY_FILE', 'LOG_PHASES', 'read_properties', 'validate_build']


KINDS = ('build', 'initial_training', 'resume_training', 'last_test')
PROPERTY_FILE = 'logs/property.jsonl'
LOG_PHASES = ('train', 'test')
BUILD_NAME = re.compile(r'build[0-9]{8}\.[0-9]{6}\.[0-9]+')


def read_properties(path):
    with suppress(FileNotFoundError):
        return {row['kind']: row for line in real_tree(path).read_text().splitlines() if line.strip()
                for row in [json.loads(line)] if row['kind'] in KINDS}
    return {}


def validate_build(value):
    if (not isinstance(value, dict) or not BUILD_NAME.fullmatch(value.get('name', ''))
            or not re.fullmatch(r'[0-9a-f]{32}', value.get('fingerprint', ''))):
        raise ValueError('Invalid model build identity')
    return value
