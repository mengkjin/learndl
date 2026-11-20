"""Normalize the renamed status callback without changing unrelated YAML values."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any

__all__ = ['OLD', 'NEW', 'rename_status_callback']

OLD = 'StatusDisplay'
NEW = 'StatusLogger'


def rename_status_callback(value: Mapping, *, legacy: bool = False) -> dict:
    source, target = (NEW, OLD) if legacy else (OLD, NEW)
    seen = {}

    def visit(node, prefix=()) -> Any:
        if isinstance(node, Mapping):
            result = {}
            for key, item in node.items():
                parts = str(key).split('.')
                path = (*prefix, *parts)
                renamed = False
                callback_key = False
                for i, part in enumerate(parts):
                    if part in (source, target) and len(prefix) + i > 0 and path[len(prefix) + i - 1] == 'callbacks':
                        callback_key = True
                        renamed |= parts[i] != target
                        parts[i] = target
                name = '.'.join(parts) if renamed else key
                item = visit(item, path)
                if callback_key:
                    normalized = (*prefix, *parts)
                    if normalized in seen:
                        if seen[normalized] != item:
                            raise ValueError(f'Conflicting {OLD} and {NEW} configuration at {".".join(path)}')
                        continue
                    seen[normalized] = item
                if name in result and result[name] != item:
                    raise ValueError(f'Conflicting {OLD} and {NEW} configuration at {".".join(path)}')
                result[name] = item
            return result
        if isinstance(node, list) and prefix and prefix[-1] == 'callbacks':
            items = []
            for item in node:
                if item in (source, target):
                    if target not in items:
                        items.append(target)
                else:
                    items.append(item)
            return items
        return node

    return visit(value)
