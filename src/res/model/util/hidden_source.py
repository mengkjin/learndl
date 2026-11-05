"""Compatibility entry point for exact hidden-source references."""
from __future__ import annotations

__all__ = [
    'hidden_source_path',
]
from .model_source import model_source_path


def hidden_source_path(reference: str):
    return model_source_path(reference)
