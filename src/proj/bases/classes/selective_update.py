"""Selective / menu-driven update protocol shared by data updaters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from src.proj.cal import CALENDAR
from src.proj.core import lit
from src.proj.bases.enums import UpdateFlag

__all__ = [
    'UpdateMenuNode',
    'SelectiveUpdateSupport',
    'resolve_force_range',
    'ALL_UNDER_LABEL',
]

ALL_UNDER_LABEL = '* All under this'


@dataclass
class UpdateMenuNode:
    """One node in a multi-level selective-update menu tree."""

    label: str
    key: str | None = None
    children: list[UpdateMenuNode] = field(default_factory=list)
    help: str = ''
    disabled: bool = False

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def all_leaf_keys(self) -> list[str]:
        """Collect enabled leaf keys under this node (depth-first)."""
        if self.disabled:
            return []
        if self.is_leaf:
            return [self.key] if self.key is not None else []
        keys: list[str] = []
        for child in self.children:
            keys.extend(child.all_leaf_keys())
        return keys

    def child_by_label(self, label: str) -> UpdateMenuNode | None:
        for child in self.children:
            if child.label == label:
                return child
        return None


def resolve_force_range(
    force: bool,
    start: int | None,
    end: int | None,
    *,
    key: lit.DataUpdateKey | None = None,
) -> tuple[bool, int | None, int | None]:
    """
    Normalize force + date range for selective updates.

    - ``force=False`` → incremental (no range).
    - ``force=True`` and both ``start``/``end`` are ``None`` → treat as non-force
      (default incremental update).
    - ``force=True`` with only one bound → fill the other via ``CALENDAR.update_schedule``.
    """
    if not force:
        return False, None, None
    if start is None and end is None:
        return False, None, None
    filled_start, filled_end = CALENDAR.update_schedule(
        start if start is not None else 19000101,
        end,
        key=key,
    )
    return True, filled_start, filled_end


class SelectiveUpdateSupport:
    """
    Mixin for orchestrators that expose a selection tree and selective update.

    Subclasses must implement ``selection_tree`` and ``selective_update``.
    """

    @classmethod
    def selection_tree(cls) -> UpdateMenuNode:
        raise NotImplementedError(f'selection_tree is not implemented for {cls.__name__}')

    @classmethod
    def selective_update(
        cls,
        selection: Sequence[str],
        *,
        force: bool = False,
        start: int | None = None,
        end: int | None = None,
        **kwargs: Any,
    ) -> UpdateFlag:
        raise NotImplementedError(f'selective_update is not implemented for {cls.__name__}')
