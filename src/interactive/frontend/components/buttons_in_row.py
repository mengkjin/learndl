# coding: utf-8
"""Horizontal ``sac.buttons`` row where each entry has its own ``on_click`` handler."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import streamlit as st
import streamlit_antd_components as sac

__all__ = ["SacBoundButton", "SacOnClickButtons"]

@dataclass
class SacBoundButton:
    """One ``sac.ButtonsItem`` plus a zero-arg callback invoked when that button is chosen."""
    label: str
    icon: Any = None
    disabled: bool = False
    color: Any = None
    on_click: Callable[[], None] = lambda: None

    def __post_init__(self) -> None:
        self.item = sac.ButtonsItem(label = self.label , icon = self.icon , color = self.color)

class SacOnClickButtons:
    """Renders ``sac.buttons`` and dispatches ``on_click`` per index (``return_index=True``)."""

    def __init__(
        self, buttons: Sequence[SacBoundButton] , key: str , 
        size: str = "md" , variant: str = "outline" , color: str | None = None , align: str = "start") -> None:
        self._base_key = key
        self._buttons = tuple(buttons)
        self._buttons_kwargs = {
            "size": size,
            "variant": variant,
            "color": color,
            "align": align,
        }

    @staticmethod
    def _picked_index(picked: object, n: int) -> int | None:
        """Coerce ``sac.buttons`` value to an item index; ``0`` must not be treated as falsy."""
        if picked is None:
            return None
        if isinstance(picked, bool):
            return None
        if isinstance(picked, int):
            i = picked
        elif isinstance(picked, str) and picked.isdigit():
            i = int(picked)
        else:
            return None
        if 0 <= i < n:
            return i
        return None

    def render(
        self,
    ) -> None:
        """Draw the row; on user click, run the matching ``on_click`` then remount via ``key`` bump.

        ``streamlit_antd_components`` does not reliably invoke ``on_change`` for ``sac.buttons``;
        the return value on the post-interaction run is used instead (same pattern as the repo demo).

        Nonce is incremented *before* invoking handlers because handlers may call ``st.rerun()``
        (e.g. :meth:`ParamInputsForm.reset_options`) and never return; otherwise the bump would be
        skipped and the widget would keep the same ``key`` and the same pick forever.
        """
        nonce_key = f"{self._base_key}__sac_nonce"
        st.session_state.setdefault(nonce_key, 0)
        row_key = f"{self._base_key}__sac_row_{st.session_state[nonce_key]}"
        handlers = tuple(b.on_click for b in self._buttons)
        items: list[str | dict[str, Any] | sac.ButtonsItem] = [b.item for b in self._buttons]

        sac_kw: dict[str, Any] = {
            "index": None,
            "return_index": True,
            "key": row_key,
            **self._buttons_kwargs,
        }
        picked = sac.buttons(items, **sac_kw) # type: ignore[reportUnknownReturnType]
        idx = self._picked_index(picked, len(handlers))
        if idx is not None:
            # Bump *before* handlers: e.g. ``ParamInputsForm.reset_options()`` calls
            # ``st.rerun()`` and never returns, so a post-handler bump would never run
            # and the same ``sac.buttons`` key would keep returning the picked index.
            st.session_state[nonce_key] = int(st.session_state[nonce_key]) + 1
            handlers[idx]()
            st.rerun()