"""Reload project modules under a filesystem tree; optional Streamlit / IPython helpers.

Compared to ``IPython.lib.deepreload`` (recursive ``reload`` of one module while
temporarily patching ``builtins.__import__``):

- **This module's** :func:`deepreload` scans ``sys.modules`` and reloads every module
  whose source file lies under *root* (typical for a monorepo ``src/`` tree). It does
  not patch global import, does not print, and is scoped by path.
- **IPython** reloads the transitive import graph starting from a *single* entry
  module. That can fix stale bindings for code imported only through that graph, but
  it temporarily replaces ``__import__``, prints ``Reloading ...`` unless suppressed,
  and is unsafe if the entry module imports Streamlit: the hook may try to reload
  ``streamlit.*`` internals. Use :func:`ipython_recursive_reload` only for
  non-Streamlit packages or after merging a broad ``streamlit.*`` exclude list (see
  implementation).
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import inspect
import io
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Literal, TypeAlias

StrPath: TypeAlias = str | Path
RerunScope: TypeAlias = Literal["app", "fragment"]

__all__ = ['deepreload' , 'ipython_recursive_reload' , 'streamlit_hot_reload']


def _resolved_source_path(module: ModuleType) -> Path | None:
    """Return a resolved ``.py`` path for *module*, or ``None`` if unknown / not file-backed."""
    raw = getattr(module, "__file__", None)
    if not raw:
        return None
    try:
        path = Path(raw).resolve()
    except OSError:
        return None
    if path.suffix in (".py", ".pyw"):
        return path
    if path.suffix == ".pyc":
        try:
            src = importlib.util.source_from_cache(str(path))
        except (OSError, ValueError):
            return None
        if not src:
            return None
        try:
            return Path(src).resolve()
        except OSError:
            return None
    return None


def _names_to_reload(
    root: Path,
    modules: Mapping[str, ModuleType | None],
    *,
    exclude_names: frozenset[str],
) -> list[str]:
    names: list[str] = []
    for name, mod in modules.items():
        if not name or mod is None or name in exclude_names:
            continue
        src = _resolved_source_path(mod)
        if src is None:
            continue
        try:
            if not src.is_relative_to(root):
                continue
        except ValueError:
            continue
        names.append(name)

    def sort_key(n: str) -> tuple[int, str]:
        return (-n.count("."), n)

    names.sort(key=sort_key)
    return names


def deepreload(
    root: StrPath,
    *,
    modules: Mapping[str, ModuleType | None] | None = None,
    exclude: Iterable[str] | None = None,
) -> tuple[list[str], list[tuple[str, BaseException]]]:
    """
    Reload all entries in *modules* whose ``__file__`` (or derived ``.py`` from ``.pyc``)
    lies under *root*.

    Modules are reloaded deepest-first (longest dotted name first) to reduce stale
    submodule references after parent packages reload.

    Args:
        root: Directory tree to match (e.g. your ``src`` folder). Resolved with
            :func:`pathlib.Path.resolve`.
        modules: Map like ``sys.modules``; defaults to ``sys.modules``.
        exclude: Extra module names to skip (e.g. ``("some_pkg._no_reload",)``).

    Returns:
        ``(reloaded, failed)`` where *reloaded* is successful ``__name__`` strings and
        *failed* is ``(name, exception)`` pairs.
    """
    root_path = Path(root).expanduser().resolve(strict=False)
    mapping = sys.modules if modules is None else modules
    exclude_names = frozenset({"__main__", *tuple(exclude or ())})
    targets = _names_to_reload(root_path, mapping, exclude_names=exclude_names)

    reloaded: list[str] = []
    failed: list[tuple[str, BaseException]] = []

    for name in targets:
        mod = mapping.get(name)
        if mod is None:
            continue
        try:
            importlib.reload(mod)
        except BaseException as exc:
            failed.append((name, exc))
        else:
            reloaded.append(name)

    return reloaded, failed


def _ipython_default_exclude() -> tuple[str, ...]:
    from IPython.lib.deepreload import reload as ipreload

    default = inspect.signature(ipreload).parameters["exclude"].default
    if isinstance(default, tuple):
        return default
    return tuple(default)


def _loaded_streamlit_module_names() -> tuple[str, ...]:
    return tuple(n for n in sys.modules if n == "streamlit" or n.startswith("streamlit."))


def ipython_recursive_reload(
    module: ModuleType,
    *,
    exclude: Iterable[str] | None = None,
    silence_output: bool = True,
) -> ModuleType:
    """
    Run ``IPython.lib.deepreload.reload`` on *module* (recursive reload via import hook).

    IPython's ``exclude`` replaces its built-in defaults; this function merges the
    signature defaults, every loaded ``streamlit.*`` name (so Streamlit is not
    deep-reloaded when your entry file imports it), and *exclude*.

    Args:
        module: Typically your app or package root module.
        exclude: Additional module names to skip.
        silence_output: When True, suppress IPython's ``Reloading ...`` prints.

    Returns:
        The reloaded module object (same as ``importlib.reload``).
    """
    from IPython.lib.deepreload import reload as ipreload

    merged = tuple(
        dict.fromkeys((*_ipython_default_exclude(), *_loaded_streamlit_module_names(), *tuple(exclude or ())))
    )
    if silence_output:
        sink = io.StringIO()
        ctx = contextlib.redirect_stdout(sink)
        ctx_err = contextlib.redirect_stderr(sink)
    else:
        ctx = contextlib.nullcontext()
        ctx_err = contextlib.nullcontext()
    with ctx, ctx_err:
        return ipreload(module, exclude=merged)


@dataclass(slots=True)
class StreamlitHotReloadResult:
    """Outcome of :func:`streamlit_hot_reload` for UI feedback."""

    reloaded: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    cleared_cache_resource: bool = False
    cleared_cache_data: bool = False
    ipython_entry_applied: bool = False


def streamlit_hot_reload(
    root: StrPath,
    *,
    clear_cache_resource: bool = True,
    clear_cache_data: bool = True,
    exclude: Iterable[str] | None = None,
    ipython_entry: ModuleType | None = None,
    rerun: bool = True,
    rerun_scope: RerunScope = "app",
) -> StreamlitHotReloadResult:
    """
    Clear Streamlit caches, reload all modules under *root*, optionally IPython-reload
    an entry module, then ``st.rerun()`` when *rerun* is True.

    Intended for a dev-only button::

        if st.button("Reload code"):
            streamlit_hot_reload(SRC_ROOT)

    Clears ``st.cache_resource`` / ``st.cache_data`` **before** module reload so
    ``on_release`` hooks run while caches still see the pre-reload world.

    Args:
        root: Filesystem root passed to :func:`deepreload`.
        clear_cache_resource: Call ``st.cache_resource.clear()`` when True.
        clear_cache_data: Call ``st.cache_data.clear()`` when True.
        exclude: Extra module names for :func:`deepreload`.
        ipython_entry: If set, run :func:`ipython_recursive_reload` after
            :func:`deepreload` for that module (optional graph cleanup).
        rerun: If True, call ``st.rerun()`` so the next run uses fresh code.
        rerun_scope: Forwarded to ``st.rerun(scope=...)``; use ``fragment`` only where
            Streamlit allows fragment-scoped reruns.

    Returns:
        :class:`StreamlitHotReloadResult`. When *rerun* is True, Streamlit normally
        stops the run before returning; use ``rerun=False`` if you need the object
        in the same run (then call ``st.rerun()`` yourself).
    """
    import streamlit as st

    result = StreamlitHotReloadResult()

    if clear_cache_resource:
        st.cache_resource.clear()
        result.cleared_cache_resource = True
    if clear_cache_data:
        st.cache_data.clear()
        result.cleared_cache_data = True

    reloaded, failed_exc = deepreload(root, exclude=exclude)
    result.reloaded = reloaded
    result.failed = [(n, repr(e)) for n, e in failed_exc]

    if ipython_entry is not None:
        try:
            ipython_recursive_reload(ipython_entry)
        except BaseException as e:
            result.failed.append((ipython_entry.__name__, repr(e)))
        else:
            result.ipython_entry_applied = True

    if rerun:
        st.rerun(scope=rerun_scope)

    return result
