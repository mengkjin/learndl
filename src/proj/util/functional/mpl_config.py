"""Process-wide matplotlib backend selection for batch-safe plotting.

Default is ``Agg`` (headless, thread-safe) so async ``Save.figs`` / ``plt.close``
on worker threads does not hit Tk/Tcl ``main thread is not in main loop``.

Override before importing matplotlib::

    export MPLBACKEND=TkAgg          # GUI
    export MPLBACKEND=module://matplotlib_inline.backend_inline  # notebooks
"""
from __future__ import annotations

import os

_DEFAULT_BACKEND = 'Agg'
_configured = False


def prefer_agg_backend() -> None:
    """Set ``MPLBACKEND`` to Agg when unset (no matplotlib import)."""
    os.environ.setdefault('MPLBACKEND', _DEFAULT_BACKEND)


def configure_matplotlib(backend: str | None = None) -> str:
    """Select matplotlib backend once, before ``pyplot`` is imported.

    Args:
        backend: Explicit backend name. When omitted, uses ``MPLBACKEND`` if set,
            otherwise ``Agg``.

    Returns:
        Active backend name after configuration.
    """
    global _configured
    import matplotlib

    if _configured:
        return str(matplotlib.get_backend())

    prefer_agg_backend()
    chosen = backend or os.environ.get('MPLBACKEND', _DEFAULT_BACKEND)
    # force=False: do not fight an already-selected interactive backend mid-session
    matplotlib.use(chosen, force=False)
    _configured = True
    return str(matplotlib.get_backend())


# Importing this module from ``src.proj`` sets the env default before pyplot loads.
prefer_agg_backend()
