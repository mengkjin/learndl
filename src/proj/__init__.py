"""Root package for project infrastructure: environment, paths, DB, calendar, logging, and ``Proj`` facade."""
from __future__ import annotations

# Side effect: ``os.environ.setdefault('MPLBACKEND', 'Agg')`` before any pyplot import.
# Override with ``MPLBACKEND`` (e.g. TkAgg / matplotlib_inline) when interactive.
from .util.functional import mpl_config as _mpl_config  # noqa: F401

from .env import MACHINE , PATH , Const , Proj , Options
from .log import Logger
from .db import Save , Load , DB
from .cal import CALENDAR , Dates
from . import bases as Base
