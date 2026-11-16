"""Dividend economic zeros with explicit, persisted source provenance."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["dividend_values"]


def dividend_values(source, universe):
    """Only stock-level missingness in a structurally valid daily source is zero."""
    if source.empty or not {"secid", "dv_ttm"} <= set(source):
        raise ValueError("dtop: daily valuation source missing or malformed")
    if source.secid.isna().any() or source.secid.duplicated().any():
        raise ValueError("dtop: invalid/duplicate source stock identifiers")
    raw = pd.to_numeric(source.set_index("secid").dv_ttm, errors="raise").reindex(
        universe
    )
    if np.isinf(raw).any():
        raise ValueError("dtop: infinite dividend source values")
    observed = raw.notna()
    return pd.DataFrame(
        {
            "dtop": raw.fillna(0) / 100,
            "dtop__source_observed": observed.astype(float),
            "dtop__source_filled": (~observed).astype(float),
        }
    ).rename_axis("secid")
