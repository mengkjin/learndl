# learndl — Agent Handoff

**Updated:** 2026-06-25  
**Focus:** Factor update failures from pivot MultiIndex column mismatch

## Root cause (confirmed)

`Buffer dtype mismatch` when mixing flat `secid` Index columns with MultiIndex columns from `pivot_table(['field'], ...)`.

Regression introduced by **2026-06-16** `1f78cef2`: `get_quotes` forced `field = ensure_name_list(field)` before pivot, turning `'close'` → `['close']`.

**Not caused by** pandas 2.3.x upgrade.

## Data-layer fix (done)

- `access.py`: added `pivot_narrow_table`; `get_specific_data` pivots via it with **original** `field` type.
- `trade_data.py`: `get_quotes` loads narrow table (`pivot=False`), adj_price, then `pivot_narrow_table`.
- `exposure.py`: `get_risks` delegates pivot to `get_specific_data` (removed local `ensure_name_list` + manual pivot).

**Design choice:** `pivot_narrow_table` keeps list semantics — single-element list `['close']` still yields MultiIndex; callers should pass string for single-field wide pivot.

## Factor-layer fixes (keep)

| Area | File | Status |
|------|------|--------|
| Turn / liquidity mask | `access.py` `mask_min_finite` | Keep — wide-table detection |
| Turn stored values | `factor_calc.py` | Keep — `eval_factor_series`, column prune, Series index flatten |
| Hidden cache | `predictor.py` `hidden_block` | Keep — malformed feather guard |

## Removed workarounds (done)

- `momentum_umr_raw.py` / `momentum_umr_new.py`: removed `risks.columns` MultiIndex flatten (redundant after data-layer fix).
- `behavior_slice.py`: removed debug `print` in `ampl_slicecp1m`.

## Next steps

- [ ] Re-run factor update batch (UMR, slice, turn, pooling)
- [ ] Optional: delete corrupted `snapshot/hidden_values/*.feather` caches

## Key paths

- Pivot: `src/data/loader/access.py` `pivot_narrow_table`, `get_specific_data`
- Quotes: `src/data/loader/trade_data.py` `get_quotes`
- Risks: `src/data/loader/exposure.py` `get_risks`
