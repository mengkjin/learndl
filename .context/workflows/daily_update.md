# Daily Update Workflow
**Purpose:** End-to-end daily data refresh: pull new market/fundamental data, recompute factors, update live portfolio tracking.

---

## Overview

Run after market close each trading day. Scripts must be run in order.

```
scripts/1_data/   → scripts/2_factor/   → scripts/4_trade/
(data refresh)      (factor refresh)      (portfolio update)
```

---

## Step-by-step

### 1. Data refresh (`scripts/1_data/`)
```bash
python scripts/1_data/0_update_trade.py       # market data (OHLCV, turnover, etc.)
python scripts/1_data/1_update_financial.py   # financial statements
python scripts/1_data/2_update_analyst.py     # analyst estimates
python scripts/1_data/3_update_index.py       # index constituents
```

Uses `DATAVENDOR.update_*()` from [[data_pipeline]].

### 2. Factor refresh (`scripts/2_factor/`)
```bash
python scripts/2_factor/0_update_factors.py   # recompute all registered factors for new date
```

Uses `FactorAPI.update()` from [[factor_engine]]. Only computes incremental dates (checks what's already saved).

### 3. Portfolio update (`scripts/4_trade/`)
```bash
python scripts/4_trade/0_update_tracking.py   # update live tracking portfolios
```

Uses `TradingAPI.update()` from [[trading]]. Generates today's trade suggestions.

---

## API Equivalents

```python
from src.api.data import DataAPI
from src.api.factor import FactorAPI
from src.api.trading import TradingAPI

DataAPI.update('trade', start=today, end=today)
FactorAPI.update('all', start=today, end=today)
TradingAPI.update()
```

---

## Common Issues

- If market data is missing for a date, factor update will skip that date silently — check logs in `PATH.log/`
- `DATAVENDOR` connections may require VPN or specific network; check `MACHINE.data_root` config
- Run `CALENDAR.is_trading_day(today)` before triggering — skip on non-trading days
