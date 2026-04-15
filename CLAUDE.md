# learndl — Claude Code Root Context

**Project:** Chinese A-share equity quantitative research system.
Covers data ingestion, factor engineering, NN/boost/GP models, and portfolio trading.

## Quick Reference

| Layer | Source Path | Entry API |
|-------|-------------|-----------|
| Infrastructure | `src/proj/`, `src/func/` | `MACHINE`, `PATH`, `CALENDAR`, `DataAPI` |
| Data Pipeline | `src/data/` | `DATAVENDOR` (facade), `DateDataAccess` singletons, `ModuleData` (model inputs) |
| Factor Engine | `src/factor/` | `FactorAPI`, `FactorCalculator` |
| NN Models | `src/res/algo/nn/` | `ModelAPI` |
| Boost Models | `src/res/algo/boost/` | `ModelAPI` (same interface) |
| GP Strategy | `src/res/gp/` | `GeneticProgramming` |
| Trading | `src/res/trading/`, `src/api/trading.py` | `TradingAPI`, `SummaryAPI` |

## Context Library
Full module documentation lives in `context/`. See `context/CLAUDE.md` for the index.

When answering questions or writing code for this project, read the relevant doc in `context/modules/` first.

## Key Rules
1. **Never hardcode paths** — always use `PATH.xxx` from `src/proj/path.py`
2. **Use the API layer** — `DataAPI`, `FactorAPI`, `ModelAPI`, `TradingAPI` are the stable interfaces; don't reach into internals directly
3. **Config-driven** — model architectures, factor params, portfolio specs all live in `configs/`; don't hardcode hyperparameters in code
4. **Numbered scripts** — pipeline scripts in `scripts/` are numbered (e.g. `1_data/`, `2_factor/`, `3_train/`); run in order
