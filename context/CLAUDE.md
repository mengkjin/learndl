# learndl Context Library

This folder is the living documentation vault for the `learndl` project. It is also a valid Obsidian vault — open the `context/` directory as a vault to get graph view and WikiLink navigation.

## Module Docs

| Doc | Covers |
|-----|--------|
| [[overview]] | Full architecture diagram, layer table, design patterns |
| [[modules/project_infra]] | `src/proj/` + `src/func/`: MACHINE, PATH, CALENDAR, DB I/O, singletons, tensor utilities |
| [[modules/data_pipeline]] | `src/data/`: DataBlock, DateDataAccess singletons, PreProcessor, DataAPI |
| [[modules/factor_engine]] | `src/factor/`: FactorCalculator, FactorAPI, factor test framework |
| [[modules/nn_models]] | `src/res/algo/nn/`: ModelConfig, DataModule, training lifecycle, SWA, ModelAPI |
| [[modules/boost_models]] | `src/res/algo/boost/`: GeneralBoostModel, OptunaBoostModel, BoostInput/Output |
| [[modules/gp_strategy]] | `src/res/gp/`: GeneticProgramming, primitives, gpParameters, EliteGroup |
| [[modules/trading]] | `src/res/trading/`: TradingPort, PortfolioBuilder, TradingAPI, SummaryAPI |
| [[modules/interactive_app]] | Interactive dashboards and visualization tools |

## Workflow Docs

| Doc | Covers |
|-----|--------|
| [[workflows/daily_update]] | Daily data refresh pipeline |
| [[workflows/training_run]] | Model training end-to-end |
| [[workflows/strategy_dev]] | New alpha signal development checklist |

## Maintenance

Selective updates are driven by `_index.yaml` (module → source directory ownership map).

When code in a source directory changes, only the owning doc needs updating:
```bash
# Future: python scripts/update_context.py --changed src/data/
```

Manual update: edit the relevant `modules/*.md` file directly. Commit alongside the code change.

**Do not regenerate the entire library on every change** — only touch the doc whose `sources:` entries overlap the changed files.
