# Unified portfolio generator backend

All four FMP entrypoints (top/T50, screen, revscreen, reinforce) use
`run_generator` in this module. Selection, evolution and accounting kernels
share one implementation. The internal `run_t50` function retains the
historical contract used by differential tests and benchmark diagnostics.

Default training uses TrainingAnalysisSnapshot for available inputs and the
original vendor for misses. See ../training_analysis_snapshot.md for storage,
fallback, weekly maintenance and manual rebuild instructions.

`auto` selects the array implementation for supported default creator parameters
and falls back to the original builder otherwise. `array` requires supported
inputs; `legacy` selects the original builder explicitly. The older
`t50_backend` option is retained as a configuration alias.

The monthly PreparedMarket and PreparedUniverse modules remain compatibility
readers and benchmark references. They are not the default preparation pipeline.
Historical benchmark scripts intentionally retain their original input contracts.
