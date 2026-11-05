# Unified portfolio generator backend

All four FMP entrypoints (top/T50, screen, revscreen, reinforce) use
`run_generator` in this module. Selection, evolution and accounting kernels
share one production implementation. Differential tests compare it with the
original builder.

Default training uses TrainingAnalysisSnapshot for available inputs and the
original vendor for misses. See ../../model/util/data/training_analysis_snapshot.md for storage,
fallback, weekly maintenance and manual rebuild instructions.

`auto` selects the array implementation for supported default creator parameters
and falls back to the original builder otherwise. `array` requires supported
inputs; `legacy` selects the original builder explicitly. The older
`t50_backend` option is retained as a configuration alias.

The monthly PreparedMarket and PreparedUniverse modules remain compatibility
readers. They are not the default preparation pipeline.
