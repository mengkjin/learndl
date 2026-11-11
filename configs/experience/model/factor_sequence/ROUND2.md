# Round 2: residual models

The ordinary idle worklist runs factorseq_v2_snapshot, factorseq_v2_temporal,
then factorseq_v2_contextual. Each uses factorseq_residual in a separate model root.
Old models and data are retained. All use seed 17, at most 30 epochs and validation
peak patience 5. DetailedAlphaAnalysis explicitly runs factor and t50 tasks.

The source suite remains 20261031_factor_sequence_verify_01. Its test dates and
immutable dataset are reused; training dates are regenerated with 252 validation
days and up to 1260 training days after 239 warmup days. Early history expands.
Label endpoints are purged. The resolved schedule stores actual train/valid dates,
pre-purge training length and full_window, and uses a 1800-day loading window.

Compact stock channels are value, valid, gap, never; performance retains daily IC,
spread, 20/60-day means of each and their valid masks. Selection is by manifest
names before alignment/tensor storage. NPZ decompression still temporarily loads
the source array; reduced resident channels are not a guarantee against host OOM.
Snapshot loads only stock with length 1; temporal loads stock length 60;
contextual additionally loads two shared streams with length 120.

The scalar output is a trainable linear factor score plus a zero-initialized
nonlinear residual. Contextual mode applies identity-initialized FiLM before the
stock GRU. The context has no direct prediction-head path.

results/factor_residual contains daily branch predictions, coverage diagnostics,
resources.json and available paired daily/annual comparisons. Standard detailed
analysis owns holdings, turnover and net returns. Paired comparisons require exact
stock/date/label agreement; incomplete peer results are not treated as a pass.

CPU regressions do not establish server RAM/CUDA acceptance. First inspect A's
initial fit and eventual detailed report/mail before accepting the workload.
No server process was started by local implementation. Resource measurements use
process lifetime RSS high-water and PyTorch allocated CUDA peak (not total GPU usage).
