# Genetic Programming Strategy
**Purpose:** Evolve symbolic factor expressions using DEAP-based genetic programming. Discovers non-linear combinations of base signals optimized for IC/rank-IC fitness.
**Key source paths:** `src/res/gp/`
**Depends on:** [[data_pipeline]], [[factor_engine]], [[project_infra]]

---

## Core Concepts

GP generates symbolic expression trees where:
- **Leaves (terminals):** base feature signals (price, volume, turnover, etc.)
- **Internal nodes (primitives):** operators (arithmetic, time-series, cross-sectional)
- **Fitness:** IC or rank-IC of the evaluated expression against forward returns

DEAP handles population management, crossover, and mutation. The main loop iterates over `n_iter` outer loops, each running `n_gen` generations of evolution.

---

## Entry Point

### `src/res/gp/main.py` — `main()`
Top-level function. Instantiates `GeneticProgramming` and calls the lifecycle in order:
1. `gp.load_data()` — load base feature data
2. `gp.preparation()` — set up DEAP toolbox, primitives, population
3. `gp.population()` — generate initial population
4. `gp.evolution()` — run generational evolution (`n_iter × n_gen`)
5. `gp.selection()` — extract elite expressions to hall of fame

---

## `GeneticProgramming` Class

Central orchestrator.

**Key attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `param` | `gpParameters` | All hyperparameters |
| `status` | dict | Runtime state (current iter/gen, timing) |
| `memory` | `EliteGroup` | Hall-of-fame expressions surviving across iters |
| `logger` | Logger | Per-session log file (from [[project_infra]]) |
| `input` | `gpInput` | Loaded base feature tensors |
| `timer` | `Duration` | Elapsed time tracking (from [[project_infra]]) |
| `evaluator` | `gpEvaluator` | Fitness evaluation engine |

**Key methods:**
| Method | Purpose |
|--------|---------|
| `load_data()` | Load feature tensors via `DataAPI` |
| `preparation()` | Register primitives in DEAP `PrimitiveSet`, configure toolbox |
| `population()` | Generate initial population via `gpGenerator` |
| `evolution()` | Outer loop: `n_iter × n_gen` DEAP `eaMuPlusLambda` |
| `selection()` | Pareto / NSGA-II selection into `EliteGroup` |

---

## `gpParameters` — Configuration

All GP hyperparameters. Loaded from `configs/algo/gp/params.yaml`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pop_num` | 3000 | Population size |
| `hof_num` | 500 | Hall-of-fame size (elites retained) |
| `n_iter` | 5 | Outer evolution iterations |
| `n_gen` | 3 | Generations per outer iteration |
| `max_depth` | 5 | Maximum expression tree depth |
| `cxpb` | 0.35 | Crossover probability |
| `mutpb` | 0.25 | Mutation probability |
| `select_offspring` | `'nsga2'` | Offspring selection: `'nsga2'` or `'tournament'` |
| `ir_floor` | 3.0 | Minimum IC ratio for an expression to survive |
| `corr_cap` | 0.6 | Max allowed pairwise correlation between retained elites |

---

## Primitive Function Set (79 registered)

### Unary operators
```
abs, sign, log, sqrt, square, inv,
ts_mean, ts_std, ts_skew, ts_kurt,
ts_rank, ts_zscore, ts_delta, ts_delay,
ts_max, ts_min, ts_argmax, ts_argmin,
cs_rank, cs_zscore, cs_demean,
sigmoid, tanh, relu
```

### Binary operators
```
add, sub, mul, div, pow,
ts_corr, ts_beta, ts_cov,
ts_decay_linear, ts_decay_exp,
cs_corr, greater, less, if_else
```

Many `ts_*` functions accept an implicit or explicit window length parameter. The window is controlled by `gpGenerator`'s `window_type` and `halflife` parameters at population-generation time.

---

## Input Data — `gpInput`

### `InputElement` (dataclass)
```python
@dataclass
class InputElement:
    name: str           # feature name (e.g., 'close_price')
    data: torch.Tensor  # shape: (N_date, N_secid)
    weight: float       # sampling weight for this feature
```

### `INPUT_MAPPING` — short name → data source
| Key | Source | Description |
|-----|--------|-------------|
| `cp` | TRADE | Close price |
| `turn` | TRADE | Turnover rate |
| `vol` | TRADE | Volume |
| `vwap` | TRADE | VWAP |
| `ret` | TRADE | Daily return |
| `ret_inday` | TRADE | Intraday return |
| `amount` | TRADE | Trading amount |
| `mktcap` | TRADE | Market capitalization |

Additional features can be added to `INPUT_MAPPING` without changing the GP framework.

---

## Fitness Evaluation — `gpEvaluator`

| Method | Purpose |
|--------|---------|
| `to_value(individual)` | Evaluate expression tree to `FactorValue` tensor `(N_date, N_secid)` |
| `assess(factor_value)` | Compute multi-objective fitness: `(ic_mean, ic_ir, ic_stability)` |

Fitness is multi-objective for NSGA-II selection. The `ir_floor` parameter gates which expressions enter the elite pool.

### `FactorValue` (dataclass)
```python
@dataclass
class FactorValue:
    value:   torch.Tensor   # (N_date, N_secid) evaluated factor
    name:    str            # serialized expression string
    fitness: tuple          # (ic_mean, ic_ir, ic_stability)
```

---

## Population Management

### `Population` / `BaseIndividual`
DEAP `Individual` subclass representing one expression tree.

| Method | Purpose |
|--------|---------|
| `purify()` | Simplify / canonicalize the expression tree |
| `revert()` | Undo last mutation (used in rejection sampling) |
| `prune()` | Trim branches exceeding `max_depth` |

### `EliteGroup`
Hall-of-fame storage with diversity enforcement:
- Keeps top `hof_num` individuals by fitness
- Rejects entries with pairwise correlation > `corr_cap` with existing elites
- Persists across outer iterations (`n_iter` loop)

### `gpGenerator`
Controls how new individuals are generated:

| Parameter | Description |
|-----------|-------------|
| `weight_scheme` | How to sample primitives: `'uniform'`, `'depth_weighted'` |
| `window_type` | How rolling windows are chosen: `'fixed'`, `'sampled'` |
| `halflife` | Halflife for exponential window weighting |

---

## Configuration — `configs/algo/gp/params.yaml`

```yaml
pop_num: 3000
hof_num: 500
n_iter: 5
n_gen: 3
max_depth: 5
cxpb: 0.35
mutpb: 0.25
select_offspring: nsga2
ir_floor: 3.0
corr_cap: 0.6

generator:
  weight_scheme: uniform
  window_type: sampled
  halflife: 10

input:
  features: [cp, turn, vol, vwap, ret, amount]
  lookback: 60

fitness:
  target: ic_ir
  min_periods: 20
```

---

## Data Flow

```
configs/algo/gp/params.yaml
        ↓
GeneticProgramming.load_data()   → gpInput (feature tensors from DataAPI)
        ↓
GeneticProgramming.preparation() → DEAP PrimitiveSet + toolbox
        ↓
GeneticProgramming.population()  → initial Population (gpGenerator)
        ↓ [n_iter outer loop]
GeneticProgramming.evolution()   → eaMuPlusLambda + gpEvaluator.assess()
        ↓
GeneticProgramming.selection()   → EliteGroup (hof_num surviving expressions)
        ↓
Serialize expressions → factor definitions or export to factor engine
```

---

## Common Patterns / Gotchas

- Full GP runs are long (hours) — reduce `n_iter` and `pop_num` for exploration
- `corr_cap` prevents the elite pool from converging to near-identical expressions; lower it to force more diversity
- `ir_floor` is a hard gate — expressions below this IC ratio are discarded regardless of other fitness components
- Expressions found by GP can be translated into explicit `FactorCalculator` subclasses for production use in [[factor_engine]]
- Always call `purify()` before serializing — raw evolved trees often contain redundant sub-expressions
- Parallelized fitness evaluation via multiprocessing; `Device` from [[project_infra]] controls CPU/GPU allocation
