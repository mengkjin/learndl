# min_chars 因子说明

分钟线重构特征，按依赖分三层写出。列级清单见同目录 [`factors.csv`](factors.csv)（一行一个已实现列）。  
东方原表词典：仓库 `.context/references/dongfang_ms_l2_chars.md`。

命名：无 `inday_` 前缀。同日列用语义名（`ret_std` 而非 `rvol`）；滚动加窗口后缀（`_ma20` / `_std20` / `_cv20` / `_sum20` / `_max5`）；池化分钟矩用 `_pool_`。

| 旧名 | 新名 |
|---|---|
| `aret` | `ret_path` |
| `rvol` / `rskew` / `rkurt` | `ret_std` / `ret_skew` / `ret_kurt` |
| `vvol` / `vhhi` / `jump` | `vol_std` / `vol_hhi` / `ret_jump` |
| `amtca` / `bopcth1` | `amt_ca` / `bopct_h1` |
| `err_ret` / `maxdd` / `vardown` / `volcv` | `ret_topk_mean` / `ret_maxdd` / `ret_vardown` / `vol_cv` |
| `ret_ac` / `vol_ac` / `vol_ret1` / `vol_vwap` | `ret_autocorr` / `vol_autocorr` / `vol_retlag_corr` / `vol_vwap_corr` |
| `smart` / `stupid` | `smart_money` / `stupid_money` |
| `vol_end15` / `vol_st5` | `vol_end15_share` / `vol_open5_share` |
| `volpct_phigh` / `volpct_plow` / `volpct_devhigh` | `vol_highrank_share` / `vol_lowrank_share` / `vol_highdev_share` |
| `am_ret` / `pm_ret` / `incvol_mom` | `ret_am` / `ret_pm` / `incvol_ret` |
| `trend` / `vwap_hl` | `vwap_trend` / `vwap_hlvol` |
| `aret_{tag}` / `retm_{tag}` / `amtsh_{tag}` | `ret_path_{tag}` / `ret_mean_{tag}` / `amt_share_{tag}` |
| `{stem}_mean`（池化） | `{stem}_pool_mean`（避免与 `ret_std_ma20` 混淆） |

---

## 0. 分表落盘

| Updater | 序 | DB key | 依赖 |
|---|---|---|---|
| `MinCharsDailyUpdater` | 110 | `min_chars/min_chars` | 当日 `trade_ts/min` |
| `MinCharsRollUpdater` | 111 | `min_chars/min_chars_roll` | 最近 20 个 **min ∩ min_chars** 日（含当日） |
| `MinCharsTaggedUpdater` | 112 | `min_chars/min_chars_tag` | 当日 min ∩ 当日 roll |

路径：`data/DataBase/DB_min_chars/{key}/{yyyy}/{key}.{yyyymmdd}.feather`。数值列为 **float32**（`date`/`secid`/`n` 为 int64；超出 float32 范围写 NaN）。可按 updater 分开补全。

`daily_update` 只补 **机器日程 ∩ 最近 20 个交易日**（本机 testing 通常是 `20250416–20250422`）。2010 起的历史请用下面的 backfill 脚本，不要靠日更。

---

## 共用 bar

`_common.prepare_ret_bars`：`minute∈[0,239]`，`sess=minute//30`，`ret=(close-preclose)/preclose`（首根 preclose=open），`px`=vwap 否则 close。  
`buy_w`：涨=1 / 跌=0 / 平=0.5（不是 L2 主买）。  
`ret_path = (∏(1+ret)−1)×100`。峰度 Pearson（`fisher=False`），`n≥3` 才写偏度/峰度。  
OHLC/`volu` 不落盘。

---

## 1. Daily（同日）

| 组 | 列 |
|---|---|
| 量价 | `amt` `twap` `vwap` |
| 流向 | `bwap` `swap` `bamt` `samt` `ret_path` `bopct` `amt_ca` `bopct_h1` |
| 已实现 | `ret_std` `ret_skew` `ret_kurt` `vol_std` `vol_hhi` `ret_jump` |
| 波动 | `ret_topk_mean` `ret_maxdd` `ret_vardown` `vol_cv`；5 分钟重采 `ret_std5` `ret_skew5` `ret_kurt5` `ret_vardown5` `vol_cv5` |
| 相关 | `mkt_beta` `mkt_corr` `ret_autocorr` `vol_autocorr` `vol_retlag_corr` `vol_vwap_corr`（+ `{k}h`，lag 不跨半小时） |
| 流动性 | `smart_money` `stupid_money` `vol_end15_share` `vol_open5_share` `vol_highrank_share` `vol_lowrank_share` `vol_highdev_share` |
| 动量日值 | `ret_am` `ret_pm` `conf_persist` `high_time` `incvol_ret` `vwap_trend` `vwap_hlvol` |

半小时茎：`amt twap bamt samt ret_path ret_std ret_skew ret_kurt vol_std vol_hhi ret_jump` + 6 个相关。公式见 csv。

hf 里所有「先算每日再 20 日 mean」的日值都在本表（上表 + 相关/波动/流动列）。

---

## 2. Roll

两块写在同一张 `min_chars_roll`（不是两套重复）：

**pool**（20 日分钟拼在一起，≠ 日值的 mean）

`n`；`{ret,vol,amt}_{p01,p05,p50,p95,p99}`；`{ret,vol,amt}_pool_{mean,std,skew,kurt}`。

**trail**（读 20 日 daily，按 hf 的 trailing 聚合；同一定义只留一列）

| 后缀 | 含义 | 对应 hf |
|---|---|---|
| `_ma20` | 20 日 mean | 绝大多数 `inday_*` |
| `_std20` | 20 日 std | `inday_*_std`、`trend_std`、`conf_persist` |
| `_cv20` | 20 日 std/mean | `inday_vol_utd`（对日 `vol_std`） |
| `_sum20` | 20 日 sum | `inday_incvol_mom` |
| `_max5` | 5 日 max | `inday_maxdd` |

`vol_cv_ma20` 同时覆盖 `inday_vol_std_1min` 与 `inday_vol_coefvar`（同日都是 `vol_cv`）。  
`ret_pool_std` ≠ `ret_std_ma20`：前者是 20×240 根分钟混在一起的 std，后者是 20 个日 `ret_std` 的平均。

未落入 trail 的 hf（需要 TRADE / 截面残差 / 拼 bar）：`inday_amap_orig` 的终值、`regain_conf_persist` 的 rank 残差、`vol_high_std`、`mom_high_volcv`、`mom_high_pstd`、`vwap_diff_hlvol` 的 20 日拼 bar 版（有 `vwap_hlvol_ma20` 作为日值平均近似）。

---

## 3. Tag

同日分钟用 roll 分位打标：`rethi99/95`、`retlo01/05`、`amthi99/95`、`amtlo01/05`。  
每标签：`ret_path_{tag}`、`ret_mean_{tag}`、`amt_share_{tag}`。

---

## 4. 次序

```
min ──► Daily(110) ──► min_chars ──┐
min ──►                            ├─► Roll(111) ──► min_chars_roll ──► Tag(112)
```

Roll 现在依赖 daily（trail）也依赖 min（pool）。Tag 只依赖 roll 阈值 + 当日 min。

历史补全（只写缺失日，日期跟随 `trade_ts/min`）：

```bash
uv run scripts/2_data/4_backfill_min_chars.py --start=20140101
# 重算区间：加 --overwrite=True
# 只补某一层：--do_daily=False --do_roll=True --do_tag=False
# 只覆盖重算 tag（float32 overflow）：
uv run scripts/2_data/5_recalc_min_chars_tag.py --start=20100101
```

---

## 5. PrePro 选列（`select`）

`factors.csv` 的 `select` 由 `_catalog.is_selected` 生成（`write_csv()`），不要手改。规则：

- 不选 5 分钟重采茎（`ret_std5` 等）及其 trail（`ret_std5_ma20` 等）
- 不选全日简单量价/路径收益（`amt` `twap` `vwap` `bwap` `swap` `bamt` `samt` `ret_path`）；**1h/8h session 同茎保留**
- 不选 2h–7h session
- 不选池化样本数 `n`

入模：全部 `select=1` 列打成一个特征块，两个 PrePro 只差变换：

- `PrePro_minc`：原始值按日截面 z-score，非有限/缺失归 0（对齐 `dfl2cs`）
- `PrePro_mincr`：先按 secid 做 250 日 rolling pct_rank（`min_samples=90`），再按日截面 z-score，缺失归 0

东方 `dfl2`/`dfl2cs` 已 `ENABLED=False`。
