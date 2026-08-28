# 东方证券分钟 / L2 特征词典

`sellside/dongfang.ms_chars` ∪ `sellside/dongfang.l2_chars` 的**已命名**字段。  
由当日分钟线与逐笔/L2 聚合为日频截面；下载入口见 `src/data/download/sellside/from_sql.py`。

样本核对：`ms_chars` 20250422（5392 × 120），`l2_chars` 20250620 / 20250422（约 5140 × 102）。

## 范围

| 表 | DB key | 粒度 | 列数 | 本文 |
|---|---|---|---|---|
| 分钟特征 | `dongfang.ms_chars` | 日频，内含 8 个半小时 | 120 | 全部纳入 |
| L2 特征 | `dongfang.l2_chars` | 日频，内含 8 个半小时 + 大单分位 | 102 | 去掉 `l2c0`–`l2c4` |
| 并集（去重名） | — | — | **214** | `date` / `secid` / `twap` 为两表共有 |

**剔除（未公开命名）：** `l2c0`, `l2c1`, `l2c2`, `l2c3`, `l2c4`。  
这五列与 `dongfang.hfq_chars.l2c0`–`l2c4` 逐点相同，源库无中文名，不写入本词典。

---

## 命名语法

两表共用同一套构词。`{k}` 一律表示半小时序号 1–8。

| 构件 | 含义 |
|---|---|
| `opri` / `hpri` / `lpri` / `cpri` | 开 / 高 / 低 / 收 |
| `volu` / `amt` | 成交量（股） / 成交额（元） |
| `twap` / `vwap` | 时间加权均价 / 成交量加权均价 |
| `bwap` / `swap` | 主动买均价 / 主动卖均价（`swap` 不是利率互换） |
| `bamt` / `samt` | 主动买成交额 / 主动卖成交额 |
| `aret` | 分段收益率（%），价格路径，不是买卖价差 |
| `bopct` | 买方占比（0–100） |
| `rvol` / `rskew` / `rkurt` | 已实现波动 / 偏度 / 峰度（分钟收益） |
| `vvol` / `vhhi` / `jump` | 成交量波动 / 成交量时间集中度 / 已实现跳跃 |
| `{k}h` | 第 `k` 个半小时，见下表 |
| `0h` | 开盘集合竞价（仅 `volu0h`） |
| `_p1` / `_p5` | 逐笔金额分位：最大约 1%（超大单）/ 5%（大单）；`p1 ⊂ p5 ⊂ 全部` |
| `oa` / `ca` | 开盘集合竞价 / 收盘集合竞价 |
| `h1` | 与 `1h` 同义（仅出现在 `bopcth1_*`） |

### 半小时网格

A 股连续竞价 4 小时 = 8 × 30 分钟。成交呈 U 型（开盘最肥，尾盘次回）。

| 后缀 | 时段 | 成交量占全日中位数（`ms_chars` 20250422） |
|---|---|---|
| `0h` | 09:15–09:25 开盘集合竞价 | ~0.4% |
| `1h` | 09:30–10:00 | ~30% |
| `2h` | 10:00–10:30 | ~12% |
| `3h` | 10:30–11:00 | ~9% |
| `4h` | 11:00–11:30 | ~6% |
| `5h` | 13:00–13:30 | ~7% |
| `6h` | 13:30–14:00 | ~7% |
| `7h` | 14:00–14:30 | ~7% |
| `8h` | 14:30–15:00（含 14:57–15:00 收盘竞价） | ~13% |

`{k}h` 与 `{k+1}h` 的收/开价衔接（含午休 4h→5h），价差中位数为 0 或 1 个 tick。

---

## 表间恒等式

在共同交易日上（20250422）核过：

| 关系 | 结果 |
|---|---|
| `ms.volu` = `volu0h + Σ volu{k}h` | 完全相等 |
| `ms.amt` ≈ `Σ amt{k}h` | 比例中位 0.996（差额即开盘竞价额） |
| `ms.cpri` = `cpri8h` | 完全相等 |
| `ms.hpri` / `lpri` = 各半小时高/低的 max / min | 相关 ≈ 1 |
| `ms.twap` ≈ `l2.twap` | 相关 ≈ 1，中位绝对差 < 0.001 元 |
| `l2.amtoa` = `ms.volu0h * ms.opri` | 相关 = 1 |
| `l2.bamt{k}h + l2.samt{k}h` = `ms.amt{k}h` | 相关 ≈ 1，比例 = 1 |
| `l2.bamt + l2.samt` ≈ `ms.amt` | 相关 0.9999，比例中位 0.98（连续竞价 vs 含竞价） |
| `l2.bamt` = `Σ bamt{k}h`，`l2.aret` = `Σ aret{k}h` | 相关 ≈ 1 |
| `l2.aret` ≈ `(cpri/opri − 1)×100` | 相关 0.997 |
| `l2.aret_p5` = `dongfang.order_flow.lnret_p05` | 逐点相等 → `_p5` 即 5% 大单口径 |
| `l2.bamt_p5` ≈ Tushare `buy_lg + buy_elg`（万元×10000） | 相关 0.94，比例 ≈ 1.02 |

`_p1` 是更严的嵌套大单桶（主动买额约占全日 20%），**不是** `order_flow.e01`。

同源备查（不在并集内）：`hfq_chars` 的 `rvol/rskew/rkurt/vhhi/rjump` 与 `ms_chars` 同名（`jump`）列逐点相同；`ovpct ≈ volu1h/volu×100`，`cvpct = volu8h/volu×100`。`hfq_chars.vvol` 已标准化，与 `ms_chars.vvol`（股数）不是同一列。

---

## 字段词典

来源：`M` = `ms_chars`，`L` = `l2_chars`，`M+L` = 两表都有。  
`{k}h` 展开为 8 列，不逐列重复解释。

### 索引

| 列 | 来源 | 含义 |
|---|---|---|
| `date` | M+L | 交易日 `yyyymmdd` |
| `secid` | M+L | 证券代码整数（`1` = 000001） |

### 分钟行情 OHLC / 量额 / TWAP

| 列 | 来源 | 单位 | 含义 |
|---|---|---|---|
| `opri` | M | 元 | 开盘价（集合竞价成交价；与 `opri1h` 差 0–1 tick） |
| `hpri` | M | 元 | 全日最高价 |
| `lpri` | M | 元 | 全日最低价 |
| `cpri` | M | 元 | 收盘价（= `cpri8h`） |
| `volu` | M | 股 | 全日成交量（含开盘竞价） |
| `amt` | M | 元 | 全日成交额 |
| `twap` | M+L | 元 | 全日时间加权均价 |
| `volu0h` | M | 股 | 开盘集合竞价成交量 |
| `opri{k}h` | M | 元 | 第 `k` 个半小时开盘价 |
| `hpri{k}h` | M | 元 | 第 `k` 个半小时最高价 |
| `lpri{k}h` | M | 元 | 第 `k` 个半小时最低价 |
| `cpri{k}h` | M | 元 | 第 `k` 个半小时收盘价 |
| `volu{k}h` | M | 股 | 第 `k` 个半小时成交量 |
| `amt{k}h` | M | 元 | 第 `k` 个半小时成交额（= 该时段 `bamt+samt`） |
| `twap{k}h` | M | 元 | 第 `k` 个半小时 TWAP |

### 已实现统计（分钟收益 / 分钟量）

由更细的分钟收益与分钟成交量计算，再聚合成全日或半小时。`rvol` 与日振幅 `(hpri/lpri−1)` 相关约 0.80；`jump` 与最大半小时收益相关约 0.72。`vhhi` 均匀分布时接近 1，越大表示成交越堆在少数分钟（max 可达数十，故不是 `[0,1]` 的原始 HHI）。

| 列 | 来源 | 含义 |
|---|---|---|
| `rvol` | M | 已实现波动率（分钟收益） |
| `rskew` | M | 已实现偏度 |
| `rkurt` | M | 已实现峰度（全日均值约 7–8，厚尾） |
| `vvol` | M | 分钟成交量波动，量纲为股；与 8 个半小时 `volu` 的标准差相关 0.96 |
| `vhhi` | M | 成交量时间集中度（Herfindahl 类） |
| `jump` | M | 已实现跳跃（有符号，% 量级）；`hfq_chars` 中名为 `rjump` |
| `rvol{k}h` | M | 第 `k` 个半小时内的已实现波动 |
| `rskew{k}h` | M | 第 `k` 个半小时已实现偏度（样本不足时可为空） |
| `rkurt{k}h` | M | 第 `k` 个半小时已实现峰度 |
| `vvol{k}h` | M | 第 `k` 个半小时内的成交量波动 |
| `vhhi{k}h` | M | 第 `k` 个半小时内的成交量时间集中度 |
| `jump{k}h` | M | 第 `k` 个半小时已实现跳跃 |

### 成交均价（L2 主动买卖）

全日均价与收盘价相关 > 0.9999，是同一价格水平的不同加权。`_p1/_p5` 只对落入该大单桶的逐笔做加权。

| 列 | 来源 | 单位 | 含义 |
|---|---|---|---|
| `vwap` | L | 元 | 成交量加权均价 |
| `bwap` | L | 元 | 主动买成交均价 |
| `swap` | L | 元 | 主动卖成交均价 |
| `bwap_p1` / `bwap_p5` | L | 元 | 超大单 / 大单主动买均价 |
| `swap_p1` / `swap_p5` | L | 元 | 超大单 / 大单主动卖均价 |

### 主动成交额与竞价额

主动方向来自逐笔主买/主卖（外盘/内盘）。金额单位为元。

| 列 | 来源 | 含义 |
|---|---|---|
| `bamt` / `samt` | L | 全日主动买 / 主动卖成交额 |
| `bamt_p1` / `samt_p1` | L | 超大单主动买 / 卖额（约占全日 20% / 24%） |
| `bamt_p5` / `samt_p5` | L | 大单主动买 / 卖额（约占 45% / 49%） |
| `bamt{k}h` / `samt{k}h` | L | 第 `k` 个半小时主动买 / 卖额 |
| `bamt{k}h_p1` / `samt{k}h_p1` | L | 第 `k` 个半小时超大单主动买 / 卖额 |
| `bamt{k}h_p5` / `samt{k}h_p5` | L | 第 `k` 个半小时大单主动买 / 卖额 |
| `amtoa` | L | 开盘集合竞价成交额（= `volu0h × opri`） |
| `amtca` | L | 收盘集合竞价成交额（约占全日 1%，约为 `amt8h` 的 8%） |

### 收益率

`aret` 是开盘→收盘的价格路径收益（%），**不含隔夜跳空**。大单收益 `aret_p5` 波动显著大于全日 `aret`（截面标准差约 12 vs 2），反映大单成交路径更噪。

| 列 | 来源 | 单位 | 含义 |
|---|---|---|---|
| `aret` | L | % | 连续竞价全日收益 ≈ `(cpri/opri−1)×100` |
| `aret_p1` / `aret_p5` | L | % | 超大单 / 大单成交路径收益 |
| `aret{k}h` | L | % | 第 `k` 个半小时收益 ≈ `(cpri{k}h/opri{k}h−1)×100` |
| `aret{k}h_p1` / `aret{k}h_p5` | L | % | 第 `k` 个半小时超大单 / 大单路径收益 |

### 大单买方占比

接近但不等同于 `bamt_p*/(bamt_p*+samt_p*)`（相关约 0.54–0.59），更像按笔数或委托侧统计。

| 列 | 来源 | 单位 | 含义 |
|---|---|---|---|
| `bopct_p1` / `bopct_p5` | L | 0–100 | 超大单 / 大单中的买方占比 |
| `bopcth1_p1` / `bopcth1_p5` | L | 0–100 | 第一半小时（09:30–10:00）的同上口径 |

---

## 完整列名（展开 `{k}h` 后）

### `dongfang.ms_chars`（120）

```
date secid
opri hpri lpri cpri volu amt twap volu0h
opri1h hpri1h lpri1h cpri1h volu1h amt1h twap1h
opri2h hpri2h lpri2h cpri2h volu2h amt2h twap2h
opri3h hpri3h lpri3h cpri3h volu3h amt3h twap3h
opri4h hpri4h lpri4h cpri4h volu4h amt4h twap4h
opri5h hpri5h lpri5h cpri5h volu5h amt5h twap5h
opri6h hpri6h lpri6h cpri6h volu6h amt6h twap6h
opri7h hpri7h lpri7h cpri7h volu7h amt7h twap7h
opri8h hpri8h lpri8h cpri8h volu8h amt8h twap8h
rvol rskew rkurt vvol vhhi jump
rvol1h rskew1h rkurt1h vvol1h vhhi1h jump1h
rvol2h rskew2h rkurt2h vvol2h vhhi2h jump2h
rvol3h rskew3h rkurt3h vvol3h vhhi3h jump3h
rvol4h rskew4h rkurt4h vvol4h vhhi4h jump4h
rvol5h rskew5h rkurt5h vvol5h vhhi5h jump5h
rvol6h rskew6h rkurt6h vvol6h vhhi6h jump6h
rvol7h rskew7h rkurt7h vvol7h vhhi7h jump7h
rvol8h rskew8h rkurt8h vvol8h vhhi8h jump8h
```

### `dongfang.l2_chars` 已命名（97，已剔除 `l2c0`–`l2c4`）

```
date secid
twap vwap bwap bwap_p5 bwap_p1 swap swap_p5 swap_p1
amtoa bamt samt bamt_p1 samt_p1 bamt_p5 samt_p5
aret aret_p1 aret_p5
bamt1h samt1h bamt1h_p1 samt1h_p1 bamt1h_p5 samt1h_p5 aret1h aret1h_p1 aret1h_p5
bamt2h samt2h bamt2h_p1 samt2h_p1 bamt2h_p5 samt2h_p5 aret2h aret2h_p1 aret2h_p5
bamt3h samt3h bamt3h_p1 samt3h_p1 bamt3h_p5 samt3h_p5 aret3h aret3h_p1 aret3h_p5
bamt4h samt4h bamt4h_p1 samt4h_p1 bamt4h_p5 samt4h_p5 aret4h aret4h_p1 aret4h_p5
bamt5h samt5h bamt5h_p1 samt5h_p1 bamt5h_p5 samt5h_p5 aret5h aret5h_p1 aret5h_p5
bamt6h samt6h bamt6h_p1 samt6h_p1 bamt6h_p5 samt6h_p5 aret6h aret6h_p1 aret6h_p5
bamt7h samt7h bamt7h_p1 samt7h_p1 bamt7h_p5 samt7h_p5 aret7h aret7h_p1 aret7h_p5
bamt8h samt8h bamt8h_p1 samt8h_p1 bamt8h_p5 samt8h_p5 aret8h aret8h_p1 aret8h_p5
amtca bopct_p1 bopct_p5 bopcth1_p1 bopcth1_p5
```

### 未纳入

```
l2c0 l2c1 l2c2 l2c3 l2c4
```

---

## 使用提示

- 读分钟路径用 `ms_chars`（OHLC、TWAP、已实现矩）；读订单流用 `l2_chars`（主买主卖、大单、买方占比）。
- 两表用 `date+secid` 对齐；量额用 `amt{k}h = bamt{k}h+samt{k}h` 衔接，竞价用 `amtoa = volu0h×opri`。
- 入模前按项目现有 `PrePro_dfl2` / `PrePro_dfl2cs` 做时序分位或截面 z-score；金额类需先取对数或除以当日 `amt`。
- 加载：`DB.loads_pl('sellside', 'dongfang.ms_chars', ...)` / `dongfang.l2_chars`。
