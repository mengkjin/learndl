# minc / mincr PrePro OOM 问题纪要

日期范围：2026-09-17 夜间 → 2026-09-18。  
范围：`PrePro_minc` / `PrePro_mincr` 的 fit reconstruct，以及为压峰值做的分年、`DataBlock.merge` 改写。  
当前状态：**强制 broadcast 后服务器仍 OOM，根因尚未确认。2026-09-18 再次核查：下文关于 meshgrid 的 131G 实际分配判断不成立，详见新增第 10 节。现已补 CLI 恢复、任务记录和诊断代码，尚未在服务器复现验收。**

---

## 1. 问题从哪来

`minc` 把三张 `min_chars` 表的入选列打成一块 fit `DataBlock`：截面 z-score，非有限值填 0。`mincr` 先做 250 日 rolling pct_rank，再同样的截面 z-score。

目标张量形状 `(N, T, 1, F)`：

| 量 | 约数 |
|---|---|
| N | A 股约 1800（2010）→ 5000+（近年） |
| T | `fit_start=20100101` 至今，约 16 年 × 242 交易日 ≈ 3800+ |
| F | 入选 152（daily 67 + roll 61 + tag 24） |
| dtype | 目标 float32 |

粗算最终 fit 张量：`5000 × 2900 × 152 × 4 ≈ 8–13 GiB`。这还只是**结果**，中间 Polars、稠密化、merge 临时量会再翻几倍。

---

## 2. 时间线

### 2.1 第一次现象：reconstruct 打一行就没了

服务器执行 minc fit preprocess，日志只有：

```text
RECONSTRUCT Preprocessed (fit) of [MINC] start...
```

之后无新输出、进程结束。判断：

- 该行之后要等 **整段 2010→今天** 的 `pre_process` 跑完才有下一条成功/合并日志。
- 中间无进度：`DB.loads_pl` 线程池读全历史 feather，再 `from_polars` 笛卡尔稠密化。
- 无 Python traceback、退出像被杀 → Linux OOM killer（常见 137 / SIGKILL）。

当时峰值估计（尚未分年）：

- 先读 daily 全列 Polars（含未入选列）约十几 GB
- 按 20 列切块 `from_polars` 再堆 list
- 152 维最终张量约 13 GB
- 加上 cartesian join 临时拷贝、线程池同时打开的文件 → 几十 GB

`mincr` 还要 250 日 rolling rank，更重。

### 2.2 第一轮修改：减冗余 + 按年切（`da3aca6a`）

针对「一次读全历史」。

**日更（不是这次 OOM 主因，顺手收了）：**

- `min_chars` roll 的 20 日 cache 只留 trail 源列，不再缓存 daily 全列
- 分钟 ret 面板改为 float32

**PrePro：**

- 只 `select` 入选列
- 同表用 `concat_feature`，z-score 落到 float32
- 增加 `load db_key` 进度日志

**按年分块（通用类参数）：**

| 属性 | 默认 | minc/mincr |
|---|---|---|
| `DateChunkYears` | `0` 关闭 | `1` |
| `ChunkFillNan` | `None` | `0.0` |

`load_with_extension` 把缺失区间拆成日历年，**每年单独完整 `pre_process`**（读盘 → 变换 → `from_polars`），再 merge。日志形如：

```text
minc chunk 20100101-20101231 (load from ...)
```

新文件：`src/data/preprocess/date_chunk.py`（`calendar_year_spans`）。

Rolling 窗口：外层 `span_load_start = CALENDAR.td(chunk_start, -CalculationWindow+1)`；`minc` 窗口 1，`mincr` 窗口 250。年界会向前多读。minc/mincr 的 `pre_process` **内部又减了一次** `CalculationWindow`，`mincr` 实际多读约 2×250 交易日，结果正确但浪费。

当时对「merge 还会不会爆」的判断：年内峰值压到「一年 + lookback」；年与年 `DataBlock.merge` 会 `torch.full` 一块 union 张量，同时拿着累计块和当前年，瞬时约 **2× 最终张量**（若最终 13 GB，merge 瞬间 ~26 GB）。以为这是剩余主风险。

### 2.3 第二轮现象：分年后 RSS 仍飙到 155G+

用户反馈：minc 执行时内存**快速累计**，到 **2021 年左右已 155G+**，撑爆。要求加变量体积的实时日志。

若只是 2× 最终张量，到不了 155G。继续查。

先排除：`feature_method='stack'` 在同名特征上会去重，**F 没有按年翻倍**。

当时怀疑 `DataBlock.merge` 里的 **4 维 `torch.meshgrid`**（后来确认以下仅是逻辑体积估算，不能据此认定实际分配）：

```text
tar_grid, src_grid = intersect_meshgrid(4 axes)
values[*tar_grid] = blk.values[*src_grid]
```

每个 mesh 是 4 个 int64 张量，形状 `(N, T, I, F)`；tar+src 共 8 个。

按 2021 年累计 `N≈5000, T≈2900, F=152`：

| 东西 | 体积 |
|---|---|
| 数据张量 float32 | ~8.2G |
| 8 个 int64 网格 | **~131G** |
| 再加 prev / year / 新 union | **落到 ~155G** |

本机用 catalog 验过：`mesh4d = 5000×2900×152×8×8 ≈ 131.4G`。

### 2.4 第二轮修改：双路径 merge + 内存日志（`cf75d09e`）

**`src/func/basic.py`**

- `intersect_copy`：通用拷贝
- `mesh`：原 `meshgrid`（小面板）
- `broadcast`：1 维下标 broadcast，额外 RAM 为 `O(N+T+I+F)`
- `auto`：`intersect_mesh_bytes` ≤ `INTERSECT_MESH_MAX_BYTES` 用 mesh，否则 broadcast
- 阈值最初 **256MiB**

**`DataBlock.merge`**

- 动手前对每个源块算 mesh 体积，整次 merge 选一种方法
- `torch.full` 使用源块 dtype（避免默认 float64）
- `align_secid_date`、boost `dataset` 合并也改走 `intersect_copy`

**临时日志 `MemTrace`（minc/mincr 为 True）**

- 新文件 `src/data/preprocess/mem_trace.py`
- 打 RSS / HWM、DataBlock/Polars nbytes、假想 `mesh4d`、`chunk-merge-mesh|broadcast`

**测速**

- `tests/test_datablock_merge_bench.py`
- 本机 16GB / 8 核 CPU：broadcast 全程约 **2× 快于 mesh**（mesh 先写整份 int64 下标）
- 本机跳过 >1.2G 的 mesh；minc 一年量级 broadcast 约 0.6s

### 2.5 强制 broadcast（`5e2762d2`）

用户把阈值改成：

```python
INTERSECT_MESH_MAX_BYTES : int = -1  # negative means use broadcast always
```

`auto` 里 `mesh_bytes <= -1` 恒为假，全部走 broadcast。

### 2.6 现在：还是 OOM

在上述修改（含强制 broadcast）之后，服务器测试 **仍然 OOM**。  
修正结论：**强制 broadcast 后仍 OOM；此前将 155G 归因于 meshgrid 的证据不足。需测量实际索引操作、读取、累计张量及保存阶段。**

---

## 3. 当前代码在做什么

minc reconstruct 路径（dump 为空）：

```
load_with_extension
  按日历年循环
    pre_process(load_start, chunk_end)     # 一年 + lookback
      对每个 db_key: loads_pl → select → 20 列一块 z-score
        DataBlock.from_polars              # secid×date 笛卡尔稠密化
        concat_feature 累加
      三表 DataBlock.merge (feature stack)
      slice_date + fillna(0)
    slice_date 切回日历年
    DataBlock.merge(prev_acc, year)        # date/secid union, feature stack 去重
  dump 整块 fit 张量
```

`MemTrace` 仍开着，日志里应能看到 `mem minc chunk-merge-broadcast ... | union=... | mesh4d=... | rss=...`。

---

## 4. 已改文件（按提交）

| 提交 | 内容 |
|---|---|
| `da3aca6a` minc and mincr fit block | `date_chunk.py`；`DateChunkYears` / `ChunkFillNan`；minc 只读入选列、float32；roll cache 瘦身 |
| `cf75d09e` fix data block merge | `intersect_copy`；merge/align/boost 改拷贝路径；`mem_trace.py`；`MemTrace`；bench 测试 |
| `5e2762d2` change INTERSECT_MESH_MAX_BYTES to -1 | 强制 broadcast |

涉及路径：

- `src/data/preprocess/core.py`
- `src/data/preprocess/date_chunk.py`
- `src/data/preprocess/processors.py`
- `src/data/preprocess/mem_trace.py`
- `src/data/util/classes/data_block.py`
- `src/func/basic.py`
- `src/res/algo/boost/util/dataset.py`
- `src/data/update/custom/min_chars/{_common.py,rolling.py}`
- `tests/test_datablock_merge_bench.py`
- `.context/modules/data_pipeline.md`（文档仍写 256MiB 阈值，与 `-1` 不一致）

---

## 5. 本机 bench 摘要（16GB Mac，非服务器）

年拼接（日期不交、secid 90% 重叠）、三表 stack、日期 overlay：broadcast / mesh ≈ 0.52–0.72，即 broadcast 更快。网格 >1.2G 的 mesh 在本机跳过。

这只说明 **CPU 上 mesh 又慢又吃索引**；不能说明 broadcast 在服务器全量 minc 上不会 OOM。

---

## 6. 去掉 mesh 之后，仍可能 OOM 的点

下列按嫌疑排序，**尚未用服务器 MemTrace 定论**。下一次应以日志里的 `rss` vs `union` vs 命名对象为准。

### 6.1 merge 仍然全量新分配（确定存在）

```python
values = torch.full((len(secid), len(date), len(inday), len(feature)), nan, dtype=...)
```

逐年累加时同时存在：累计块 + 当前年 + 新 union。峰值约 **2× 当时累计张量**，随年线性涨，但按 float32 到 2021 也只是十几～二十几 GB 量级，**单独解释不了 155G**，在去掉 mesh 之后却足以在内存偏紧或还有别的拷贝时把机器打满。

### 6.2 broadcast 赋值是否仍会物化大下标（未证实）

```python
dst[t0[:,None,None,None], t1[None,:,None,None], ...] = payload
```

设计意图是下标体积 `N+T+I+F`。若 PyTorch 在 CPU 高级索引里把 broadcast **展开成 `(N,T,I,F)` 的 gather 下标**，峰值会再次接近「数据大小的数倍」，只是不再是 8 份 int64 mesh。需要用 MemTrace 看 `chunk-merge-broadcast` 前后 rss 是否远大于 `union`。

### 6.3 `from_polars` 笛卡尔稠密化（每年都有）

`secid × date` left join 后再 `to_numpy` + `torch.from_numpy`。一年 `N≈5000, T≈250, F=20~67` 的稠密化与导出可能复制；`torch.from_numpy` 本身共享 NumPy 存储，不能直接算双份，reshape 是否复制应看真实布局；再叠加三张表和 `concat_feature` 拷贝。年尺度通常数 GB，不是 155G，但会叠在累计块上。

### 6.4 年内三表 + feature 分块

`FEATURE_CHUNK_SIZE=20` 时 `concat_feature` 每次 `copy` + `torch.concatenate`。三表 `merge` 再分配一块 union。年内峰值 = 三表块 + merge 结果 + Polars。

### 6.5 必须在内存里拿着全历史才能 dump

分年只降低**单年处理**峰值，不改变「fit dump 是一整块 `(N, T_all, F)`」。到最后一年，累计块本身就是最终 8–13 GB。若 dump/后续 `mask`/`fillna`/`save` 再复制，还会再加一截。

### 6.6 双 lookback、`fillna`、未释放的 Polars/线程池

`mincr` 内外各减一次 250 日；`fillna(0)` 可能新张量；`loads_pl` 加速器线程。相对 mesh 都是次要，但叠在 2× merge 上仍可观。

### 6.7 日志本身

`MemTrace` 只读 nbytes / RSS，不应打出 100G+。若用户测试的是 **bench `--profile minc` 且把 `--max-mesh-gb` 开很大还跑了 mesh**，那是故意分配网格，与生产 `auto/-1` 不是同一条路。需确认挂掉的是 **minc reconstruct** 还是 **bench**。

---

## 7. 当时提出、尚未做的方向

对话里提过、**没有落地**的：

1. 年累加不要走通用 4 轴 merge：按 date 轴 `concat` + 1 维 secid 重排（`concat_date`），避免 union `torch.full` 的 2× 峰值。
2. 去掉 minc/mincr 内层重复的 `CalculationWindow`。
3. `from_polars` 避免先做成巨大 numpy 再转 torch；或按 feature 写 mmap，年块落盘再拼。
4. fit dump 改为按年 mmap / 分文件，reconstruct 不必在 RAM 里拼全历史。
5. 确认 broadcast 高级索引有没有隐藏的 `(N,T,I,F)` 展开；若有，应采用明确写回原张量的有界分块拷贝。不要使用 `dst[secid_idx][:, date_slice] = ...`：第一步高级索引通常返回副本，后续赋值可能不写回 dst。
6. `MemTrace` 稳定后关掉。
7. 文档里 `INTERSECT_MESH_MAX_BYTES` 仍写 256MiB，与仓库 `-1` 不同步。

---

## 8. 下次建议先看的日志

minc reconstruct 开着 `MemTrace=True` 时，对照：

```text
mem minc chunk-merge-broadcast YYYYMMDD: rss=.. | hwm=.. | prev=.. | year=.. | union=.. | mesh4d=..
mem minc chunk-acc YYYYMMDD: rss=.. | acc=..
mem minc {db_key} loaded / transformed
mem minc tables-merge / year-block
```

判读：

- `rss ≈ union`（或 `prev+year+union`）→ 符合 2× 数据张量，要改的是 **不要全量 `torch.full` / 不要在 RAM 里攒全历史**
- `rss` 跟着 `mesh4d` 走 → 仍在物化 4 维下标（mesh 没关掉，或 broadcast 内部展开）
- 单年 `loaded`/`from_polars` 就把 rss 打到几十 G → 稠密化 / Polars，不是年 merge

服务器确认 OOM killer：

```bash
dmesg -T | grep -iE 'killed process|out of memory'
# 退出码 137 → SIGKILL
```

---

## 9. 一句话

minc fit 第一次静默退出，是全历史一次进内存。按年切之后，2021 年 155G 对得上 **当时推测的索引物化（此推断已撤回，不能把逻辑网格体积当作实际分配）**。改成 broadcast 并强制关闭 mesh 后 **仍然 OOM**，说明还要处理：union 全量新分配、可能的索引展开、`from_polars` 稠密化，以及「dump 前必须持有整段 fit 张量」这件事本身。


## 10. 再次核查与已实现的诊断（2026-09-18）

### 已确认事实

1. 本机 PyTorch 2.7.1：10×20 meshgrid 的两个输出逻辑大小各 1600 字节，底层 storage 分别仅 80 和 160 字节，stride 分别为 `(1, 0)` 与 `(0, 1)`。因此第 2.3 节的 131G 是完全物化时的逻辑估算，不能认定为实测内存。高级索引的峰值仍待服务器测量。
2. `loads_pl` 当前先读全列，再做 select；多文件线程读取和合并的瞬时峰值可能早于原来的 loaded 日志。
3. `concat_feature` 先 deepcopy 再 concatenate；按年 merge 分配新 union；全历史 fillna 分配新张量；mask 分配全尺寸布尔数组。这些路径确定存在，是否触发本次 OOM 未定。
4. **额外发现：** `ArrayMemoryMap.save` 使用 `f.write(values.tobytes())`，即使保存目标叫 mmap，写入仍会生成完整 bytes 副本；非连续数组还先转连续。这是可解释保存阶段峰值的具体路径，本轮没有优化它。
5. minc/mincr 不在当前 HistNormTasks 中，`save_norm` 对这两个 key 直接返回，不应再把历史归一化的重计算列为当前路径的主要嫌疑。
6. 整个 WezTerm 消失需结合 kernel、systemd-oomd 和 cgroup 证据；SIGKILL/137 本身不证明 OOM。父级 cgroup 限额、同组其他任务、系统内存压力都需要检查。

### 已实现

- `cli_recovery` 每分钟检查已登记 Hub，用户服务启动恢复窗口；正常退出或 `/quit` 不恢复，失败任务不重跑；登录 autostart 刷新桌面会话，支持 pause/resume/status。
- CLI 重建确认后按 key/frame 建独立任务、UUID 日志目录，stdout/stderr 持续 flush；普通异常和外部强杀均由持久失败通知流程处理。成功仅记录，失败发送有界尾部与日志路径。
- 独立采样器每两秒记录进程树 RSS、系统可用内存、swap、cgroup 当前量/峰值/限额/事件（包括父组）。无法获取的信息记录缺口。
- 探针覆盖读取前/全列读取后、Polars join/NumPy/张量、concat、merge 分配/拷贝、fill/mask 及保存边界。tensor 显示 logical 与 storage；`mesh4d` 显示名修正为 `mesh_logical_estimate`。
- 安装及验收说明见 `runs/systemd/README.md`。服务器需更新调度安装并在桌面手动打开一次新版 CLI 才登记恢复意图。

### 下一次服务器复现要回答的问题

- 峰值发生于哪个阶段？当前进程 RSS、进程树 RSS、cgroup 总量是否一致？注意进程树 RSS 求和可能重复计共享页，cgroup 总量还含其他进程及文件缓存。
- 内存上升后对象逻辑体积是否同步增长？若不同步，再调查分配器保留、共享存储、其他任务与隐藏临时分配，不直接称为泄漏。
- kernel/oomd 是否记录该 PID 或 cgroup？限额是否来自父组？若权限不足，只能保持“未确认 OOM”。
- 在下一轮优化前先保存本轮证据；当前输出格式和计算算法没有改变，不能声称已经解决 OOM。
