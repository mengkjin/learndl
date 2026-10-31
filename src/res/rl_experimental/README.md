# RL experimental portfolio construction

这是一个与 learndl 现有交易流程隔离的最小强化学习实验。它回答的问题是：在已有 alpha 和股票特征之后，PPO 能否学习每日选择最多约 50 只股票并给出权重，同时把换手费用和后续收益纳入长期奖励。

## 模型实际做什么

每个交易日按以下顺序运行：

1. 从当日可投资股票中按 alpha 选出候选股，并加入仍在持有的股票。
2. 把股票放入固定容量的槽位。槽位只为便于组成 batch，并不绑定股票身份。
3. 共享逐股网络用同一套参数编码每只股票，结合组合汇总信息，输出选股修正和配权偏好；critic 估计当前状态的长期价值。
4. 硬约束映射选出不超过指定数量的股票，并保证只做多、单股上限和现金约束。
5. 环境扣减交易费用、应用下一期收益、更新权重，并按可配置 reward 计算净值增长及可选的换手、下行和集中度惩罚。
6. Stable-Baselines3 负责 rollout、GAE、PPO minibatch 更新、checkpoint 和 TensorBoard 日志。

结构借鉴 EIIE 的关键思想：逐资产共享参数。因此换入一只新股票并不会改变网络参数维度。首版使用 MLP 编码已整理好的特征；时间卷积是后续实验项。

## 掩码和变化的股票池

观察中有两个掩码：

- `present_mask` 表示槽位中是否存在真实股票，用于屏蔽 padding 和计算组合汇总。
- `investable_mask` 表示股票能否成为目标持仓，用于屏蔽动作概率和最终选股。

候选池外的旧持仓仍会出现在槽位中。若它已经不可投资，critic 和记账仍能看到它，但目标权重被强制为零。掩码会进入网络；仅把 padding 特征填成零不足以阻止带偏置网络产生动作。

## 数据格式

`PanelData` 使用以下 NumPy 数组：

- `stock_ids [N]`：稳定且唯一的股票标识。
- `decision_dates [T]` 与 `return_end_dates [T]`：决策时点和收益结束时点。
- `features [T,N,F]`：决策时已知的特征。
- `alpha [T,N]`：决策时已知的监督学习分数。
- `investable [T,N]`：当时的投资资格。
- `forward_returns [T,N]`：决策后的收益，只供环境结算。

真实数据快照使用 schema v4，并另外保存 `execution_dates`、隔夜/日内收益、`can_buy`、`can_sell`、估值陈旧标志和逐日行业编码。行业编码只在构造 observation 时展开为 one-hot，不参与连续特征标准化。未来交易状态只由环境在结算时读取，不会进入 t 日 observation。

NPZ 可以用 `PanelData.save_npz()` 和 `PanelData.load_npz()` 读写。checkpoint 会记录 schema、特征顺序、证券和日期指纹；重新评估时数据不匹配会直接报错。

## 安装和运行

数据准备、训练和 notebooks 统一使用项目主环境 `.venv`。Gymnasium、Stable-Baselines3、Torch 和 ipykernel 随项目根依赖管理，不再创建模块独立环境或重复维护依赖清单。从项目根目录运行：

```bash
.venv/bin/python -m src.res.rl_experimental demo --timesteps 4096
```

4090 机器应先按 PyTorch 官方方式安装相同版本的 CUDA build，再安装其余依赖。训练时添加 `--device cuda`。首版环境在 CPU 上运行，小网络未必因 GPU 获得明显加速，输出中的 steps/second 用于判断瓶颈。

离线数据训练：

```bash
.venv/bin/python -m src.res.rl_experimental train panel.npz --output results/rl_experimental/run-01
```

重新加载最佳 checkpoint 并导出测试期确定性权重：

```bash
.venv/bin/python -m src.res.rl_experimental evaluate panel.npz results/rl_experimental/run-01/best_model.zip --output results/rl_experimental/run-01/export
```

结果包括模型、归一化参数、时间切分、依赖版本、TensorBoard 日志、验证及测试轨迹，以及 alpha 前 50 等权和 robust top50 两条基线。两条基线与 RL 共用成交内核；robust top50 的隔离实现逐项复刻项目生成器的缓冲、换手和行业槽位规则，并用实际成交后的持仓生成下一日目标。

## 接入 learndl 真实数据

数据准备和训练使用同一个项目主环境。alpha 必须显式指定，格式为 `factor@名称`、`pred@名称` 或 `sellside@名称@列名`。

先只检查 alpha 覆盖、股票数量和预计内存：

```bash
.venv/bin/python -m src.res.rl_experimental prepare-real \
  --alpha pred@<预测名称> --start 20180101 --end 20250421 \
  --output results/rl_experimental/snapshots/<实验名> --dry-run
```

检查通过后去掉 `--dry-run`。输出包含 `panel.npz`、`manifest.json` 和 `quality_report.json`。默认要求 alpha 每个交易日严格存在；低频信号必须显式传入 `--max-alpha-staleness`。`--alpha-lag` 用于人为滞后 alpha，`--alpha-sample-status` 记录它是样本外、样本内或未知；未知数据会标记为工程实验。

先做无训练回放：

```bash
.venv/bin/python -m src.res.rl_experimental replay-baselines \
  results/rl_experimental/snapshots/<实验名>/panel.npz \
  --output results/rl_experimental/baselines/<实验名>
```

再在项目主环境进行短训练：

```bash
.venv/bin/python -m src.res.rl_experimental train \
  results/rl_experimental/snapshots/<实验名>/panel.npz \
  --output results/rl_experimental/runs/<实验名>-smoke --timesteps 4096 --device cpu
```

正式三随机种子入口默认每个种子训练 100,000 步：

```bash
.venv/bin/python -m src.res.rl_experimental train-suite \
  results/rl_experimental/snapshots/<实验名>/panel.npz \
  --output results/rl_experimental/runs/<实验名>-suite --device cuda
```

默认按时间 60%/20%/20% 切分；可用 `--train-end-date` 和 `--valid-end-date` 固定边界。

## Reward：训练目标与选模目标

默认 reward 与旧版本完全相同：`100 × log(NAV[t+1] / NAV[t])`。`ExperimentConfig.reward` 接受 `RewardConfig`，可增加三种非负惩罚：

```python
from src.res.rl_experimental.reward import RewardConfig

reward = RewardConfig(
    turnover_penalty=0.001,       # 交易费之外的额外低换手偏好
    downside_penalty=2.0,         # 单期净负收益平方
    concentration_penalty=0.01,   # 结算后股票权重平方和
)
```

逐步公式为 `reward_scale × [log-growth - λT×turnover - λD×min(net_return,0)^2 - λC×sum(stock_weight^2)]`。手续费已经进入净收益，换手项不会替代费用模型。每个分项进入逐日 history、`update_history.csv` 和 TensorBoard 的 `rollout/reward_component/`。

无状态自定义函数用 `module:function` 配置，签名为 `RewardContext, JSON params -> RewardResult`。可直接运行的示例是 `src.res.rl_experimental.reward:example_drawdown_aware_reward`。实现入口、参数和源文件 SHA-256 会写入 `metrics.json`。函数拿不到未来 panel，返回值非有限或分项之和不等于 reward 会立即报错。

`selection_metric="nav"` 默认按验证净值保存最佳 checkpoint；可改为 `"discounted_reward"`。训练 reward、验证选模和测试报告是三个不同目标，测试集始终不参与选模。滚动 Sharpe、路径回撤等有状态目标尚未纳入首版接口。

## 服务器一键运行、邮件与本机分析

真实快照使用 schema v4（保留 v3 的停牌估值规则）：成交始终依赖 `day` 的真实开盘和可交易状态；收盘估值依次使用有效的 `day.close × adjfactor`、`adjprice.close`、仍上市期间的最近有效估值。无可成交开盘时冻结买卖，有可信收盘时仍计入价格变化；双表缺失可承接估值并汇总 warning，不认定为已确认停牌。复牌跳涨/跳跌完整进入隔夜收益，退市或没有任何历史估值时不继续补值。

质量报告的 `valuation` 记录来源、冲突和连续缺失区间；轨迹增加 `carried_holdings`、`missing_data_carried_holdings`、`max_held_stale_days`。承接天数从载入的历史窗口开始累计，更早的停牌时长未知。价格来源冲突时优先 day 的真实复权收盘并告警。所有估值处理只在 RL 内存中进行。

升级后需将实验 JSON 的 `rebuild_snapshot` 临时设为 `true`（或使用新的快照目录）生成 v4 快照，后续可改回 `false`。旧快照/checkpoint 仍可按原契约回放，但新实验入口不会静默沿用旧估值逻辑。建议先运行 `replay-baselines <快照目录>/panel.npz --split valid --train-end-date 20230101 --valid-end-date 20250101 --output results/rl_experimental/valuation_check`，确认验证基线通过，再运行服务器训练脚本。

RL 准备数据时会自动检查日行情、复权价格、涨跌停价、风格暴露和 alpha 的日期/证券重复键，以及上市记录与行业资料的对应键。发现重复时输出 `WARNING`，按源读取顺序保留最后一行；内容冲突时额外报告冲突键数量，这不代表识别出了最新修订值。处理仅发生在内存，不改写数据库；清理记录写入 manifest 的 `audit.duplicate_cleanup` 和 `quality_report.json`。缺失整日数据和非法价格仍按原规则报错。复用已有快照时不会重读源表，如需重新检查源数据请重建快照。

终端、notebook 和 HtmlCatcher 会实时显示带时间戳的 `[RL/阶段]` 输出，包括每批 128 个日期的数据读取、特征和资格构建、快照保存、实际时间切分、每轮 PPO 采样/更新、验证净值与最佳轮次、测试基线、报告及打包。数据和训练期间的输出同时进入运行目录的 stdout/stderr 日志。

如需修复数据库本身，可在项目 CLI 的 `2_data` 中选择 `Repair Day Limit Duplicates`。先以 `dry_run=True` 检查，再以 `dry_run=False` 修复；`end` 需包含最后决策日的下一交易日。该独立脚本只合并整行完全相同的记录，冲突文件保留原样并报错。逐日审计报告和原始备份位于 `results/data_repairs/day_limit/<运行时间>/`。修复时不要同时运行该表的数据更新任务。

项目脚本只发送一封任务邮件，统一附带 HtmlCatcher 日志、实验 ZIP（或全部分卷）和 JSON 清单；失败运行同样附带已生成的诊断包。邮件遵循项目的 `MACHINE.emailable` 设置及标准发送失败队列。分卷不会降低这封邮件的总附件大小。

项目交互 CLI 中可选择 `4_train / 3_train_rl_experiment`（Train RL Experiment）。脚本默认读取 `src/res/rl_experimental/example_server_experiment.json`。可设置 `config_path`、覆盖任务邮件的 `recipient`，或设置 `no_email=True` 跳过整封邮件。收件人依次取脚本参数、实验配置、项目默认邮箱。配置中的相对路径均以项目根目录为基准；训练步数、日期、alpha 和设备在 JSON 中修改。

服务器配置模板是 `example_server_experiment.json`，默认 CUDA、`pred@gru_day_V1`、100,000 步和 seed 7。以下模块命令不经过项目任务邮件，保留独立分卷发送方式；需要 HTML 日志与结果包合并发送时，使用上述项目脚本：

```bash
.venv/bin/python -m src.res.rl_experimental run-experiment \
  src/res/rl_experimental/example_server_experiment.json
```

命令按顺序校验或准备快照、训练、生成离线中文报告、打包并发邮件。若 CUDA 不可用会在训练前失败并生成诊断包。本机 CPU 验收模板为 `example_local_experiment.json`：

```bash
.venv/bin/python -m src.res.rl_experimental run-experiment \
  src/res/rl_experimental/example_local_experiment.json --no-email
```

附件包含 TensorBoard、CSV 轨迹、指标、图、HTML 报告、最佳/最终 checkpoint、归一化参数和数据清单，不包含 `panel.npz`。超过 `bundle_max_mb` 时生成 `.zip.001`、`.zip.002` 等分卷；默认及硬上限为每卷 20,000,000 字节，压缩后按最少卷数均分。全部下载到同一目录后，用 Keka / 7-Zip 打开 `.zip.001` 即可解压。旧附件可将所有 `.zip.partNNN` 改名为 `.zip.NNN` 后解压（使用下方 analyze-bundle 校验时仍需保留清单中的原文件名）。每封邮件附一卷和同一份 `.parts.json`。每卷和包内文件都有大小及 SHA-256。邮件失败只影响 delivery 状态，不会重跑训练：

```bash
.venv/bin/python -m src.res.rl_experimental resend-bundle \
  results/rl_experimental/server_runs/bundles/<run>.parts.json
```

在本机把 `.parts.json` 和全部 ZIP/分卷放在同一目录，然后校验及安全解包：

```bash
.venv/bin/python -m src.res.rl_experimental analyze-bundle \
  /path/to/<run>.parts.json --output results/rl_experimental/received
```

解包后的 `report.html` 可离线打开，Notebook 06 提供训练曲线、reward 分解和 TensorBoard 入口。分析过程不访问数据库，也不加载模型；checkpoint 重放仍需与数据契约匹配的原始 `panel.npz`。失败实验也会发送诊断包，但 `analyze-bundle` 会明确返回失败状态，不能当成成功结果。

## TensorBoard：每轮更新与等权比较

从项目根目录启动，然后打开 http://localhost:6006；notebook 05 也提供训练前启动的内嵌入口：

```bash
.venv/bin/python -m tensorboard.main --logdir results/rl_experimental/runs --host localhost --port 6006
```

在 Runs 中选择一个或多个实验。日志保存在各次输出目录的 `tensorboard/PPO_1/` 中。Scalars 里的横轴是累计环境步数，每个更新点对应一批 rollout 完成全部 PPO 优化；默认每轮 4 × 128 = 512 条经验。第 0 轮仅记录初始验证，不参与最佳模型选择。

- `train/`：更新编号、policy/value loss、entropy、KL、clip fraction、学习率等。`explained_variance` 基于该轮采样时的价值预测；`loss` 是 SB3 最后一个 minibatch 的损失，并非整轮平均损失。
- `rollout/`：这一批训练经验的平均 reward、换手、费用、现金和持仓；真实环境还包括冻结与拒单。费用均值以各段初始净值为单位，随机起点的训练经验不拼成投资净值曲线。
- `validation/ppo/`、`validation/equal_weight/`、`validation/comparison/`：固定验证段、从现金开始的收益、回撤、波动、费用及差异。等权基准是同一 alpha 前 50 等权，受相同成交限制。
- `time/`：本轮采样、优化、验证耗时与吞吐量。每轮记录末尾刷新文件，包含最后一轮更新。
- `test/`：训练完成后最佳模型的测试指标与基准比较；Images 中的 `test/comparison/curves` 展示净值、回撤、累计费用、换手、现金及相对等权净值。

`ExperimentConfig.eval_freq=0`（默认）表示每轮验证；正数表示环境步数间隔，达到阈值后在完整更新结束时执行。最后一轮必定验证。CLI 可用 `--eval-freq 4096` 降低长验证段的开销；即使降低验证频率，更新指标仍逐轮记录。

`update_history.csv` 每轮一行，`validation_updates.csv` 包括第 0 轮和实际执行的验证；`validation_equal_weight_history.csv` 保存固定验证基准。原 `validation_history.csv` 仍是最佳模型的逐日验证轨迹。最终 `test_comparison.png` 与 TensorBoard 图一致，`metrics.json` 另存 `best_update`、`completed_updates` 和 `test_comparison`。

收益差 `return_difference_pp` 使用百分点，相对净值 `relative_nav_return` 为 `NAV_PPO / NAV_equal − 1`；费用以初始净值为单位。所有回报均扣除交易费用。测试只做最终报告，不用于选 checkpoint。旧日志可继续读取，但无法恢复此前未记录的最后一轮指标。

## 当前边界

真实环境按“t 日收盘决策 → t+1 日开盘执行 → t+1 日收盘估值”结算。停牌或无有效开盘时冻结；开盘触及涨停不能买、触及跌停不能卖；先卖后买，费用按实际成交额计算。冻结仓位占用持仓数，被动超过单股上限时不会强平，但禁止继续增仓。

它仍是日频近似，尚未处理开盘排队、成交容量、部分成交和市场冲击，日线也无法完整还原盘中停牌。行业目前作为输入和 robust top50 规则使用，尚未成为 RL 的硬暴露约束。合成数据和真实数据工程通过都不能作为投资有效性的证据。

运行测试：

```bash
MPLCONFIGDIR=/tmp/learndl-mpl .venv/bin/python -m unittest discover -s src/res/rl_experimental/tests -v
```

## 观察实验过程的 notebooks

`notebooks/` 提供六个按顺序阅读的实验：

1. `01_synthetic_data_and_environment.ipynb`：检查合成行情、alpha、投资资格和单步交易记账。
2. `02_ppo_training_diagnostics.ipynb`：训练小型 PPO，比较验证、测试和等权基线，并打开 TensorBoard。
3. `03_masks_policy_and_real_data.ipynb`：检查共享逐股网络、掩码及真实 `PanelData` 接口。
4. `04_real_data_and_execution.ipynb`：以 `pred@gru_day_V1` 审计并导出真实快照，检查覆盖率、特征、候选池和成交事件，回放训练段基线。
5. `05_real_data_ppo_comparison.ipynb`：独立准备或复用同一快照，配置 reward，运行 4,096 步 PPO，读取训练曲线，对比 PPO、等权和 robust top50，并验证 checkpoint 重载。
6. `06_server_bundle_analysis.ipynb`：在本机校验并解包服务器附件，离线阅读报告、reward 分解和 TensorBoard；不需要行情数据库。

在 notebook 的内核选择器中选择项目的 `.venv/bin/python`，不再使用旧的 “RL Experimental” 内核。第一段代码会显示实际解释器路径。若使用只识别已注册内核的 Jupyter 前端，可从项目根目录注册主环境：

```bash
.venv/bin/python -m ipykernel install --user --name learndl --display-name "Python (learndl)"
```

04、05 默认使用 2025-01-02 至 2025-02-28 的决策日期，输出位于 `results/rl_experimental/`。依次运行即可完成数据准备；日期、alpha 方向/滞后及样本外状态均可在配置单元修改。默认每个 alpha 日期必须存在，行情需要覆盖特征回看期和最后决策日的下一交易日。dry-run 只审计 alpha 与预计数组内存，正式导出才完整检查行情及其他数据。

两本 notebook 共享默认快照位置；复用前检查配置、文件指纹和数据契约。源数据库变化不会自动重建快照，设置 `REBUILD_SNAPSHOT=True` 才会覆盖导出。05 每次训练创建独立运行目录，默认为 CPU；4090 可改为 `DEVICE="cuda"`。此短实验的 alpha 样本外状态默认未知，结果标记为工程实验。


### 退市核销与危险股票（schema v4）

真实快照在决策日用 `DATAVENDOR.INFO.get_st(t)` 的全部异常类别及项目 `warnst` 的 90 个自然日公告规则硬剔除股票，不增加低价/弱势筛选。优先读取当日 `exposure/warns_st_and_delist_90`，缺缓存时只读同期公告，不写回数据库。公告标题匹配“可能.*风险警示”或“可能.*退市”。缺文件与空文件分别记录；缓存来源覆盖未经验证也单独标记。历史覆盖不足会告警并继续应用已有规则，不代表完整复现实盘筛选。

风险股的候选资格和目标权重归零，robust 旧持仓缓冲不能绕过；卖不出的实际持仓仍保留。风险解除后可以再次入选。结算在历史 `delist_dt` 当日或其后第一个交易日开盘前，将剩余上一收盘持仓金额全额核销：无现金回收，无费用和换手。**这是实验估值假设，并非所有真实退市股票的最终回收价值。** 普通停牌、上市前或缺价不会被当成退市。未来退市事件不进入此前观测。

新增 `risk_excluded`、原因位（1=ST/异常，2=公告预警，可叠加）和 `delisting_event` 数组均参与指纹。旧 panel/checkpoint 可按旧契约读取；`run-experiment` 必须重建 v4 快照。全部资产核销后 NAV=0、收益=-100%，RL 和基准停止交易；内置对数奖励对倍率使用 `1e-12` 下限，自定义函数收到真实零倍率。零净值基准的相对净值指标为未定义。

轨迹新增 `written_off_stock_ids`、`delisting_loss`（初始净值单位）、`delisting_events`（核销持仓数量）、`risk_unsold_holdings` 和 `assets_exhausted`。汇总记录累计损失/事件数；训练 CSV/TensorBoard 记录对应数值分项。质量报告及 manifest 保存逐日风险来源/覆盖缺口；终端与 HtmlCatcher 捕获覆盖摘要及核销事件。

服务器更新代码后先将 JSON 中 `rebuild_snapshot` 设为 `true`，或单独执行 `prepare-real` 创建新目录；先用上方 `replay-baselines --split valid` 验证两条基线，再开始 PPO。`train_experiment` 现在也在 PPO 开始前回放验证期两条基线，并保存 `validation_baseline_metrics.json` 与对应逐日 CSV。重建成功后可将该开关改回 `false`。

## 低换手与季度滚动实验

`example_walk_forward.json` 固定了本阶段研究协议：先在 2017–2020 训练、2021–2022 验证，筛选换手惩罚 `{0, 0.0003, 0.001, 0.003}` 与持仓保留加分 `{0, 0.5}`；种子为 `7/17/29`。筛选 24 次训练，每次 30,000 步；随后原始 RL、仅惩罚、惩罚＋保留三组在 2023Q1–2025Q4 每季度重新训练四年、验证一年，每次 100,000 步，合计 108 次。实际步数按完整 rollout 向上取整。

持仓保留是在标准化 alpha＋actor 排序分数上给符合资格的旧持仓加分，不改变配权网络，也不能绕过 ST/预警过滤。无训练 `retention_rule` 使用同样候选、排序及持仓加分，actor 动作为零，因此配权为等权。继续报告原等权，但以 robust 为主要基准。

主要训练/选模成本为原费用 3.5 bp＋额外摩擦 3 bp；评估额外 `0/1/3/5/10 bp`，分别从账户状态重新执行交易与模型推理。该模型是按成交额统一收费的近似，**不是经过实盘校准的滑点、容量或冲击模型**。`base_fee`、`extra_cost` 分别记录。换手奖励惩罚是额外偏好，不计入交易费用。

```bash
# 只列出任务与窗口，不训练、不写输出目录
.venv/bin/python -m src.res.rl_experimental run-walk-forward \
  src/res/rl_experimental/example_walk_forward.json --dry-run

# 先测量本机/服务器环境；有 CUDA 时自动同时测 CPU 与 CUDA
.venv/bin/python -m src.res.rl_experimental profile-env \
  results/rl_experimental/snapshots/pred_gru_day_V1_20170101_20251230/panel.npz \
  --output results/rl_experimental/profile --steps 512 --train-steps 512

# 可分两阶段，也可省略 --stage 一次串联
.venv/bin/python -m src.res.rl_experimental run-walk-forward \
  src/res/rl_experimental/example_walk_forward.json --stage screen
.venv/bin/python -m src.res.rl_experimental run-walk-forward \
  src/res/rl_experimental/example_walk_forward.json --stage quarters
```

服务器任务列表继续选择 **Train RL Experiment**，把 `config_path` 改成 `src/res/rl_experimental/example_walk_forward.json`。脚本识别 `kind=walk_forward`，将汇总 ZIP/JSON 加到原任务邮件，保留 HtmlCatcher；子训练不单独发送邮件。可用配置中的 `stage` 分阶段运行。新入口读取已经准备好的 v4 快照；需要时先执行 `prepare-real`，不会自动改写数据库或重建已有快照。

### 账户、时间与恢复契约

- 每条方案/种子/成本路径只在首季度从现金开始，之后延续实际持仓、现金和净值；换模型不免费清仓。固定模型对照始终使用首季度 checkpoint。
- 按决策日选择季度模型，按结算日统计季度收益。内部季度边界的 transition 不丢弃、不重复；整体实验结束日及之后结算的收益排除，避免多出一个只有一天的季度。训练和验证收益不能跨越各自截止日。
- 新实验使用 `observation_version=2`：账户第二维固定为零，不让模型利用回测结束倒计时；旧配置默认版本 1，旧 checkpoint 仍能读取。
- 每折只在训练段拟合标准化；同折标准化数组、候选排序及验证基线跨种子复用。训练只保留统计，详细持仓保留在验证/回放轨迹中；自定义 reward 仍收到原上下文。
- 完成训练的模型、normalizer、指标保存哈希；恢复时校验配置、代码和数据指纹。失败训练重跑该任务，成功任务复用。连续评估失败后从首季度重新回放账户，不跳过失败季度，不重训已完成模型。改变研究配置或代码时应使用新输出目录。
- `selected.json` 是阶段 A 冻结的参数，带独立指纹，不允许根据季度结果手动改成“更优”参数后继续复用任务。

### 研究判断与输出

主要成本下至少 2/3 种子超过 robust、季度中位数胜率至少 60%、中位数换手不超过 robust 的 1.5 倍、最大回撤恶化不超过 2 个百分点，且额外 5 bp 情景仍保留中位数累计收益优势，才标记 `research_pass=true`。即使通过，`production_ready` 仍为 false；历史 alpha 的样本外来源、资金容量与前向模拟仍需审计。2023–2025 均标为历史开发验证，不作为全新盲测。

`summary/` 含所有连续轨迹、季度收益、种子离散度、5/10/20 日区块重采样、研究门槛判断及离线 HTML；`screen/` 和 `quarters/` 保存完整模型、日志、TensorBoard 与 normalizer。邮件只含汇总包，完整模型留在服务器。Notebook `07_walk_forward_analysis.ipynb` 可在本机读取汇总目录，完全不读取数据库或加载模型。

性能报告分别列出无缓存/详细日志与缓存/精简日志的相同动作序列耗时，并检查净值、换手和费用一致。预计算成本单独记录；计时与 cProfile/tracemalloc 分开，避免把测量开销算成提速。CPU/CUDA 的训练、推理批量 1/4 和验证耗时分别保存。`--train-steps 0` 只测环境。短测 ETA 不包含数据准备和全部成本回放，不是运行时间承诺。

## YAML 配置与服务器空闲调度

服务器脚本默认配置改为 `configs/strategy/rl_experiment.yaml`。文件内同时提供
`single` 和 `walk_forward` 两套完整参数，通过 `mode` 选择；旧 JSON 路径仍可显式传入。
当前配置选择季度方案；切换方案前检查其行情快照、日期及完整训练计算量。

```bash
python scripts/4_train/3_train_rl_experiment.py --config_path configs/strategy/rl_experiment.yaml
python scripts/4_train/3_train_rl_experiment.py --config_path src/res/rl_experimental/example_server_experiment.json
python -m src.automation.scheduling.idle_worklist rl-status
```

自动模式仅响应已提交并同步到服务器的 YAML 修改。修改 `rerun_mark` 可重跑；失败
不自动重试。等待期间多次修改只运行最新版。每次自动请求有独立输出目录和配置
来源记录；邮件仍包含任务 HTML 与 ZIP/JSON，不额外发送 RL 结果邮件。
部署及等待条件见 `configs/automation/README.md` 的 RL 实验章节。

### 精简日志、任务邮件与结果补发（2026-10-28）

季度实验主日志默认每季度两条：训练/验证完成、全部成本情景测试回放完成。
筛选和整套实验另有开始/完成摘要。完整输出保留在运行目录 `detail.log`，
每个子训练的 `stdout.log`、`stderr.log`、CSV 和 TensorBoard 继续保存。
若长时间仍持续产生进展，最多每小时补一条心跳，避免触发 12 小时无输出超时；
心跳不是独立计时线程，程序真正卡住时不会掩盖超时。

ZIP 分卷不等于邮件大小受限。旧脚本会把所有分卷与 HTML 一并装入单封邮件。
现在 RL 任务在 HtmlCatcher 关闭之后按 **完整 MIME 编码大小**分组，默认每封
不超过 18 MiB（工程预算，不是对邮箱供应商限额的声明）。小结果仍为一封任务邮件；
超限时才发送同一任务的编号分批邮件，不额外另发日志通知。大于 512 KiB 的 HTML
使用 `.html.gz`，解压后可离线查看。极大附件使用 `.mailpartNNN` 与其 JSON 清单，
须先按清单顺序拼回原文件，再使用 `analyze-bundle`。普通 8/10 MiB ZIP 分卷无需此步骤。

每封的实际 MIME 大小、收件人、SHA-256、SMTP 接受状态及错误写入
`bundles/task_mail/<id>/delivery.json`；补发任务的记录在 `recovery/<id>/mail/`。
SMTP 接受不等于收件箱已收到。部分失败不会重训，也不会把同一封邮件放进全局
outbox 重复排队。只重试未确认发送的分批邮件：

```bash
python -m src.proj.util.web.task_mail <包含-delivery.json-的目录>
```

如果进程在发送过程中被中断，`in_flight=true` 表示结果不明，工具拒绝盲目补发；
先核对邮箱/SMTP 日志，再决定是否重新发送。

Autofix `20261028_recover_rl_results_5d17935c_01` 专门检查运行
`5d17935c80694717a8df5fbd055791d4`。它核验全部计划训练的状态和模型/normalizer/
指标哈希、训练预算、冻结选模参数，以及每条测试轨迹对快照日期轴的完整覆盖和
净值/换手/费用一致性。只读取快照的日期数组，不加载模型或数据库、不重新训练。
检查通过才创建新结果包；失败只附诊断 JSON 和本次任务 HTML。原模型和结果不改动。
新包含离线报告、全部测试轨迹、训练更新/验证 CSV 及审计结论，不含行情和模型文件；
原模型与完整 TensorBoard 保留在服务器。

本机可用 Notebook 07 或 `analyze-bundle` 打开补发包。进程退出码 0 只是正常退出
证据，不能替代这份产物审计。审计确认的是工程完整性与一致性，不是新的前向实盘验证。

### 季度任务恢复（2026-10-28）

已有恢复粒度是一个“窗口 × 方案 × 种子”训练任务：`task_status.json=success`
且模型、normalizer、指标的 SHA-256 校验通过时直接复用。尚未完成的子任务从头
训练，不是从中断的 PPO minibatch 继续。所有训练已完成而测试缺失时，只重新
回放测试；连续持仓/现金的所有路径仍从首季度开始，尚不支持测试回放中途接续。

旧流程要求整个 RL 包代码指纹一致，日志或邮件代码变化也会阻止旧目录恢复。
现在显式恢复读取原 `system.json.git_revision`，从本地 Git 导出该提交的 RL
Python 源码到 `recovery/resume-*/source/`，在独立 worker 进程中使用原模块路径
加载，保留 checkpoint 和自定义函数导入语义。套件/子任务原指纹、完整行情快照
契约、已完成产物哈希和依赖版本仍要匹配，不通过修改指纹来绕过验证。
原提交不在本地 Git、已完成模型损坏、配置变化、冻结筛选丢失，或完成状态不明
（已有 metrics 却没有 success 回执）时停止并报告；不覆盖或重训可疑的已完成结果。

新增 autofix `20261028_resume_rl_5d17935c_02`，替代尚未启动的只审计 `_01`：

- 完整：只打包补发结果，无训练请求。
- 未完整且可安全恢复：写入主机 `rl_experiment_resume/requests/` 的不可变请求。
  autofix 自身不训练，不需要 GPU；维护结束后，现有 idle worker 等待 GPU、维护门禁
  和训练锁，与正常模型、普通 RL 共用一个训练插槽。
- 恢复 worker 使用原输出目录和原参数，保存恢复计划与来源备份；原成功模型复用，
  缺失子任务补齐后重新审计全部测试，再发送带附件的任务结果邮件。

同一显式请求 ID 只尝试一次；成功或失败都不自动重跑，也不修改旧运行账本。
新请求与原 YAML `rerun_mark` 独立，不触发另一次完整季度实验。失败后应先处理诊断，
再发布新 autofix ID。查看状态仍使用 `python -m src.automation.scheduling.idle_worklist rl-status`。
仅邮件失败应使用 `task_mail` 补发，不需要请求训练恢复。
