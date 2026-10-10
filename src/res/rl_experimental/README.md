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

真实数据快照使用 schema v2，并另外保存 `execution_dates`、隔夜/日内收益、`can_buy`、`can_sell`、估值陈旧标志和逐日行业编码。行业编码只在构造 observation 时展开为 one-hot，不参与连续特征标准化。未来交易状态只由环境在结算时读取，不会进入 t 日 observation。

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

若准备阶段报 `trade_ts/day_limit contains duplicate date/secid rows`，在项目 CLI 的 `2_data` 中选择 `Repair Day Limit Duplicates`。先以 `dry_run=True` 检查，再以 `dry_run=False` 修复；`end` 需包含最后决策日的下一交易日。脚本只合并整行完全相同的记录，冲突文件保留原样并报错。逐日审计报告和被修改文件的原始备份位于 `results/data_repairs/day_limit/<运行时间>/`，可按报告中的原路径恢复备份。修复时不要同时运行该表的数据更新任务。此脚本只检查已存在的日文件，不补全缺失日期或重新下载冲突数据。

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

附件包含 TensorBoard、CSV 轨迹、指标、图、HTML 报告、最佳/最终 checkpoint、归一化参数和数据清单，不包含 `panel.npz`。超过 `bundle_max_mb` 时生成 `.part001` 等分卷；每封邮件附一卷和同一份 `.parts.json`。每卷和包内文件都有大小及 SHA-256。邮件失败只影响 delivery 状态，不会重跑训练：

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
