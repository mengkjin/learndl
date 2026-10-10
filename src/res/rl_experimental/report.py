"""Generate a self-contained Chinese explanation and result report."""
from __future__ import annotations

import base64
import csv
import html
import json
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: Any, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _percent(value: Any, digits: int = 2) -> str:
    return f"{100 * float(value):.{digits}f}%"


def _image_data(path: Path) -> str:
    if not path.is_file():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def generate_report(run_dir: str | Path, output: str | Path | None = None) -> Path:
    """Create an offline HTML report from a completed run; no panel or database needed."""
    run = Path(run_dir)
    metrics_path = run / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"completed run lacks metrics.json: {run}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = metrics["config"]
    updates = _read_csv(run / "update_history.csv")
    validations = _read_csv(run / "validation_updates.csv")
    test_history = _read_csv(run / "test_history.csv")
    split = metrics["splits"]
    train_steps = int(split["train"][1]) - int(split["train"][0])
    test_steps = int(split["test"][1]) - int(split["test"][0])
    experience_per_update = int(config["n_envs"]) * int(config["rollout_steps"])
    minibatches = experience_per_update // int(config["batch_size"])
    optimizer_steps = int(metrics.get("completed_updates", len(updates))) * minibatches * int(config["update_epochs"])
    test = metrics["test"]
    equal = metrics["alpha_equal_weight_baseline"]
    robust = metrics.get("robust_top50_baseline")
    first_day = test_history[0] if test_history else {}
    comparison = metrics.get("test_comparison") or {}
    reward = metrics.get("reward_definition", {
        "entrypoint": "legacy: reward_scale * log(nav_multiplier)",
        "config": {}, "scale": config.get("reward_scale", 100.0), "source_sha256": None,
    })
    image = _image_data(run / "test_comparison.png")
    best_update = metrics.get("best_update", "未知")
    run_name = html.escape(run.name)
    update_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(row.get(key, ''))}</td>" for key in (
            "update", "environment_steps", "train/value_loss", "train/explained_variance",
            "train/approx_kl", "validation/ppo/final_nav",
        )) + "</tr>"
        for row in updates
    )
    latest_validation = validations[-1] if validations else {}
    robust_cells = (
        f"<td>{_percent(robust['total_return'])}</td><td>{_percent(robust['max_drawdown'])}</td>"
        f"<td>{_percent(robust['annualized_volatility'])}</td><td>{_number(robust['mean_turnover'])}</td>"
        if robust else "<td colspan='4'>不适用</td>"
    )
    report = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RL 组合实验报告 · {run_name}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif;max-width:1120px;margin:32px auto;padding:0 24px;color:#17202a;line-height:1.7;background:#fbfbfa}}
h1{{font-size:28px}} h2{{margin-top:40px;border-bottom:1px solid #d7d9dc;padding-bottom:7px}} h3{{margin-top:25px}}
.lead{{font-size:17px;color:#34495e}} .flow{{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:20px 0}}
.step{{border:1px solid #adb5bd;border-radius:6px;padding:9px 12px;background:white}} .arrow{{color:#6c757d}}
.note{{border-left:4px solid #607d8b;background:#eef2f3;padding:10px 14px;margin:16px 0}} .warn{{border-left-color:#b36b00;background:#fff5e6}}
table{{border-collapse:collapse;width:100%;background:white;font-size:14px}} th,td{{border:1px solid #d7d9dc;padding:7px 9px;text-align:right}} th:first-child,td:first-child{{text-align:left}} th{{background:#f0f2f4}}
code,pre{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#f1f3f5}} code{{padding:2px 4px}} pre{{padding:14px;overflow:auto;border-radius:6px}}
.formula{{font-size:17px;text-align:center;padding:14px;background:white;border:1px solid #d7d9dc}} img{{width:100%;height:auto;background:white}}
.small{{font-size:13px;color:#59636e}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}} .metric{{padding:12px;border-top:3px solid #607d8b;background:white}}
</style></head><body>
<h1>强化学习组合构建：从数据到 PPO 更新</h1>
<p class="lead">本报告解释 Notebook 05 和 <code>train_experiment()</code> 实际执行的管线，并以运行 <code>{run_name}</code> 的输出为例。报告文件自包含，可离线打开。</p>

<h2>1. 这次实验说明了什么</h2>
<div class="grid">
<div class="metric">测试净收益<br><strong>{_percent(test['total_return'])}</strong></div>
<div class="metric">等权净收益<br><strong>{_percent(equal['total_return'])}</strong></div>
<div class="metric">相对等权收益差<br><strong>{_number(comparison.get('return_difference_pp', 0), 3)} 个百分点</strong></div>
<div class="metric">最佳 checkpoint<br><strong>第 {best_update} 轮</strong></div>
</div>
<p>测试段只有 <strong>{test_steps}</strong> 个 transition。PPO 最终净收益约 {_percent(test['total_return'])}，比 alpha 前 50 等权高 {_number(comparison.get('return_difference_pp', 0), 3)} 个百分点；同时最大回撤为 {_percent(test['max_drawdown'])}，年化波动为 {_percent(test['annualized_volatility'])}。这些数字证明工程链路可运行，不能据此判断策略具有稳定的样本外优势。</p>
<table><tr><th>策略</th><th>净收益</th><th>最大回撤</th><th>年化波动</th><th>平均换手</th></tr>
<tr><td>PPO</td><td>{_percent(test['total_return'])}</td><td>{_percent(test['max_drawdown'])}</td><td>{_percent(test['annualized_volatility'])}</td><td>{_number(test['mean_turnover'])}</td></tr>
<tr><td>Alpha 前 50 等权</td><td>{_percent(equal['total_return'])}</td><td>{_percent(equal['max_drawdown'])}</td><td>{_percent(equal['annualized_volatility'])}</td><td>{_number(equal['mean_turnover'])}</td></tr>
<tr><td>Robust top50</td>{robust_cells}</tr></table>
{f'<img src="{image}" alt="测试期净值、回撤、费用、换手、现金和相对等权净值图">' if image else ''}

<h2>2. 从数据到一天的组合结算</h2>
<div class="flow"><span class="step">真实 panel</span><span class="arrow">→</span><span class="step">仅训练段拟合归一化</span><span class="arrow">→</span><span class="step">alpha 候选 + 旧持仓</span><span class="arrow">→</span><span class="step">固定槽位与掩码</span><span class="arrow">→</span><span class="step">actor 两维动作</span><span class="arrow">→</span><span class="step">硬约束映射</span><span class="arrow">→</span><span class="step">次日开盘执行</span><span class="arrow">→</span><span class="step">收盘估值与 reward</span></div>
<p>每个槽位中的第一维动作修正 alpha 的选股排序，第二维经 softplus 转为配权偏好。<code>project_action()</code> 再强制只做多、最多 50 只、单股不超过 3% 和现金约束。未来的停牌与涨跌停状态不会进入 t 日 observation；环境只在 t+1 开盘结算时读取它们。</p>
<p>真实环境先把旧持仓从 t 日收盘计价至 t+1 日开盘，再先卖后买、扣除实际费用，最后从开盘计价到收盘。单日记录同时保存净值倍数、费用、换手、持仓、冻结/拒单以及 reward 分项。</p>
<h3>本次测试期第一步记账示例</h3>
<table><tr><th>决策日</th><th>结算日</th><th>单期净收益</th><th>隔夜收益</th><th>换手</th><th>费用/初始净值</th><th>期末 NAV</th></tr>
<tr><td>{html.escape(first_day.get('decision_date', '—'))}</td><td>{html.escape(first_day.get('return_end_date', '—'))}</td><td>{_percent(first_day.get('period_return', 0))}</td><td>{_percent(first_day.get('overnight_return', 0))}</td><td>{_number(first_day.get('turnover', 0))}</td><td>{_number(first_day.get('fee_estimate', 0), 6)}</td><td>{_number(first_day.get('nav', 0), 6)}</td></tr></table>
<p class="small">第一天从现金开始，隔夜收益通常为 0；开盘成交后产生换手和费用，随后日内收益进入期末 NAV。后续交易日的隔夜收益来自已有持仓。</p>

<h2>3. 一轮 PPO 训练到底做了什么</h2>
<p>训练段只有 <strong>{train_steps}</strong> 个 transition。每个环境在走完训练段后重置到随机起点，因此 128 步 rollout 会跨越多次 episode。4 个环境并行得到 4 × 128 = <strong>{experience_per_update}</strong> 条经验；它们重复使用同一段历史，并未创造 512 个独立的市场样本。</p>
<pre>for update in 1..{metrics.get('completed_updates', len(updates))}:
    rollout: 保存 state, action, log_prob_old, reward, done, V_old
    GAE: delta_t = r_t + gamma * V(s[t+1]) - V(s[t])
         advantage_t = delta_t + gamma * lambda * advantage[t+1]
         value_target_t = advantage_t + V_old(s[t])
    repeat {config['update_epochs']} epochs:
        shuffle {experience_per_update} experiences into {minibatches} minibatches of {config['batch_size']}
        update actor with clipped PPO objective
        update critic toward value_target
    replay fixed validation period; save checkpoint if selection score improves</pre>
<p>每轮有 {minibatches} 个 minibatch × {config['update_epochs']} 个 epoch = <strong>{minibatches * int(config['update_epochs'])}</strong> 次优化器更新；共 {metrics.get('completed_updates', len(updates))} 轮，即约 <strong>{optimizer_steps}</strong> 次。critic 并不预测下一日收益，而是估计“从当前状态继续按策略行动的折扣累计 reward”。GAE 用 critic 降低策略梯度方差，actor 根据 advantage 提高好于预期动作的概率。</p>
<div class="note">价值损失下降、explained variance 上升，只说明 critic 更贴近当前 rollout 的训练目标。它不等于组合净值提高，也不等于相对基准更强。</div>

<h3>逐轮诊断</h3>
<table><tr><th>轮次</th><th>累计经验</th><th>Value loss</th><th>Explained variance</th><th>Approx KL</th><th>验证 NAV</th></tr>{update_rows}</table>
<p class="small">最后一次验证的选择分数：{html.escape(latest_validation.get('validation/ppo/selection_score', '未知'))}。TensorBoard 原始事件和 CSV 均随结果包保存。</p>

<h2>4. Reward、选模和最终评价是三个目标</h2>
<div class="formula">rₜ = scale × [ log(NAVₜ₊₁/NAVₜ) − λT·turnover − λD·min(Rnet,0)² − λC·Σwᵢ² ]</div>
<p>actor 直接优化上式的折扣累计值。交易费已经进入净值倍数，换手惩罚表达的是额外的低换手偏好，不能理解为再次估算手续费。下行项只惩罚单期负收益；集中度项惩罚结算后股票权重平方和，较大的惩罚可能促使策略保留更多现金。<code>reward_scale</code> 只改变优化数值尺度。</p>
<p>当前 reward 实现为 <code>{html.escape(str(reward['entrypoint']))}</code>，scale={reward['scale']}，配置为 <code>{html.escape(json.dumps(reward.get('config', {}), ensure_ascii=False, sort_keys=True))}</code>。源文件 SHA-256 为 <code>{html.escape(str(reward.get('source_sha256')))}</code>。</p>
<ol><li><strong>训练目标：</strong>rollout 的折扣累计 reward。</li><li><strong>选模目标：</strong><code>{html.escape(str(metrics.get('selection_metric', config.get('selection_metric', 'nav'))))}</code>，只用固定验证段。</li><li><strong>最终评价：</strong>测试段净收益、回撤、波动、换手、费用及相对基准表现。测试不参与选模。</li></ol>
<div class="note warn">滚动 Sharpe、峰值回撤等依赖历史路径的目标需要有状态 reward 或在 observation 中加入足够状态。首版自定义接口刻意限定为无内部历史状态，避免隐藏不可复现的跨 episode 状态。</div>

<h2>5. 如何定制 Reward</h2>
<p>常用目标直接在 <code>RewardConfig</code> 中设置三个非负系数。自定义函数使用 <code>module:function</code>，接收只读的 <code>RewardContext</code> 和 JSON 参数，返回 <code>RewardResult</code>。上下文只含决策时状态和已经实现的一步成交反馈，不暴露未来 panel。</p>
<pre>RewardConfig(
    turnover_penalty=0.001,
    downside_penalty=2.0,
    concentration_penalty=0.01,
)

# 自定义示例
RewardConfig(
    custom_function="src.res.rl_experimental.reward:example_drawdown_aware_reward",
    custom_params={{"downside_penalty": 2.0, "turnover_penalty": 0.001}},
)</pre>

<h2>6. 代码与产物索引</h2>
<ul><li><code>real_data.py / prepare_real_data()</code>：点时数据快照。</li><li><code>env.py / PortfolioEnv.step()</code>：状态、动作、成交、reward。</li><li><code>policy.py / SharedStockNetwork</code>：共享逐股 actor/critic。</li><li><code>training.py / train_experiment()</code>：切分、rollout、PPO、验证和测试。</li><li><code>reward.py</code>：内置及自定义 reward 契约。</li><li><code>metrics.json</code>、<code>update_history.csv</code>、<code>validation_updates.csv</code>：本报告的数据来源。</li></ul>

<h2>术语表</h2>
<dl><dt>Transition</dt><dd>一次状态、动作、reward、下一状态的环境变化。</dd><dt>Rollout</dt><dd>更新网络前收集的一批 transition。</dd><dt>Advantage</dt><dd>某动作结果相对 critic 预期好多少。</dd><dt>GAE</dt><dd>在偏差和方差之间折中的 advantage 估计。</dd><dt>Clip</dt><dd>限制新旧策略概率比，避免一次更新过大。</dd><dt>Checkpoint</dt><dd>某轮训练后的模型参数快照。</dd></dl>
<p class="small">生成来源：{html.escape(str(run.resolve()))}。本报告是工程解释与实验诊断，不构成投资结论。</p>
</body></html>"""
    destination = Path(output) if output is not None else run / "report.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    return destination
