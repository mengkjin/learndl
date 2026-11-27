"""Generate offline Graph, market, event and learning-rate comparisons from server results."""

from __future__ import annotations

import argparse
from html import escape
import json
import math
from pathlib import Path
import re
from typing import Any, cast
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from .context_diagnostics import residual_snapshot

__all__ = ["compare_v5", "compare_learning_rate", "main"]

CAMPAIGN_FIELDS = (
    "input.factor_sequence.dataset",
    "train.factor_sequence.folds",
    "model.labels",
    "env.random_seed",
    "train.max_epoch",
    "model.window",
    "train.trainer.optimizer.name",
    "train.trainer.scheduler.name",
    "train.trainer.scheduler.param.step_size",
    "train.trainer.scheduler.param.gamma",
)
SOURCES = (
    "factorseq_v4_graph",
    "factorseq_v5_graph_market",
    "factorseq_v5_graph_market_events",
)
BRANCHES = (
    "market_factor_correction",
    "market_stock_correction",
    "event_factor_correction",
)


def _config(root, name, lr):
    config = yaml.safe_load((root / "configs/model.yaml").read_text())
    if config["model.name"] != name or config["train.trainer.learn_rate.base"] != lr:
        raise ValueError(f"Unexpected model name or learning rate in {root}")
    return config


def _campaign(config, baseline):
    for key in CAMPAIGN_FIELDS:
        if config[key] != baseline[key]:
            raise ValueError(f"Incomparable campaigns: {key}")


def _test_metrics(root, expected_nodes):
    from src.res.model.analytic.original_comparison import daily_results

    rows = daily_results(root)
    if rows is None:
        raise FileNotFoundError(f"Server BasicTest snapshot unavailable: {root}")
    daily = _unique_index(
        rows.query('model_num == 0 and submodel == "best"'), "date"
    ).sort_index()
    if not daily.index.equals(expected_nodes.index):
        raise ValueError(f"BasicTest dates differ from the frozen benchmark in {root}")
    mismatch = ~daily.model_date.eq(expected_nodes).fillna(False)
    if mismatch.any():
        date = mismatch[mismatch].index[0]
        raise ValueError(f"Wrong training node on {date} in {root}: "
                         f"expected {expected_nodes.at[date]}, found {daily.at[date, 'model_date']}")
    return daily


def _unique_index(frame, keys):
    indexed = frame.set_index(keys)
    if not indexed.index.is_unique:
        raise ValueError(f"Duplicate keys in report: {keys}")
    return indexed


def _score_ic(frame, score, label):
    return frame[score].corr(frame[label], method="spearman")


def _snapshot_rows(left, right, daily):
    """Verify stock/label identity and attribute rank/Top 50 changes one date at a time."""
    baseline, candidate = residual_snapshot(left), residual_snapshot(right)
    expected = {f"{date}.feather" for date in daily.index}
    if any(
        {p.name for p in folder.glob("*.feather")} != expected
        for folder in (baseline, candidate)
    ):
        raise ValueError("Residual snapshots and BasicTest dates differ")
    for date, row in daily.iterrows():
        a = _unique_index(pd.read_feather(baseline / f"{date}.feather"), "secid")
        b = _unique_index(
            pd.read_feather(candidate / f"{date}.feather"), "secid"
        ).sort_index()
        if len(a) != len(b) or not a.index.isin(b.index).all():
            raise ValueError(f"Stock membership differs on {date}")
        if not np.array_equal(
            a.label.to_numpy(), b.label.reindex(a.index).to_numpy(), equal_nan=True
        ):
            raise ValueError(f"Training labels differ on {date}")
        scores = b[["pred", "graph_score", *BRANCHES]].to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(scores).all():
            raise ValueError(f"Non-finite prediction or score contribution on {date}")
        if not np.allclose(b.pred, b.graph_score + b[list(BRANCHES)].sum(axis=1), atol=1e-6):
            raise ValueError(f"Market score decomposition differs on {date}")
        b["same_weights_base"] = b.graph_score
        if "market_stock_correction" in a and "event_factor_correction" in b:
            b["same_weights_base"] += (
                b.market_factor_correction + b.market_stock_correction
            )
        old_top = set(
            b.sort_values("same_weights_base", ascending=False, kind="stable")
            .head(50)
            .index
        )
        new_top = set(
            b.sort_values("pred", ascending=False, kind="stable").head(50).index
        )
        yield {
            **row.to_dict(),
            "date": date,
            "n_stocks": len(b),
            "same_weights_delta_std_ic": _score_ic(b, "pred", "label")
            - _score_ic(b, "same_weights_base", "label"),
            "top50_replaced": len(new_top - old_top),
            "top50_enter_label_mean": b.label[b.index.isin(new_top - old_top)].mean(),
            "top50_exit_label_mean": b.label[b.index.isin(old_top - new_top)].mean(),
        }


def _hac_interval(delta, lag=20):
    values = np.asarray(delta, dtype=float)
    if len(values) <= lag or not np.isfinite(values).all():
        raise ValueError("Incomplete paired daily Rank IC for Newey–West estimation")
    centered = values - values.mean()
    variance = centered @ centered
    for i in range(1, lag + 1):
        variance += 2 * (1 - i / (lag + 1)) * (centered[i:] @ centered[:-i])
    half = 1.96 * math.sqrt(max(0.0, variance)) / len(values)
    return values.mean() - half, values.mean() + half


def _event_table(context, daily):
    from src.data.preprocess.factor_sequence.schema import EVENTS

    context = _unique_index(context, "date").sort_index()
    if not context.index.equals(pd.Index(daily.date, name="date")):
        raise ValueError("Event diagnostics do not match paired test dates")
    rows = []
    for event in EVENTS:
        flag = context[event]
        trigger = flag.eq(1)
        # A contiguous trigger segment is one event. A source interruption
        # invalidates the active segment until a fresh confirmation is seen.
        starts = trigger & ~trigger.shift(fill_value=False)
        segment = starts.cumsum().where(flag.notna())
        known_segment = starts.groupby(flag.isna().cumsum()).cumsum().gt(0)
        age = np.expm1(context[event + "__age"] * np.log1p(252)).round()
        for begin, end in ((0, 5), (6, 20), (21, 60)):
            mask = age.between(begin, end) & known_segment & flag.notna()
            rows.append(
                {
                    "事件": event,
                    "交易日间隔": f"{begin}–{end}",
                    "已知触发段数": int(starts.sum()),
                    "区间触发段数": segment[mask].nunique(),
                    "天数": int(mask.sum()),
                    "平均 ΔRank IC": daily.set_index("date")
                    .loc[mask, "delta_ic"]
                    .mean(),
                }
            )
    return pd.DataFrame(rows)


def _portfolio(roots, names):
    fields = {
        "annualized": "年化超额",
        "mdd": "最大超额回撤",
        "ir": "IR",
        "turnover": "累计换手",
    }
    rows = []
    windows = []
    for root, name in zip(roots, names, strict=True):
        table = pd.read_excel(
            root / "results/detailed_alpha_data.xlsx", "t50@frontface"
        )
        top = table.query(
            'factor_name == "best" and benchmark == "univ" and strategy == "Top_50" and suffix == "lag0"'
        )
        if len(top) != 1:
            raise ValueError(f"Unexpected Top 50 report in {root}")
        record = top.iloc[0]
        windows.append((record.start, record.end))
        rows.append(
            {"模型": name, **{label: record[key] for key, label in fields.items()}}
        )
    if len(set(windows)) != 1:
        raise ValueError("Top 50 report ranges differ")
    return pd.DataFrame(rows)


def _chart(title, x_label, y_label, series):
    points = [point for values in series.values() for point in values]
    xmin, xmax = min(x for x, _ in points), max(x for x, _ in points)
    ymin, ymax = (
        min(0.0, min(y for _, y in points)),
        max(0.0, max(y for _, y in points)),
    )
    xspan, yspan = max(1, xmax - xmin), max(1e-8, ymax - ymin)

    def xpos(x):
        return 62 + 638 * (x - xmin) / xspan

    def ypos(y):
        return 170 - 142 * (y - ymin) / yspan

    grid = []
    for y in np.linspace(ymin, ymax, 5):
        location = ypos(y)
        grid.append(
            f'<path d="M 62 {location:.1f} H 700" stroke="#cbd8d2"/>'
            f'<text x="56" y="{location + 4:.1f}" text-anchor="end">{y:.3f}</text>'
        )
    for x in sorted(
        {pts[0][0] for pts in series.values()} | {pts[-1][0] for pts in series.values()}
    ):
        grid.append(f'<text x="{xpos(x):.1f}" y="190" text-anchor="middle">{x}</text>')
    lines, legend = [], []
    for (name, values), color, dash in zip(
        series.items(), ("#08776b", "#a64b25"), ("none", "6 4")
    ):
        coords = " ".join(f"{xpos(x):.1f},{ypos(y):.1f}" for x, y in values)
        lines.append(
            f'<polyline fill="none" points="{coords}" stroke="{color}" stroke-width="2.5" stroke-dasharray="{dash}"/>'
        )
        legend.append(f'<span style="color:{color}">{escape(name)}</span>')
    return (
        f"<figure><figcaption>{escape(title)} · 纵轴 {escape(y_label)}</figcaption>"
        f'<svg viewBox="0 0 722 198" role="img" aria-label="{escape(title)}">'
        + "".join(grid + lines)
        + f'</svg><div>{escape(x_label)}</div><div class="legend">{" · ".join(legend)}</div></figure>'
    )


def _html(path, title, description, tables, charts=()):
    sections = []
    for heading, frame in tables:
        content = frame.to_html(
            index=False, border=0, na_rep="—", float_format=lambda x: f"{x:.6f}"
        )
        sections.append(
            f'<section><h2>{escape(heading)}</h2><div class="scroll">{content}</div></section>'
        )
    path.write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(title)}</title>"
        "<style>body{max-width:1050px;margin:auto;padding:24px 16px;background:#f4f7f5;color:#172e28;"
        'font:16px/1.6 "PingFang SC",sans-serif}h1{border-top:4px solid #147463;padding-top:16px}'
        "h2{font-size:20px}.scroll{overflow:auto;background:white;border:1px solid #ccd9d2}"
        "table{border-collapse:collapse;width:100%;white-space:nowrap;font-variant-numeric:tabular-nums}"
        "th,td{padding:6px 12px;border-bottom:1px solid #dbe4df}thead{background:#e4eee8}"
        "section{margin:24px 0}details{margin:24px 0}summary{cursor:pointer}"
        "figure{margin:16px 0;background:white;border:1px solid #ccd9d2;padding:12px}"
        "figure svg{width:100%;height:auto;font-size:12px}figure div{text-align:center}</style>"
        f"<main><h1>{escape(title)}</h1><p>{escape(description)}</p>"
        + "".join(charts)
        + "".join(sections[:-1])
        + "<details><summary>查看明细</summary>"
        + sections[-1]
        + "</details></main></html>",
        encoding="utf-8",
    )


def compare_v5(graph, market, events, output):
    roots = (graph, market, events)
    reference = _config(graph, SOURCES[0], 0.001)
    folds = reference["train.factor_sequence.folds"]
    nodes = [int(f["model_date"]) for f in folds[::2]]
    if len(nodes) != 19 or len(set(nodes)) != len(nodes):
        raise ValueError("Reports must match the frozen 19-node sparse benchmark")
    # Each fitted node covers its own test fold and the following skipped fold.
    dates, daily_nodes = [], []
    for i, fold in enumerate(folds):
        dates.extend(int(day) for day in fold["test"])
        daily_nodes.extend([nodes[i // 2]] * len(fold["test"]))
    expected_nodes = pd.Series(daily_nodes, index=pd.Index(dates, name="date"))
    tests = []
    for root, name in zip(roots, SOURCES, strict=True):
        _campaign(_config(root, name, 0.001), reference)
        tests.append(_test_metrics(root, expected_nodes))

    portfolio = _portfolio(roots, SOURCES)
    context = pd.read_csv(events / "results/factor_context_v5/daily_context.csv")
    comparisons = {}
    for i, j, key in ((0, 1, "market"), (1, 2, "events"), (0, 2, "market_and_events")):
        a, b = tests[i], tests[j]
        paired = pd.DataFrame(
            {
                "model_date": b.model_date,
                "base_raw_ic": a.rankic_rtn,
                "candidate_raw_ic": b.rankic_rtn,
                "delta_ic": b.rankic_rtn - a.rankic_rtn,
            }
        )
        x = pd.DataFrame(_snapshot_rows(roots[i], roots[j], paired))
        low, high = _hac_interval(x.delta_ic)
        summary = pd.DataFrame(
            [
                {
                    "日数": len(x),
                    "原始收益 ΔRank IC": x.delta_ic.mean(),
                    "95% 下限": low,
                    "95% 上限": high,
                    "同权重修正 Δ标签 IC": x.same_weights_delta_std_ic.mean(),
                    "日均换入数量": x.top50_replaced.mean(),
                    "换入股票平均训练标签": x.top50_enter_label_mean.mean(),
                    "换出股票平均训练标签": x.top50_exit_label_mean.mean(),
                }
            ]
        )
        annual = (
            x.assign(year=x.date // 10000)
            .groupby("year", as_index=False)
            .delta_ic.agg(["mean", "count"])
        )
        folds = x.groupby("model_date", as_index=False).delta_ic.agg(["mean", "count"])
        comparisons[key] = x, summary, annual, folds

    output.mkdir(parents=True, exist_ok=False)
    for key, (daily, _, _, _) in comparisons.items():
        daily.to_csv(output / f"{key}_daily.csv", index=False)
    _, market_summary, market_annual, market_folds = comparisons["market"]
    _, event_summary, event_annual, event_folds = comparisons["events"]
    _, total_summary, total_annual, total_folds = comparisons["market_and_events"]
    _html(
        output / "market-vs-graph.html",
        "每日市场状态对照",
        f"{SOURCES[1]} 对比 {SOURCES[0]}。Top 50 换入换出明细使用训练标签，组合绩效采用项目评价器。",
        [
            ("整体差异与 Newey–West 区间（滞后 20 日）", market_summary),
            ("年度原始收益 ΔRank IC", market_annual),
            ("Top 50 组合", portfolio.iloc[:2]),
            ("滚动训练期差异", market_folds),
        ],
        [
            _chart(
                "每日市场的年度增量",
                "年份",
                "原始收益 ΔRank IC",
                {
                    "市场版 − Graph": list(
                        zip(market_annual.year, market_annual["mean"])
                    )
                },
            )
        ],
    )
    _html(
        output / "events-vs-market.html",
        "稀疏市场事件对照",
        f"{SOURCES[2]} 对比 {SOURCES[1]}，并附相对 {SOURCES[0]} 的总体差异。Top 50 换入换出明细使用训练标签。",
        [
            ("事件层增量与 Newey–West 区间（滞后 20 日）", event_summary),
            ("相对 Graph 的总差异", total_summary),
            ("年度事件层 ΔRank IC", event_annual),
            ("年度总 ΔRank IC", total_annual),
            ("事件类型与衰减区间", _event_table(context, comparisons["events"][0])),
            ("Top 50 组合", portfolio),
            (
                "滚动训练期差异",
                event_folds.merge(
                    total_folds, on="model_date", suffixes=("_event", "_total")
                ),
            ),
        ],
        [
            _chart(
                "事件层与完整版的年度增量",
                "年份",
                "原始收益 ΔRank IC",
                {
                    "事件版 − 市场版": list(
                        zip(event_annual.year, event_annual["mean"])
                    ),
                    "事件版 − Graph": list(
                        zip(total_annual.year, total_annual["mean"])
                    ),
                },
            )
        ],
    )
    (output / "model_roots.json").write_text(
        json.dumps(dict(zip(SOURCES, map(str, roots))), indent=2)
    )


def _fit_detail(root, node):
    report = root / "results/factor_context_v4/fit" / f"{node}.0.json"
    best = json.loads(report.read_text())
    path = root / "snapshot/metrics" / f"model.0.{node}.xlsx"
    keys = ["attempt", "redo", "phase", "epoch"]
    with pd.ExcelFile(path) as book:
        train = _unique_index(pd.read_excel(book, "train_epoch_totals"), keys)
        valid = _unique_index(pd.read_excel(book, "valid_epoch_totals"), keys)
        counts = (
            pd.read_excel(book, "train_batch_totals", usecols=keys).groupby(keys).size()
        )
    curve = pd.DataFrame(
        {"TrainIC": train.rankic, "ValidIC": valid.rankic, "batch_steps": counts}
    )
    if curve.isna().any().any():
        raise ValueError(f"Incomplete training metrics for {node}")
    # Workbook order follows training, including retries and phase transitions.
    curve = curve.reindex(train.index)
    curve["cumulative_steps"] = curve.batch_steps.cumsum()
    trial = re.fullmatch(r"Trial(\d+)-(\d+)", best["best_attempt"])
    if trial is None:
        raise ValueError(f"Invalid best_attempt in {report}: {best['best_attempt']!r}")
    point = (int(trial[1]), int(trial[2]), best["phase"], best["best_epoch"])
    if point not in curve.index:
        raise ValueError(f"Best epoch {point} missing from {path}")
    steps = int(cast(Any, curve.at[point, "cumulative_steps"]))
    log = (root / "logs/train.md").read_text()
    reasons = re.findall(rf"@{node}\|Trial\d+-\d+ ModelEpoch#\d+ ([^|]+)\|", log)
    if len(reasons) != 1:
        raise ValueError(f"Expected one training stop reason for {node} in {root}")
    return dict(best, steps_to_peak=steps, stop_reason=reasons[0]), curve.reset_index()


def compare_learning_rate(graph, probe, output):
    from src.res.model.experiment.factor_sequence.repaired_schedule import (
        GRAPH_LR_PROBE_NODES,
    )

    _campaign(
        _config(probe, "factorseq_v4_graph_lr_probe", 0.0003),
        _config(graph, SOURCES[0], 0.001),
    )
    actual = {p.stem for p in (probe / "results/factor_context_v4/fit").glob("*.json")}
    if actual != {f"{node}.0" for node in GRAPH_LR_PROBE_NODES}:
        raise ValueError("The probe must contain exactly the three prescribed nodes")
    summaries, curves = [], []
    for node in GRAPH_LR_PROBE_NODES:
        for root, lr in ((graph, 0.001), (probe, 0.0003)):
            summary, curve = _fit_detail(root, node)
            summaries.append(dict(summary, lr=lr))
            curves.append(curve.assign(model_date=node, lr=lr))
    output.mkdir(parents=True, exist_ok=False)
    history = pd.concat(curves, ignore_index=True)
    history.to_csv(output / "epoch_curves.csv", index=False)
    charts = []
    for node, fold in history.groupby("model_date"):
        charts.append(
            _chart(
                f"{node} 验证曲线",
                "累计优化步数",
                "验证 Rank IC",
                {
                    f"LR {lr:g}": list(zip(run.cumulative_steps, run.ValidIC))
                    for lr, run in fold.groupby("lr", sort=False)
                },
            )
        )
    _html(
        output / "graph-learning-rate.html",
        "Graph 三折学习率小试",
        "同折验证比较 0.001 和 0.0003。优化步数从训练批次记录累计。依据三个验证期的表现与稳定性判断，主要 v5 比较仍使用 0.001。",
        [
            ("所选 epoch、停止原因与步数", pd.DataFrame(summaries)),
            ("逐 epoch 曲线数据", history),
        ],
        charts,
    )


def main():
    from src.proj import PATH, Logger

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode, names in (
        ("market", ("graph", "market", "events")),
        ("learning-rate", ("graph", "probe")),
    ):
        cmd = sub.add_parser(mode)
        cmd.add_argument(
            "--publish",
            action="store_true",
            help="Publish only the generated report directory",
        )
        for name in names:
            cmd.add_argument(
                "--" + name, type=lambda p: Path(p).resolve(), required=True
            )
    args = vars(parser.parse_args())
    publish = args.pop("publish")
    mode = args.pop("mode")
    run_id = uuid4().hex
    output = PATH.rslt_custom / "factor_sequence" / f"v5-{mode}-{run_id}"
    report = compare_v5 if mode == "market" else compare_learning_rate
    report(**args, output=output)
    if publish:
        from src.automation.artifact_exchange import ArtifactSpec
        from src.automation.artifact_exchange.research import publish_safe

        publish_safe(
            "factorseq_v5_compare",
            run_id,
            [
                ArtifactSpec(
                    output.resolve(),
                    category="factor_sequence",
                    experiment=f"v5/{mode}",
                )
            ],
            status="success",
        )
    Logger.stdout(str(output))


if __name__ == "__main__":
    main()
