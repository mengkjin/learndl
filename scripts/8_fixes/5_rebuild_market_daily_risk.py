#!/usr/bin/env python3
# description: Rebuild all historical market daily risk with pandas/Polars verification
# email: true
# mode: shell
# parameters:
#   start:
#     type: int
#     default: null
#     desc: Default earliest supported minute date
#   end:
#     type: int
#     default: null
#     desc: Default latest available confirmed minute date
#   validate_only:
#     type: bool
#     default: false
#     desc: Validate and save candidate without replacing database
from __future__ import annotations
import hashlib
import argparse
import sys
import json
from typing import Any
import os
from pathlib import Path
import shutil
from datetime import datetime, timezone
from uuid import uuid4
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.proj.util.script import ScriptTool
from src.data.update.custom.market_daily_risk import (
    MINUTE_VERSION as VERSION,
    MINUTE_METRICS as METRICS,
    pandas_market_returns,
    polars_minutes,
)


def digest(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_day(inputs, result):
    metrics, polars_returns = polars_minutes(
        inputs["quote"], inputs["val"], inputs["min"], inputs["moneyflow"]
    )
    pandas_returns = pandas_market_returns(
        inputs["quote"], inputs["val"], inputs["min"]
    )
    pd.testing.assert_index_equal(
        pandas_returns.index, polars_returns.index, check_names=False
    )
    np.testing.assert_allclose(
        pandas_returns.to_numpy(),
        polars_returns.to_numpy(),
        rtol=1e-9,
        atol=1e-12,
        equal_nan=True,
    )
    records = []
    for metric in METRICS:
        a, b = float(result[metric]), metrics[metric]
        if not np.isclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True):
            raise ValueError(f"{metric}: pandas={a}, polars={b}")
        records.append(
            dict(metric=metric, pandas=a, polars=b, absolute_difference=abs(a - b))
        )
    if not np.isfinite(result[list(METRICS)].to_numpy(dtype=float)).all():
        raise ValueError(
            "Minute metrics are nonfinite, even though both implementations agree"
        )
    return records


def replace_checked(target, staged, expected_digest):
    """Do not replace a file changed by the daily updater during reconstruction."""
    if digest(target) != expected_digest:
        raise RuntimeError(
            "Risk source changed during rebuild; rerun in an idle maintenance window"
        )
    os.replace(staged, target)


def rebuild(start=None, end=None, validate_only=False):
    from src.proj import Logger
    from src.proj import DB, CALENDAR, PATH, Save, Load
    from src.data.update.custom.market_daily_risk import (
        _get_inputs,
        calc_market_daily_risk,
        MarketDailyRiskUpdater,
    )
    from src.automation.artifact_exchange.core import ArtifactSpec
    from src.automation.artifact_exchange.research import register

    run_id = (
        "market_risk_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + uuid4().hex[:8]
    )
    folder = PATH.result / "market_risk_rebuild" / run_id
    reports = folder / "reports"
    reports.mkdir(parents=True, exist_ok=False)
    target = DB.path("market_daily", "risk")
    before = digest(target)
    old = DB.load("market_daily", "risk")
    if not old.empty and ("date" not in old or old.date.duplicated().any()):
        raise ValueError(
            "Existing risk dates missing/duplicated; refusing ambiguous replacement"
        )
    report: dict[str, Any] = dict(
        version=VERSION,
        run_id=run_id,
        status="running",
        source=str(target),
        old_sha256=before,
        validate_only=validate_only,
        completed_days=0,
        rtol=1e-9,
        atol=1e-12,
    )
    staged = None
    comparisons = []
    try:
        start = int(
            start
            if start is not None
            else min(
                MarketDailyRiskUpdater.START_DATE,
                int(old.date.min())
                if not old.empty
                else MarketDailyRiskUpdater.START_DATE,
            )
        )
        end = int(
            end
            if end is not None
            else min(
                int(CALENDAR.updated()), DB.max_date("trade_ts", "5min", use_alt=True)
            )
        )
        dates = [int(d) for d in CALENDAR.range(start, end, "td")]
        if not dates:
            raise ValueError("No historical trading dates")
        if not validate_only and not old.empty and int(old.date.max()) > end:
            report["preserved_later_dates"] = int((old.date > end).sum())
        report.update(start=start, end=end, requested_days=len(dates))
        if target.exists():
            backup = folder / "backup" / target.name
            backup.parent.mkdir()
            shutil.copy2(target, backup)
            if digest(backup) != before:
                raise RuntimeError("Source changed while taking the backup")
            report["backup"] = str(backup)
        values = []
        for date in dates:
            inputs = _get_inputs(date)
            if any(frame.empty for frame in inputs.values()):
                raise ValueError(f"{date}: required source missing; database unchanged")
            computed = calc_market_daily_risk(date, inputs=inputs)
            checked = validate_day(inputs, computed.iloc[0])
            if not np.isfinite(computed.drop(columns="date").to_numpy()).all():
                raise ValueError(f"{date}: nonfinite daily result; database unchanged")
            values.append(computed)
            comparisons.extend(dict(date=date, **r) for r in checked)
            report["completed_days"] = len(values)
            report["last_date"] = date
            if len(values) % 50 == 0:
                Logger.stdout(
                    f"[market-risk] verified {len(values)}/{len(dates)} through {date}",
                    flush=True,
                )
                (reports / "summary.json").write_text(json.dumps(report, indent=2))
        new = pd.concat(values, ignore_index=True)
        baseline = (
            old.set_index("date").reindex(new.date)
            if not old.empty
            else pd.DataFrame(index=new.date)
        )
        changes = pd.DataFrame(new.set_index("date")) - baseline
        changes.to_csv(reports / "historical_differences.csv")
        pd.DataFrame(comparisons).to_csv(
            reports / "pandas_polars_comparison.csv", index=False
        )
        # Keep dates outside an explicitly requested interval. Backups/candidate
        # data are private; only diagnostic reports are registered for delivery.
        result = (
            pd.concat(
                [old.loc[~old.date.isin(dates)] if not old.empty else old, new],
                ignore_index=True,
            )
            .sort_values("date")
            .reset_index(drop=True)
        )
        private = folder / "candidate"
        private.mkdir()
        Save.df(result, private / target.name, overwrite=True, async_save=False)
        if not validate_only:
            target.parent.mkdir(parents=True, exist_ok=True)
            staged = target.with_name("." + target.stem + "." + run_id + target.suffix)
            Save.df(result, staged, overwrite=True, async_save=False)
            restored = Load.df(staged)
            pd.testing.assert_frame_equal(
                result, restored, check_dtype=False, rtol=1e-12, atol=1e-12
            )
            replace_checked(target, staged, before)
            report["new_sha256"] = digest(target)
        report["status"] = "validated" if validate_only else "replaced"
        return folder
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        pd.DataFrame(comparisons).to_csv(
            reports / "pandas_polars_comparison.csv", index=False
        )
        if staged is not None:
            staged.unlink(missing_ok=True)
        (reports / "summary.json").write_text(json.dumps(report, indent=2, default=str))
        register(
            ArtifactSpec(
                reports, category="market_risk", experiment=VERSION, run_id=run_id
            ),
            "market_risk_rebuild",
            run_id,
            report["status"],
        )
        Logger.stdout(f"[market-risk] {report['status']}: {reports}", flush=True)


@ScriptTool("rebuild_market_daily_risk", markdown_catcher=True)
def main(start=None, end=None, validate_only=False):
    return rebuild(start, end, validate_only)


if __name__ == "__main__":

    def boolean(value):
        if str(value).lower() not in ("true", "false"):
            raise argparse.ArgumentTypeError("Expected true or false")
        return str(value).lower() == "true"

    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int)
    p.add_argument("--end", type=int)
    p.add_argument(
        "--validate-only",
        "--validate_only",
        type=boolean,
        nargs="?",
        const=True,
        default=False,
    )
    p.add_argument("--task_id", help=argparse.SUPPRESS)
    p.add_argument("--source", help=argparse.SUPPRESS)
    p.add_argument("--email", type=boolean, help=argparse.SUPPRESS)
    main(**{k: v for k, v in vars(p.parse_args()).items() if v is not None})
