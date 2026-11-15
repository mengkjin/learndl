"""Server audit entry: daily bounded stock reads, independent of training loaders."""

from __future__ import annotations
import json
import hashlib
from typing import Any
from pathlib import Path
import subprocess
import traceback
from datetime import datetime, timezone
from uuid import uuid4
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd
from .context_audit import (
    FACTORS,
    EVENTS,
    RISK_NAMES,
    PERF_NAMES,
    CoverageState,
    dictionary,
    candidate_market,
    context_quality,
    event_summary,
    strata,
)
from .numeric import group_metrics
from .bundle import file_digest
from .risk_audit import source_quality, RISK_NOTES

__all__ = [
    'write_json',
    'append_csv',
    'read_factor',
    'run_server',
    'deliver',
    'new_run_id',
]


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n"
    )


def append_csv(path, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if not frame.empty:
        frame.to_csv(path, mode="a", header=not Path(path).exists(), index=False)


def read_factor(calc, name, date, universe, db, first):
    """Missing files are not normal stock NaNs; malformed identifiers are fatal to source validity."""
    record = dict(date=int(date), factor=name, status="available", error="", sources="")
    result = pd.Series(np.nan, index=universe, name=name)
    try:
        dbs = (
            calc.full_dbs()
            if hasattr(calc, "full_dbs")
            else [dict(src=calc.db_src, key=calc.db_key)]
        )
        paths = [db.path(d["src"], d["key"], int(date)) for d in dbs]
        present = [p for p in paths if p.exists()]
        record["sources"] = json.dumps({str(p): file_digest(p) for p in present})
        if not present:
            record["status"] = (
                "not_started" if first is None or date < first else "file_missing"
            )
            return result, record
        frame = calc.Loads([int(date)], normalize=False, closest=False)
        if frame.empty:
            record["status"] = "empty_file"
            return result, record
        if (
            not {"secid", "date", name}.issubset(frame.columns)
            or frame[["secid", "date"]].isna().any().any()
        ):
            raise ValueError("Missing identifiers/feature")
        if not frame.date.eq(date).all() or frame.secid.duplicated().any():
            raise ValueError("Duplicate identifiers or wrong file date")
        result = pd.to_numeric(frame.set_index("secid")[name], errors="raise").reindex(
            universe
        )
        result = result.where(np.isfinite(result))
    except Exception as exc:
        record.update(status="corrupt", error=str(exc))
    return result, record


def run_server(folder, start=20110101, end=None, model_roots=None):
    """Always leave partial tables and a non-success summary on failure."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = dict(
        run_id=folder.name,
        status="running",
        requested_start=start,
        errors=[],
        limitations=[
            "Daily factor file validity cannot recover upstream fills or historical disclosure revisions.",
            "Close-time availability assumed; vendor arrival timestamps are not available.",
            "Risk corrected references are diagnostic only; source data are not overwritten.",
        ],
    )
    write_json(folder / "feature_dictionary.json", dictionary())
    write_json(folder / "risk_methodology.json", RISK_NOTES)
    append_csv(
        folder / "context_availability.csv",
        [
            dict(
                block=item["block"],
                feature=item["name"],
                first_valid=None,
                last_valid=None,
                valid_rate=None,
            )
            for item in dictionary()["features"]
        ],
    )
    try:
        from src.proj import CALENDAR, DB
        from src.data import DATAVENDOR
        from src.res.factor.calculator import StockFactorHierarchy
        from src.data.update.custom.market_daily_risk import (
            _get_inputs,
            calc_market_daily_risk,
        )

        end = int(end or CALENDAR.updated())
        if end > int(CALENDAR.updated()):
            raise ValueError("End date exceeds latest confirmed project date")
        dates = np.asarray(
            CALENDAR.range(int(CALENDAR.td(start, -252)), end, "td"), dtype=int
        )
        if not len(dates):
            raise ValueError("No trading calendar")
        report["actual_end"] = end
        report["code_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        report["audit_code_sha256"] = {
            p.name: file_digest(p) for p in Path(__file__).parent.glob("*.py")
        }
        desc = DATAVENDOR.INFO.get_list_dt(
            offset=91, keep_columns=["list_dt", "delist_dt"]
        )
        if desc.empty or desc.index.has_duplicates:
            raise ValueError("Independent security master unavailable/duplicate")
        universe = desc.index
        calcs = {
            f.lstrip("+-"): StockFactorHierarchy.get_factor(f.lstrip("+-"))
            for f in FACTORS
        }
        first = {
            n: (int(c.stored_dates().min) if len(c.stored_dates()) else None)
            for n, c in calcs.items()
        }
        report["factor_first_dates"] = first
        if any(d is None for d in first.values()):
            raise ValueError(
                "Required factor sources never stored: "
                + ",".join(n for n, d in first.items() if d is None)
            )
        market = pd.DataFrame(index=dates)
        # Read full small event histories to retain last confirmed event before the requested start.
        event_frames = {}
        for event in EVENTS:
            frame = StockFactorHierarchy.get_factor(event).Load(None)
            if frame.empty or frame.date.duplicated().any():
                raise ValueError(f"Unavailable/duplicate event history: {event}")
            event_frames[event] = frame.set_index("date")[event]
        risk = DB.load("market_daily", "risk")
        if risk.empty or "date" not in risk or risk.date.duplicated().any():
            raise ValueError("Unavailable/duplicate market daily risk history")
        quote = DB.load("index_daily_ts", "000985.CSI")
        if quote.empty or quote.trade_date.duplicated().any():
            raise ValueError("Unavailable/duplicate market index history")
        # Include all known earlier event dates, not merely a 252-day warm-up.
        event_start = min(int(x.index.min()) for x in event_frames.values())
        md = np.asarray(
            CALENDAR.range(min(event_start, int(dates[0])), end, "td"), dtype=int
        )
        market = market.reindex(md)
        for event, x in event_frames.items():
            market[event] = x.reindex(md)
        market["market_return"] = (
            quote.set_index("trade_date").pct_chg.reindex(md) / 100
        )
        for n in RISK_NAMES:
            market[n] = risk.set_index("date")[n].reindex(md)
        source_frames = {
            "security_master": desc,
            "market_index": quote,
            "market_risk": risk,
            **event_frames,
        }
        write_json(
            folder / "source_fingerprints.json",
            {
                name: dict(
                    rows=len(frame),
                    sha256=hashlib.sha256(
                        pd.util.hash_pandas_object(frame, index=True)
                        .to_numpy()
                        .tobytes()
                    ).hexdigest(),
                )
                for name, frame in source_frames.items()
            },
        )
        report["risk_first_source_dates"] = {
            n: risk.set_index("date")[n]
            .replace([np.inf, -np.inf], np.nan)
            .first_valid_index()
            for n in RISK_NAMES
        }
        view = candidate_market(market)
        view.loc[start:end].to_csv(
            folder / "factormarket_values.csv", index_label="date"
        )
        view.loc[start:end].replace([np.inf, -np.inf], np.nan).notna().to_csv(
            folder / "factormarket_validity.csv", index_label="date"
        )
        context_quality(view).loc[start:end].to_csv(
            folder / "factormarket_windows.csv", index_label="date"
        )
        event_summary(market.loc[start:end]).to_csv(
            folder / "events_annual.csv", index=False
        )
        report["market_flat_days"] = int(
            market.loc[start:end, "market_return"].eq(0).sum()
        )
        state = CoverageState()
        prev = None
        perf = []
        horizon = []
        any_valid = {n: 0 for n in calcs}
        source_errors = 0
        for pos, date in enumerate(dates):
            date = int(date)
            eligible = universe[(desc.list_dt <= date) & (desc.delist_dt > date)]
            values = pd.DataFrame(index=universe)
            statuses = []
            for n, c in calcs.items():
                values[n], record = read_factor(c, n, date, universe, DB, first[n])
                statuses.append(record)
                if date >= start:
                    any_valid[n] += int(values.loc[eligible, n].notna().sum())
                    source_errors += int(
                        record["status"] in ("corrupt", "file_missing", "empty_file")
                    )
            row, stats, complete = state.update(date, values, eligible)
            if date >= start:
                append_csv(folder / "coverage_daily.csv", [row])
                byname = {r["factor"]: r for r in statuses}
                for s in stats:
                    s.update(source_status=byname[s["factor"].lstrip("+-")]["status"])
                append_csv(folder / "coverage_factors.csv", stats)
                append_csv(folder / "source_status.csv", statuses)
            # Previous day factor values are direction-adjusted and eligibility-masked;
            # return is same adjusted close series used by the original builder.
            p = DATAVENDOR.TRADE.get_specific_data(
                int(CALENDAR.td(date, -1)), date, "adj", "close", prev=False, pivot=True
            )
            returns = (
                p.sort_index()
                .pct_change(fill_method=None)
                .reindex([date])
                .reindex(columns=universe)
                .iloc[0]
            )
            metrics = []
            for signed in FACTORS:
                n = signed.lstrip("+-")
                m = (
                    group_metrics(prev[n].to_numpy(), returns.to_numpy(), 50)
                    if prev is not None
                    else np.full(4, np.nan)
                )
                metrics.extend([m[0], m[3] * 100])
                if date >= start:
                    paired = (
                        np.isfinite(prev[n].to_numpy())
                        & np.isfinite(returns.to_numpy())
                        if prev is not None
                        else np.zeros(len(universe), dtype=bool)
                    )
                    reason = (
                        "available"
                        if np.isfinite(m[[0, 3]]).all()
                        else "insufficient_pairs"
                        if paired.sum() < 50
                        else "constant_or_missing_extreme_group"
                    )
                    append_csv(
                        folder / "factorperf_reasons.csv",
                        [
                            dict(
                                date=date,
                                factor=n,
                                paired_stocks=int(paired.sum()),
                                reason=reason,
                                ic_valid=bool(np.isfinite(m[0])),
                                spread_valid=bool(np.isfinite(m[3])),
                            )
                        ],
                    )
            perf.append([date] + metrics)
            if date >= start:
                append_csv(
                    folder / "factorperf_values.csv",
                    [dict(zip(["date", *PERF_NAMES], [date] + metrics))],
                )
            if model_roots and "contextual" in model_roots:
                recorded = (
                    Path(model_roots["contextual"])
                    / "results/factor_residual"
                    / f"{date}.feather"
                )
                if recorded.exists():
                    label = (
                        pd.read_feather(recorded)
                        .set_index("secid")
                        .label.reindex(universe)
                    )
                    targets = {}
                    for signed in FACTORS:
                        n = signed.lstrip("+-")
                        factor = values[n].where(values.index.isin(eligible)) * (
                            -1 if signed.startswith("-") else 1
                        )
                        targets[n] = group_metrics(
                            factor.to_numpy(), label.to_numpy(), 50
                        )[0]
                    horizon.append(dict(date=date, **targets))
            prev = values.copy()
            prev.loc[~prev.index.isin(eligible)] = np.nan
            for signed in FACTORS:
                if signed.startswith("-"):
                    prev[signed[1:]] *= -1
            if date >= start:
                try:
                    q = DB.load("trade_ts", "day", date).set_index("secid")
                    val = DB.load("trade_ts", "day_val", date).set_index("secid")
                    industry = DATAVENDOR.INFO.get_indus(date).indus
                    size = val.circ_mv.reindex(eligible)
                    size_group = pd.cut(
                        size.rank(pct=True),
                        [0, 0.2, 0.8, 1],
                        labels=["small", "middle", "large"],
                    ).astype(object)
                    tradable = pd.Series(
                        np.where(q.amount > 0, "traded", "not_traded"), index=q.index
                    )
                    rows = (
                        strata(date, complete, industry, "industry")
                        + strata(date, complete, size_group, "size")
                        + strata(date, complete, tradable, "traded_proxy")
                    )
                    append_csv(folder / "coverage_strata.csv", rows)
                except Exception as exc:
                    report["errors"].append(
                        dict(date=date, stage="strata", error=str(exc))
                    )
                try:
                    inputs = _get_inputs(date)
                    for source, raw_input in inputs.items():
                        numeric = raw_input.select_dtypes(include="number")
                        if not numeric.empty:
                            append_csv(
                                folder / "risk_source_missing.csv",
                                [
                                    dict(
                                        date=date,
                                        source=source,
                                        field=c,
                                        rows=len(numeric),
                                        invalid=int((~np.isfinite(numeric[c])).sum()),
                                    )
                                    for c in numeric
                                ],
                            )
                    rows, minutes = source_quality(inputs)
                    production = calc_market_daily_risk(date).iloc[0]
                    for r in rows:
                        r.update(
                            date=date,
                            production=production[r["metric"]],
                            stored=risk.set_index("date")[r["metric"]].get(
                                date, np.nan
                            ),
                        )
                        r["difference"] = r["reference"] - r["production"]
                    append_csv(folder / "risk_quality.csv", rows)
                    # Summaries only: no full minute panel retained.
                    append_csv(
                        folder / "risk_minutes.csv",
                        [
                            dict(
                                date=date,
                                min_coverage=minutes.valid_weight_coverage.min(),
                                mean_coverage=minutes.valid_weight_coverage.mean(),
                            )
                        ],
                    )
                except Exception as exc:
                    append_csv(
                        folder / "risk_errors.csv", [dict(date=date, error=str(exc))]
                    )
            if pos % 50 == 0:
                report["last_completed_date"] = date
                write_json(folder / "summary.json", report)
                print(f"[context audit] {date} ({pos + 1}/{len(dates)})", flush=True)
        perf = pd.DataFrame(perf, columns=["date", *PERF_NAMES]).set_index("date")
        perf.loc[start:end].to_csv(folder / "factorperf_values.csv", index_label="date")
        perf.loc[start:end].replace([np.inf, -np.inf], np.nan).notna().to_csv(
            folder / "factorperf_validity.csv", index_label="date"
        )
        context_quality(perf).loc[start:end].to_csv(
            folder / "factorperf_windows.csv", index_label="date"
        )
        joint = pd.concat([perf, view.reindex(perf.index)], axis=1)
        context_quality(joint).loc[start:end].to_csv(
            folder / "context_joint_windows.csv", index_label="date"
        )
        maturity = pd.DataFrame(index=market.index)
        for n in RISK_NAMES:
            clean = market[n].replace([np.inf, -np.inf], np.nan)
            maturity[n + "__source_valid"] = clean.notna()
            maturity[n + "__past_count"] = (
                clean.shift(1).notna().rolling(120, min_periods=1).sum()
            )
            maturity[n + "__normalized_valid"] = view[n].notna()
        maturity.loc[start:end].to_csv(folder / "risk_maturity.csv", index_label="date")
        if horizon:
            from .context_diagnostics import horizon_lags

            horizon_lags(
                perf,
                pd.DataFrame(horizon).set_index("date"),
                folder / "horizon_lags.csv",
            )
        else:
            report["limitations"].append(
                "No saved contextual daily labels: horizon lag analysis unavailable"
            )
        availability = []
        for block, frame in [("factorperf", perf), ("factormarket", view)]:
            for n in frame:
                x = frame[n].replace([np.inf, -np.inf], np.nan)
                availability.append(
                    dict(
                        block=block,
                        feature=n,
                        first_valid=x.first_valid_index(),
                        last_valid=x.last_valid_index(),
                        valid_rate=x.loc[start:end].notna().mean(),
                    )
                )
        pd.DataFrame(availability).to_csv(
            folder / "context_availability.csv", index=False
        )
        report["unavailable_factors"] = [n for n, v in any_valid.items() if v == 0]
        report["source_error_count"] = source_errors
        report["risk_error_days"] = (
            len(pd.read_csv(folder / "risk_errors.csv"))
            if (folder / "risk_errors.csv").exists()
            else 0
        )
        report["candidate_complete_120"] = {
            b: int(context_quality(f).loc[start:end].complete_120.eq(1).sum())
            for b, f in [("factorperf", perf), ("factormarket", view)]
        }
        report["status"] = (
            "incomplete"
            if (
                source_errors
                or report["errors"]
                or report["unavailable_factors"]
                or report["risk_error_days"]
            )
            else "completed"
        )
        report["training_gate"] = (
            "not_approved: historical risk revision and source availability require review"
        )
    except Exception:
        report["status"] = "failed"
        report["errors"].append(dict(stage="audit", error=traceback.format_exc()))
    finally:
        try:
            from .context_diagnostics import diagnose_roots

            report["diagnostics"] = diagnose_roots(
                model_roots or {}, folder / "diagnostics"
            )
        except Exception:
            report["diagnostics"] = dict(status="failed", error=traceback.format_exc())
        write_json(folder / "summary.json", report)
    return report


def deliver(folder, recipient=None):
    """Normal task publication precedes mail; ZIP always contains the daily CSVs."""
    from src.automation.artifact_exchange.core import ArtifactSpec
    from src.automation.artifact_exchange.research import publish_safe, defer_mail

    folder = Path(folder)
    report = json.loads((folder / "summary.json").read_text())
    from .context_diagnostics import recover_reports, MODES
    from src.proj import PATH

    recovered = recover_reports(
        {m: PATH.model_nn / f"factorseq_residual@factorseq_v2_{m}@2" for m in MODES},
        folder.name,
    )
    write_json(folder / "report_recovery.json", recovered)
    package = folder / "daily_results.zip"
    with ZipFile(package, "w", ZIP_DEFLATED) as z:
        for path in sorted(folder.rglob("*")):
            if (
                path.is_file()
                and path.suffix in (".csv", ".json", ".md")
                and "mail" not in path.relative_to(folder).parts
            ):
                z.write(path, path.relative_to(folder))
    transfers = publish_safe(
        "factor_sequence_audit",
        folder.name,
        [
            ArtifactSpec(
                folder,
                category="factor_sequence",
                experiment="contextual_audit",
                run_id=folder.name,
                members=(
                    "summary.json",
                    "feature_dictionary.json",
                    "daily_results.zip",
                ),
            )
        ],
        report["status"],
    )

    def mail():
        from src.proj import MACHINE
        from src.proj.util.web.task_mail import TaskResultMail

        if not MACHINE.emailable:
            result: dict[str, Any] = dict(
                status="skipped", reason="machine email disabled"
            )
        else:
            try:
                ok = TaskResultMail(folder / "mail")(
                    "Contextual audit [" + report["status"] + "] " + folder.name,
                    json.dumps(report, ensure_ascii=False, indent=2)
                    + "\nPublication: "
                    + json.dumps(
                        [
                            {k: r.get(k) for k in ("id", "status", "reason", "error")}
                            for r in transfers
                        ],
                        default=str,
                    ),
                    recipient=recipient,
                    attachments=[folder / "summary.json", package],
                )
                result = dict(status="sent" if ok else "failed")
            except Exception:
                result = dict(
                    status="failed", error=traceback.format_exc()
                )
        write_json(folder / "mail_status.json", result)
        return result

    return defer_mail(mail) or mail()


def new_run_id():
    return (
        "context_audit_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + uuid4().hex[:8]
    )
