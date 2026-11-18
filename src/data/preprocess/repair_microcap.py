"""Rebuild microcap returns and market inputs without training or model mutation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from uuid import uuid4
from typing import Any

import numpy as np
import pandas as pd

from src.proj import CALENDAR, DB, PATH, Logger, Save, Load
from src.data.loader import DATAVENDOR
from src.data.util import DataBlock, INFO
from src.data.update.custom.custom_index import MicroCap_400
from .processors.market import PrePro_market

__all__ = ["rebuild_microcap"]


def digest(path):
    path = Path(path)
    if not path.exists():
        return None
    files = sorted(path.rglob("*")) if path.is_dir() else [path]
    result = hashlib.sha256()
    for file in files:
        if file.is_file():
            result.update(
                str(file.relative_to(path) if path.is_dir() else file.name).encode()
            )
            with file.open("rb") as stream:
                result.update(hashlib.file_digest(stream, "sha256").digest())
    return result.hexdigest()


def checked_return(weights, returns, date):
    """Never interpret an empty portfolio or unmatched returns as zero performance."""
    if weights.empty or weights.secid.duplicated().any():
        raise ValueError(f"{date}: empty or duplicate microcap holdings")
    if returns.index.name == "secid":
        returns = returns.reset_index()
    if (
        not {"secid", "ret"}.issubset(returns.columns)
        or returns.secid.duplicated().any()
    ):
        raise ValueError(f"{date}: invalid return schema or duplicate securities")
    joined = weights.merge(
        returns[["secid", "ret"]], on="secid", how="left", validate="one_to_one"
    )
    if not np.isfinite(joined[["weight", "ret"]].to_numpy()).all():
        raise ValueError(
            f"{date}: missing/nonfinite holdings returns; repair price inputs first"
        )
    if not np.isclose(joined.weight.sum(), 1.0) or (joined.weight <= 0).any():
        raise ValueError(f"{date}: invalid portfolio weights")
    return float((joined.weight * joined.ret).sum() * 100)


def check_prices(date):
    day = DB.load("trade_ts", "day", date)
    required = {"secid", "close", "adjfactor"}
    if day.empty or not required.issubset(day.columns) or day.secid.duplicated().any():
        raise ValueError(f"{date}: missing or malformed daily quotes")
    # Match calc_adjprice's listed universe; raw day files can contain names
    # that the adjusted-price writer intentionally excludes.
    day = day.loc[day.secid.isin(INFO.get_secid(int(date)))].set_index("secid")
    if day.empty:
        raise ValueError(f"{date}: no listed securities in daily quotes")
    close = day.close.astype(float)
    # Both calc_adjprice and the raw-quote return fallback use 1 for a missing
    # adjustment factor. Validate against that same production convention.
    factor = day.adjfactor.astype(float).fillna(1.0)
    raw = close * factor
    invalid = (~np.isfinite(close) | (close <= 0) |
               ~np.isfinite(factor) | (factor <= 0) | ~np.isfinite(raw) | (raw <= 0))
    if invalid.any():
        examples = "; ".join(
            f"secid={secid}, close={close.loc[secid]:.8g}, "
            f"adjfactor={day.adjfactor.loc[secid]}, effective_adjfactor={factor.loc[secid]:.8g}"
            for secid in day.index[invalid][:5]
        )
        raise ValueError(
            f"{date}: nonfinite/nonpositive adjusted close in raw quotes; "
            f"{int(invalid.sum())}/{len(day)} listed quotes invalid. {examples}"
        )
    adj = DB.load("trade_ts", "adjprice", date)
    if not adj.empty:
        if not {"secid", "close"}.issubset(adj.columns) or adj.secid.duplicated().any():
            raise ValueError(f"{date}: invalid adjprice schema")
        actual = adj.set_index("secid").close.reindex(raw.index)
        mismatch = ~np.isclose(actual, raw, rtol=1e-5, atol=1e-6)
        if mismatch.any():
            examples = "; ".join(
                f"secid={secid}, stored={actual.loc[secid]:.8g}, expected={raw.loc[secid]:.8g}"
                for secid in raw.index[mismatch][:5]
            )
            raise ValueError(
                f"{date}: stale adjprice; {int(mismatch.sum())}/{len(raw)} listed quotes differ "
                f"({int(actual.isna().sum())} missing). {examples}. "
                "Rebuild this date from valid daily quotes first"
            )


def market_cache_paths(root):
    """Select only legacy ModuleData caches explicitly containing the market input."""
    meta = root / "cache_metadata.json"
    if not meta.exists():
        return []
    records = json.loads(meta.read_text())
    result = []
    for key, record in records.items():
        types = record.get("data_type_list", [])
        if isinstance(types, str):
            types = types.split("+")
        if record.get("type") == "module_data" and "market" in types:
            folder = (root / key).resolve()
            if not folder.is_relative_to(root.resolve()):
                raise ValueError(f"Invalid cache key: {key}")
            if (folder / "data.pt").exists():
                result.append(folder / "data.pt")
    return result


def save_block(block, path):
    if path.suffix == ".mmap":
        Save.mmap(
            block.values,
            path,
            index={"date": block.date, "secid": block.secid, "feature": block.feature},
        )
    elif path.suffix == ".feather":
        Save.df(block.to_dataframe(), path)
    elif path.suffix == ".pt":
        from src.data.util.classes.data_block import save_dict

        save_dict(
            dict(
                values=block.values,
                date=block.date,
                secid=block.secid,
                feature=block.feature,
            ),
            path,
        )
    else:
        raise ValueError(f"Unsupported market dump: {path}")
    # Verify the serialized format as well as the in-memory candidate.
    if path.suffix == ".mmap":
        data = Load.mmap(path, values=True, index=True)
        restored = DataBlock(
            data["values"],
            data["index"]["secid"],
            data["index"]["date"],
            data["index"]["feature"],
        )
    elif path.suffix == ".pt":
        from src.data.util.classes.data_block import load_dict

        restored = DataBlock(**load_dict(path))
    else:
        restored = DataBlock.from_pandas(pd.read_feather(path))
    pd.testing.assert_frame_equal(block.to_dataframe(), restored.to_dataframe())


def copy_path(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)


def remove_path(path):
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def install(replacements, baseline, folder):
    """Persist rollback instructions before changing any output. Requires idle writers."""
    entries = []
    for i, (target, staged) in enumerate(replacements):
        if digest(target) != baseline[target]:
            raise RuntimeError(
                f"Output changed during repair: {target}; rerun while idle"
            )
        backup = folder / "backups" / str(i)
        if target.exists():
            copy_path(target, backup)
        entries.append(
            dict(
                target=str(target),
                backup=str(backup),
                existed=target.exists(),
                staged=str(staged) if staged else None,
            )
        )
    journal = folder / "transaction.json"
    journal.write_text(json.dumps(dict(status="installing", entries=entries), indent=2))
    touched = []
    try:
        for (target, staged), entry in zip(replacements, entries):
            if digest(target) != baseline[target]:
                raise RuntimeError(f"Concurrent output change: {target}")
            temporary = target.with_name(target.name + ".repair-" + uuid4().hex)
            try:
                if staged is not None:
                    copy_path(staged, temporary)
                touched.append(entry)
                if target.is_dir() or staged is None:
                    remove_path(target)
                if staged is not None:
                    os.replace(temporary, target)
            finally:
                remove_path(temporary)
        journal.write_text(
            json.dumps(dict(status="complete", entries=entries), indent=2)
        )
    except BaseException:
        for entry in reversed(touched):
            target = Path(entry["target"])
            remove_path(target)
            if entry["existed"]:
                copy_path(Path(entry["backup"]), target)
        journal.write_text(
            json.dumps(dict(status="rolled_back", entries=entries), indent=2)
        )
        raise


def rebuild_microcap(start: int | np.integer = MicroCap_400.START_DATE, end=None, validate_only=False):
    from src.automation.artifact_exchange.core import ArtifactSpec
    from src.automation.artifact_exchange.research import register
    from src.proj.util.functional.parallel import serial_execution

    start = int(start)
    end = int(CALENDAR.updated() if end is None else end)
    if start < MicroCap_400.START_DATE or start > end or end > CALENDAR.updated():
        raise ValueError(
            f"Require {MicroCap_400.START_DATE} <= start <= end <= latest confirmed trading date"
        )
    run_id = "microcap_" + uuid4().hex
    folder = PATH.result / "microcap_repair" / run_id
    reports = folder / "reports"
    reports.mkdir(parents=True)
    stages = folder / "staged"
    stages.mkdir()
    report: dict[str, Any] = dict(
        status="running", start=start, end=end, models_retrained=False
    )
    try:
        target = DB.path("index_daily_custom", "microcap_400")
        outputs = [
            target,
            *(
                DataBlock.path_preprocess("market", frame)
                for frame in ("fit", "predict")
            ),
        ]
        caches = market_cache_paths(PATH.datacache)
        baseline = {p: digest(p) for p in outputs + caches}
        old = DB.load("index_daily_custom", "microcap_400")
        dates = CALENDAR.range(start, end, "td")
        rows = []
        checked = set()
        with serial_execution():
            DATAVENDOR.TRADE.clear_all()
            for date in dates:
                date = int(date)
                previous = CALENDAR.td(date, -1).as_int()
                Logger.stdout(
                    f"[microcap repair] date={date}, holdings_date={previous}",
                    vb_level=0,
                )
                for d in (previous, date):
                    if d not in checked:
                        check_prices(d)
                        checked.add(d)
                index = MicroCap_400()
                weights = index.rebalance_portfolio(previous)
                returns = DATAVENDOR.get_quote_ret(previous, date)
                value = checked_return(weights, returns, date)
                production = index.index_return(date) * 100
                if not np.isclose(value, production, rtol=1e-7, atol=1e-7):
                    raise ValueError(
                        f"{date}: production return {production} differs from checked return {value}"
                    )
                rows.append(dict(trade_date=date, pct_chg=value, holdings=len(weights)))
                DATAVENDOR.TRADE.clear_all()
            rebuilt = pd.DataFrame(rows)
            if rebuilt.empty:
                raise ValueError("No trading dates to rebuild")
            rebuilt.to_csv(reports / "recomputed.csv", index=False)
            candidate = rebuilt[["trade_date", "pct_chg"]]
            if not old.empty:
                candidate = (
                    pd.concat(
                        [old.loc[~old.trade_date.isin(candidate.trade_date)], candidate]
                    )
                    .sort_values("trade_date")
                    .reset_index(drop=True)
                )
                old.merge(rebuilt, on="trade_date", suffixes=("_old", "_new")).to_csv(
                    reports / "comparison.csv", index=False
                )
            index_stage = stages / target.name
            Save.df(candidate, index_stage)
            pd.testing.assert_frame_equal(
                Load.df(index_stage), candidate.reset_index(drop=True)
            )
            replacements: list[tuple[Path, Path | None]] = [(target, index_stage)]

            class RepairMarket(PrePro_market):
                @classmethod
                def _load_index_logret(cls, db_src, db_key):
                    if (db_src, db_key) == ("index_daily_custom", "microcap_400"):
                        return pd.Series(
                            np.log1p(candidate.pct_chg.to_numpy() / 100),
                            index=candidate.trade_date.to_numpy(),
                        )
                    return super()._load_index_logret(db_src, db_key)

            for frame, path in zip(("fit", "predict"), outputs[1:]):
                proc = RepairMarket(frame=frame)
                # Full reconstruction avoids leaving zero returns in older dump segments.
                block = proc.pre_process()
                required_features = set(PrePro_market.RISK_FEATURES) | set(
                    PrePro_market.INDEX_SOURCES
                )
                if block.empty or not required_features.issubset(block.feature):
                    raise ValueError(
                        f"{frame}: incomplete market inputs; restore risk and benchmark sources first"
                    )
                values = block.to_dataframe().reset_index()
                expected = candidate.set_index("trade_date").pct_chg.reindex(
                    values.date
                )
                valid = expected.notna().to_numpy()
                np.testing.assert_allclose(
                    values.loc[valid, "logret_microcap_400"],
                    np.log1p(expected.to_numpy()[valid] / 100),
                    rtol=1e-5,
                    atol=1e-7,
                )
                stage = stages / (frame + path.suffix)
                save_block(block, stage)
                replacements.append((path, stage))
            replacements.extend((p, None) for p in caches)
            report.update(
                dates=len(rows),
                invalidated_caches=[str(p) for p in caches],
                outputs=[str(p) for p in outputs],
            )
            if not validate_only:
                install(replacements, baseline, folder)
            report["status"] = "validated" if validate_only else "complete"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (reports / "summary.json").write_text(json.dumps(report, indent=2))
        register(
            ArtifactSpec(
                reports,
                category="data_repair",
                experiment="microcap_400",
                run_id=run_id,
            ),
            "microcap_repair",
            run_id,
            report["status"],
        )
        Logger.stdout(f"[microcap repair] {report['status']}: {folder}", vb_level=0)
    return report
