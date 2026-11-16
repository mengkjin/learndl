"""Manual, checkpointed factor repairs; never launch audits or model training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
from typing import Any
from uuid import uuid4
import numpy as np
import pandas as pd

from .schema import FACTORS
from .bundle import file_digest, write_json
from .rebuild_data import REPAIR_VERSION, build_repaired_data

__all__ = ["Checkpoint", "dependency_order", "run_repair", "arguments"]


def dependency_order(registry, changed):
    """Transitive declared dependencies plus known event-weight consumers."""
    selected = set(changed)
    pools = {}
    while True:
        added = False
        for name, cls in registry.items():
            if cls.meta_type != "pooling" or name in selected:
                continue
            deps = {str(f).lstrip("+-") for f in getattr(cls, "sub_factors", [])}
            if deps & selected or name in (
                "event_factor_momentum",
                "event_factor_momentum_test",
            ):
                pools[name] = deps
                selected.add(name)
                added = True
        if not added:
            break
    ordered = []
    while pools:
        ready = sorted(n for n, deps in pools.items() if not deps & pools.keys())
        if not ready:
            raise ValueError("Cyclic pooling dependencies")
        for name in ready:
            ordered.append(name)
            del pools[name]
    return ordered


class Checkpoint:
    """SQLite commits complete steps only; backups precede each mutation."""

    def __init__(self, folder, identity, reuse=None):
        self.reuse = sqlite3.connect(reuse) if reuse is not None else None
        self.reuse_identity = hashlib.sha256(
            json.dumps(
                {k: identity.get(k) for k in ("version", "code", "raw_sources")},
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if self.reuse is not None:
            self.reuse.execute(
                "create table if not exists reusable (key text primary key, outputs text)"
            )
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.folder / "checkpoint.sqlite")
        self.db.execute("create table if not exists identity (value text)")
        self.db.execute(
            "create table if not exists steps (key text primary key, status text, error text)"
        )
        self.db.execute(
            "create table if not exists outputs (path text primary key, digest text)"
        )
        self.db.execute("create table if not exists pending (path text primary key)")
        encoded = json.dumps(identity, sort_keys=True)
        row = self.db.execute("select value from identity").fetchone()
        if row and row[0] != encoded:
            raise ValueError(
                "Resume rejected: code/config/source inventory changed; use a new run ID"
            )
        if not row:
            self.db.execute("insert into identity values (?)", (encoded,))
        self.db.commit()
        pending = {r[0] for r in self.db.execute("select path from pending")}
        for path, digest in self.db.execute("select path,digest from outputs"):
            if path not in pending and (
                not Path(path).exists() or file_digest(path) != digest
            ):
                raise ValueError(f"Resume rejected: completed output revised: {path}")

    def watch(self, path):
        path = Path(path)
        if (
            self.db.execute(
                "select 1 from outputs where path=?", (str(path),)
            ).fetchone()
            is None
        ):
            self.db.execute(
                "insert into outputs values (?,?)", (str(path), file_digest(path))
            )
            self.db.commit()

    def step(self, key, paths, action):
        row = self.db.execute("select status from steps where key=?", (key,)).fetchone()
        if row and row[0] == "passed":
            return False
        paths = [Path(p) for p in paths]
        reusable = key.startswith(("factor:dtop:", "factor:ff_", "event:"))
        cache_key = self.reuse_identity + ":" + key
        if reusable and self.reuse is not None:
            cached = self.reuse.execute(
                "select outputs from reusable where key=?", (cache_key,)
            ).fetchone()
            outputs = json.loads(cached[0]) if cached else {}
            if (
                outputs
                and set(outputs) == {str(p) for p in paths}
                and all(p.exists() and file_digest(p) == outputs[str(p)] for p in paths)
            ):
                self.db.execute(
                    "insert or replace into steps values (?, ?, ?)",
                    (key, "passed", "reused verified output"),
                )
                for path, digest in outputs.items():
                    self.db.execute(
                        "insert or replace into outputs values (?,?)", (path, digest)
                    )
                self.db.commit()
                return False
        private = (
            self.folder / "backups" / hashlib.sha256(key.encode()).hexdigest()[:24]
        )
        private.mkdir(parents=True, exist_ok=True)
        backup_records = []
        for path in paths:
            backup = private / (
                hashlib.sha256(str(path).encode()).hexdigest()[:16] + path.suffix
            )
            if path.exists() and not backup.exists():
                before = file_digest(path)
                shutil.copy2(path, backup)
                if file_digest(backup) != before or file_digest(path) != before:
                    raise ValueError(f"Output changed during backup: {path}")
            backup_records.append(
                dict(
                    source=str(path),
                    backup=str(backup),
                    old_sha256=file_digest(backup) if backup.exists() else None,
                )
            )
            self.db.execute("insert or ignore into pending values (?)", (str(path),))
        write_json(
            private / "backup_manifest.json", dict(step=key, files=backup_records)
        )
        self.db.execute(
            "insert or replace into steps values (?, ?, ?)", (key, "running", "")
        )
        self.db.commit()
        try:
            action()
            for path in paths:
                if not path.exists():
                    raise ValueError(f"Step produced no output: {path}")
                self.db.execute(
                    "insert or replace into outputs values (?,?)",
                    (str(path), file_digest(path)),
                )
                self.db.execute("delete from pending where path=?", (str(path),))
            self.db.execute("update steps set status=? where key=?", ("passed", key))
            self.db.commit()
            if reusable and self.reuse is not None:
                self.reuse.execute(
                    "insert or replace into reusable values (?,?)",
                    (cache_key, json.dumps({str(p): file_digest(p) for p in paths})),
                )
                self.reuse.commit()
            return True
        except BaseException as exc:
            self.db.execute(
                "update steps set status=?,error=? where key=?",
                ("failed", str(exc), key),
            )
            self.db.commit()
            raise


def _inventory(root, end):
    """Metadata fingerprints detect revision; output files additionally use SHA256."""
    digest = hashlib.sha256()
    for folder in sorted(Path(root).glob("DB_*")):
        for path in sorted(folder.rglob("*.feather")):
            if any(
                x in path.parts for x in ("1min", "5min", "15min", "30min", "60min")
            ):
                continue  # risk uses the already computed market_daily source
            match = re.search(r"\.(\d{8})\.", path.name)
            if match and int(match[1]) > end:
                continue
            stat = path.stat()
            digest.update(f"{path}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _factor_inventory(registry, names, managed, end, min_date=0):
    """Fingerprint unchanged factor dependencies as well as underlying raw data."""
    from src.proj.db.interface.db_path import DBPath

    digest = hashlib.sha256()
    for name in sorted(set(names) - set(managed)):
        cls = registry[name]
        if hasattr(cls, "full_dbs"):
            continue  # affiliate stores are included in the raw DB inventory
        parent = DBPath.Parent(cls.db_src, cls.db_key)
        for path in sorted(parent.rglob("*.feather")):
            match = re.search(r"\.(\d{8})\.", path.name)
            if match and not min_date <= int(match[1]) <= end:
                continue
            stat = path.stat()
            digest.update(f"{path}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _valid_file(cls, date, db):
    try:
        frame = db.load(cls.db_src, cls.db_key, int(date), key_column="date")
        if frame.empty or not {"secid", "date", cls.factor_name} <= set(frame):
            return False
        if (
            frame.secid.isna().any()
            or frame.secid.duplicated().any()
            or not frame.date.eq(date).all()
        ):
            return False
        if np.isinf(pd.to_numeric(frame[cls.factor_name], errors="raise")).any():
            return False
        if cls.factor_name == "dtop" and not {
            "dtop__source_observed",
            "dtop__source_filled",
        } <= set(frame):
            return False
        return True
    except Exception:
        return False


def run_repair(
    mode, start=20110101, end=None, run_id=None, dry_run=False, batch_size=32
):
    from src.proj import PATH, CALENDAR, DB
    from src.data import DATAVENDOR
    from src.res.factor.calculator import StockFactorHierarchy
    from src.res.factor.calculator.factor_calc import AffiliateFactorCalculator
    from src.res.factor.defs.stock.level0.behavior.behavior_ff3 import FamaFrench3
    from src.automation.artifact_exchange.core import ArtifactSpec
    from src.automation.artifact_exchange.research import register

    if mode not in ("rebuild", "backfill") or batch_size < 1:
        raise ValueError("Invalid mode/batch size")
    latest = int(CALENDAR.updated())
    end = int(end or (20161230 if mode == "backfill" else latest))
    if start > end or end > latest or (mode == "backfill" and end >= 20170101):
        raise ValueError("Invalid date range")
    run_id = (
        run_id
        or f"{mode}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    )
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Unsafe run ID")
    folder = PATH.result / "factor_sequence_fixes" / run_id
    reports = folder / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = dict(
        run_id=run_id,
        status="running",
        mode=mode,
        start=start,
        end=end,
        dry_run=dry_run,
        attempt_id=uuid4().hex[:12],
    )
    checkpoint = None
    try:
        registry = StockFactorHierarchy().pool
        base = [f.lstrip("+-") for f in FACTORS]
        ff = sorted(
            n for n, c in registry.items() if c.__module__.endswith("behavior_ff3")
        )
        changed = sorted(set(base + ff))
        pools = dependency_order(registry, changed)
        targets = base if mode == "backfill" else ["dtop", *ff]
        affected_end = (
            latest  # revisions to event weights and rolling state propagate forward
        )
        old_summary = reports / "summary.json"
        if old_summary.exists() and run_id:
            prior = json.loads(old_summary.read_text())
            affected_end = int(prior.get("affected_end", affected_end))
        status["affected_end"] = affected_end
        code = hashlib.sha256()
        for directory in [
            PATH.srcs / "func",
            PATH.srcs / "res" / "factor",
            Path(__file__).parent,
        ]:
            for p in sorted(directory.rglob("*.py")):
                code.update(str(p).encode())
                code.update(file_digest(p).encode())
        identity = dict(
            version=REPAIR_VERSION,
            mode=mode,
            start=start,
            end=end,
            affected_end=affected_end,
            code=code.hexdigest(),
            raw_sources=_inventory(PATH.db, affected_end),
            targets=targets,
            pools=pools,
            semantic_rebuild=["dtop", *ff],
            later_factors=_factor_inventory(
                registry, base, ["dtop", *ff], affected_end, end + 1
            )
            if mode == "backfill"
            else None,
            input_factors=_factor_inventory(
                registry,
                base,
                base if mode == "backfill" else ["dtop", *ff],
                affected_end,
            ),
        )
        write_json(reports / "plan.json", identity)
        if dry_run:
            status["status"] = "dry_run"
            return folder
        checkpoint = Checkpoint(folder, identity, folder.parent / "reuse.sqlite")
        dates = [int(d) for d in CALENDAR.range(start, affected_end, "td")]

        def clear(force=True):
            FamaFrench3._cache_models.clear()
            if force:
                for name in (
                    "TRADE",
                    "RISK",
                    "INDI",
                    "BS",
                    "IS",
                    "CF",
                    "FINA",
                    "ANALYST",
                ):
                    loader = getattr(DATAVENDOR, name)
                    if hasattr(loader, "clear_all"):
                        loader.clear_all()

        # Process a date across all factors before moving on; retain no growing FF3 cache.
        for i, date in enumerate(dates):
            day_targets = sorted(
                set(
                    ["dtop", *ff] + (base if mode == "backfill" and date <= end else [])
                )
            )
            for name in day_targets:
                cls = registry[name]
                if date < cls.init_date:
                    continue
                if issubclass(cls, AffiliateFactorCalculator):
                    if not len(cls.Loads([date], normalize=False)):
                        raise ValueError(
                            f"Affiliate source unavailable: {name}@{date}; upstream rebuild required"
                        )
                    continue
                path = DB.path(cls.db_src, cls.db_key, date)
                if (
                    mode == "backfill"
                    and name not in ["dtop", *ff]
                    and _valid_file(cls, date, DB)
                ):
                    checkpoint.watch(path)
                    continue

                def calculate(cls=cls, date=date):
                    before = DB.load(cls.db_src, cls.db_key, date)
                    calc = cls()
                    if not calc.calc_and_deploy(date, overwrite=True):
                        raise ValueError(
                            f"Factor save failed: {cls.factor_name}@{date}"
                        )
                    if not _valid_file(cls, date, DB):
                        raise ValueError(f"Invalid output {cls.factor_name}@{date}")
                    after = DB.load(cls.db_src, cls.db_key, date)
                    if cls.factor_name == "ff_r2_1m" and date in {
                        20110104,
                        20250826,
                        20250905,
                        20251009,
                        affected_end,
                    }:
                        fit = FamaFrench3.get_ff3_model(1, date)
                        x = fit.ff3_x.to_numpy(dtype=float)
                        details = []
                        for column, stock in enumerate(fit.stk.columns):
                            y = fit.stk.iloc[:, column].to_numpy(dtype=float)
                            valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
                            error = None
                            if fit.fit_reason.iloc[column] == "valid":
                                reference = np.linalg.pinv(x[valid]) @ y[valid]
                                error = float(
                                    np.max(np.abs(reference - fit.coef[:, column]))
                                )
                                if not np.allclose(
                                    reference, fit.coef[:, column], rtol=1e-7, atol=1e-9
                                ):
                                    raise ValueError(
                                        f"FF3 independent regression mismatch {date}/{stock}"
                                    )
                            details.append(
                                dict(
                                    secid=int(str(stock)),
                                    observations=int(valid.sum()),
                                    reason=fit.fit_reason.iloc[column],
                                    coefficient_error=error,
                                )
                            )
                        pd.DataFrame(details).to_csv(
                            reports / f"ff3_{date}.csv", index=False
                        )
                    a = (
                        before.set_index("secid")[cls.factor_name]
                        if not before.empty and "secid" in before
                        else pd.Series(dtype=float)
                    )
                    b = after.set_index("secid")[cls.factor_name]
                    frame = pd.concat([a.rename("old"), b.rename("new")], axis=1)
                    row = dict(
                        date=date,
                        factor=cls.factor_name,
                        old_valid=int(a.notna().sum()),
                        new_valid=int(b.notna().sum()),
                        changed=int(
                            (~np.isclose(frame.old, frame.new, equal_nan=True)).sum()
                        ),
                    )
                    with (reports / "differences.jsonl").open("a") as stream:
                        stream.write(json.dumps(row) + "\n")

                checkpoint.step(f"factor:{name}:{date}", [path], calculate)
            clear(force=(i % batch_size == batch_size - 1))
            if i % 25 == 0:
                print(
                    f"[{mode}] factors {i + 1}/{len(dates)} through {date}", flush=True
                )
        # Event calculation includes prehistory; explicitly replace all computed rows.
        event_cls = registry["high_level_switch"]
        event_path = DB.path(event_cls.db_src, event_cls.db_key)

        def event():
            old = DB.load(event_cls.db_src, event_cls.db_key)
            cls = event_cls()
            new = cls.validate_value(
                cls.calc_history(affected_end), affected_end, strict=True
            )
            if not old.empty:
                new = pd.concat(
                    [new, old.loc[old.date > affected_end]], ignore_index=True
                ).sort_values("date")
            DB.save(new, event_cls.db_src, event_cls.db_key)
            new[
                [
                    "date",
                    "source_valid",
                    "source_reason",
                    "industry_required",
                    "industry_available",
                ]
            ].to_csv(reports / "event_sources.csv", index=False)

        checkpoint.step(f"event:high_level_switch:{affected_end}", [event_path], event)

        def stats(name):
            cls = registry[name]
            calc = cls()
            first = max(start, cls.init_date)
            for period in ("daily", "weekly"):
                # Existing project period eligibility and return horizons are retained.
                eligible = np.asarray(
                    calc.stats_target_dates(first, affected_end, overwrite=True)[
                        period
                    ],
                    dtype=int,
                )
                for year in np.unique(eligible // 10000):
                    subset = eligible[eligible // 10000 == year]
                    path = DB.path(f"factor_stats_{period}", cls.db_key)

                    def update(subset=subset, period=period, path=path):
                        # Avoid parallel calculator loads and swallowed errors.
                        factor = calc.stock_factor(
                            subset,
                            normalize=True,
                            multi_thread=False,
                            ignore_error=False,
                        )
                        new = getattr(factor, f"{period}_stats")()
                        if new.empty:
                            raise ValueError(f"Empty stats: {name}/{period}/{year}")
                        from src.func.factor_groups import GROUPING_VERSION

                        new["grouping_version"] = GROUPING_VERSION
                        old = DB.load(f"factor_stats_{period}", cls.db_key)
                        result = (
                            pd.concat([old, new])
                            .drop_duplicates("date", keep="last")
                            .sort_values("date")
                        )
                        DB.save(result, f"factor_stats_{period}", cls.db_key)

                    checkpoint.step(f"stats:{name}:{period}:{year}", [path], update)
                    clear()

        for name in changed:
            stats(name)
        for name in pools:
            cls = registry[name]
            calc: Any = cls()
            pool_dates = [
                int(d)
                for d in CALENDAR.range(max(start, cls.init_date), affected_end, "td")
            ]
            if not pool_dates:
                continue
            if hasattr(calc, "calc_pooling_weight"):
                path = DB.path("pooling_weight", cls.db_key)

                def weights():
                    new = calc.calc_pooling_weight(dates=pool_dates, overwrite=True)
                    if "date" not in new:
                        new = new.reset_index()
                    if new.empty or not set(pool_dates) <= set(new.date):
                        raise ValueError(f"Incomplete pooling weights: {name}")
                    old = DB.load("pooling_weight", cls.db_key)
                    result = (
                        pd.concat([old, new])
                        .drop_duplicates("date", keep="last")
                        .sort_values("date")
                    )
                    DB.save(result, "pooling_weight", cls.db_key)
                    calc.__dict__.pop("loaded_weight", None)

                checkpoint.step("weights:" + name, [path], weights)
            for pool_index, date in enumerate(pool_dates):
                path = DB.path(cls.db_src, cls.db_key, date)

                def pooled(date=date):
                    if not calc.calc_and_deploy(date, overwrite=True):
                        raise ValueError(f"Failed pooling save {name}@{date}")

                checkpoint.step(f"pool:{name}:{date}", [path], pooled)
                if pool_index % batch_size == batch_size - 1:
                    clear()
            stats(name)
            clear()
        marker = reports / "dataset.json"

        def data():
            if _inventory(PATH.db, affected_end) != identity["raw_sources"]:
                raise ValueError(
                    "Raw sources changed during repair; use a new version/run before publishing data"
                )
            # Full prefix required for persistent observation/event state.
            target = build_repaired_data(
                PATH.interim / "factor_sequence_repaired",
                20110101,
                affected_end,
                batch_size,
            )
            write_json(marker, dict(dataset=str(target), version=REPAIR_VERSION))

        checkpoint.step("data", [marker], data)
        from .rebuild_data import load_repaired_manifest

        load_repaired_manifest(json.loads(marker.read_text())["dataset"])
        status["status"] = "completed"
        return folder
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if checkpoint is not None:
            status["steps"] = dict(
                checkpoint.db.execute(
                    "select status,count(*) from steps group by status"
                )
            )
            checkpoint.db.close()
            if checkpoint.reuse is not None:
                checkpoint.reuse.close()
        write_json(reports / "summary.json", status)
        register(
            ArtifactSpec(
                reports,
                category="factor_sequence_fixes",
                experiment=mode,
                run_id=run_id + "_" + status["attempt_id"],
            ),
            mode,
            run_id + "_" + status["attempt_id"],
            status["status"],
        )
        print(f"[{mode}] {status['status']}: {reports}", flush=True)


def arguments():
    def boolean(value):
        if str(value).lower() not in ("true", "false"):
            raise argparse.ArgumentTypeError("Expected true or false")
        return str(value).lower() == "true"

    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=20110101)
    p.add_argument("--end", type=int)
    p.add_argument("--run-id", "--run_id")
    p.add_argument(
        "--dry-run", "--dry_run", type=boolean, nargs="?", const=True, default=False
    )
    p.add_argument("--batch-size", "--batch_size", type=int, default=32)
    p.add_argument("--task_id", help=argparse.SUPPRESS)
    p.add_argument("--source", help=argparse.SUPPRESS)
    p.add_argument("--email", type=boolean, help=argparse.SUPPRESS)
    return {k: v for k, v in vars(p.parse_args()).items() if v is not None}
