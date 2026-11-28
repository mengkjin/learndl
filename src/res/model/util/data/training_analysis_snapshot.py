"""Annual, versioned market/evaluation inputs for model-training analysis."""

from __future__ import annotations

__all__ = [
    'SCHEMA',
    'SnapshotBuildReport',
    'root_path',
    'source_state',
    'market_frame',
    'evaluation_frame',
    'membership_frame',
    'build_snapshot',
    'TrainingAnalysisSnapshot',
    'weekly_update',
    'prune_generations',
    'main',
]
import contextlib
import csv
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import NotRequired, TypedDict
from time import perf_counter

import numpy as np
import pandas as pd

from src.data import DATAVENDOR
from src.proj import PATH, DB, CALENDAR, Const, Load, Logger
from src.proj.env.machine import DFDATA_PATH
from src.res.factor.fmp.prepared_market import default_contract
from src.res.factor.util.classes.prepared_universe import PreparedUniverse

from src.res.factor.util.classes.analysis_inputs import (
    CURRENT,
    LABEL,
    Axis,
    EvaluationKind,
    LabelParams,
    Quote,
    return_label_scope,
)
from . import snapshot_audit

SCHEMA = 4
DEPENDENCY_VERSION = 1
ROUTINE_INFO = ("calendar", "industry", "description", "change_name")
TABLES = ("market", "evaluation", "membership")
AUDIT_CATEGORY = "training_analysis_snapshot_audit"


class SnapshotBuildReport(TypedDict):
    compute_seconds: float
    save_seconds: float
    total_seconds: float
    rewritten_years: list[int]
    recomputed_years: list[int]
    reused_years: list[int]
    rebuild_reasons: dict[int, list[str]]
    generation: str
    bytes: NotRequired[int]
    removed_generations: NotRequired[list[str]]
    audit_path: NotRequired[str]


def root_path():
    return PATH.interim / "training_analysis_snapshot"


def source_state(dates, *, universe_files=None):
    dates = sorted(set(map(int, dates)))
    first, last = dates[0], dates[-1]
    # Quote security coverage uses ±60 natural days. Forward returns also load
    # prices from 20 natural days before their own lookback window.
    return_start = CALENDAR.cd(CALENDAR.td(first, -10), -20)
    return_end = int(CALENDAR.td(last, LABEL[0] + LABEL[1] + 10))
    quote_start = min(CALENDAR.cd(first, -60), return_start)
    quote_end = max(CALENDAR.cd(last, 60), return_end)
    # Fingerprint civil dates, including absent files, independently of routine
    # changes to the calendar or the current update horizon. The small tail
    # covers the next market endpoint across weekends and exchange holidays.
    price_end = max(CALENDAR.cd(last, 14), int(CALENDAR.td(last, 1)))

    def civil_days(start, end):
        return map(int, pd.date_range(str(start), str(end)).strftime("%Y%m%d"))

    industry_mapping = DFDATA_PATH / "mapping/industry/tushare.yaml"
    if not industry_mapping.exists():
        industry_mapping = industry_mapping.with_suffix(".json")
    paths_to_check = {
        DB.path("information_ts", "list_day_price"),
        industry_mapping,
    }
    paths_to_check.update(DB.path("trade_ts", "day", d) for d in civil_days(quote_start, quote_end))
    paths_to_check.update(DB.path("trade_ts", "adjprice", d) for d in civil_days(first, price_end))
    paths_to_check.update(DB.path("models", "tushare_cne5_exp", d) for d in dates)
    paths: dict[str, object] = {
        str(p): PreparedUniverse.fingerprint(p) for p in sorted(paths_to_check)
    }
    universe = PreparedUniverse(root_path() / "unused", "top-1000")
    paths["universe_dependencies"] = universe.dependencies(
        dates, file_fingerprints=universe_files, exclude_info=ROUTINE_INFO
    )
    paths["universe_sources"] = universe.sources(dates)
    return paths


def _routine_info_files():
    return {
        str(path): PreparedUniverse.fingerprint(path)[1:]
        for key in ROUTINE_INFO
        for path in [DB.path("information_ts", key)]
    }


def market_frame(dates):
    frames = []
    for d in dates:
        ind = DATAVENDOR.INFO.get_indus(d).rename(
            columns={"indus": "generator_industry"}
        )
        frame = ind.reset_index().rename(columns={"index": "secid"})
        for price in ("close", "open", "vwap"):
            q = DATAVENDOR.get_quote_ret(
                d, int(DATAVENDOR.td(d, 1)), "close", price, portfolio=True
            ).reset_index()
            name = f"close_to_next_{price}"
            q = q.rename(columns={"ret": name, "index": "secid"})
            q[name + "_present"] = True
            frame = frame.merge(
                q[["secid", name, name + "_present"]], on="secid", how="outer"
            )
            frame[name + "_present"] = (
                frame[name + "_present"].fillna(False).astype(bool)
            )
        for price in ("close", "open", "vwap"):
            column = f"close_to_next_{price}_present"
            frame[column] = frame[column].fillna(False).astype(bool)
        frames.append(frame.assign(date=d))
    result = pd.concat(frames, ignore_index=True).sort_values(["date", "secid"])
    result["generator_industry"] = result.generator_industry.fillna("unknown").astype(
        "category"
    )
    return result.reset_index(drop=True)


def evaluation_frame(dates):
    secids = DATAVENDOR.get_quotes_block(dates, extend=60).secid
    frame = (
        DATAVENDOR.nday_fut_ret(secids, np.asarray(dates), *LABEL)
        .to_dataframe()
        .rename(columns={"ret": "forward_return_10_2_close"})
    )
    risk = DATAVENDOR.risk_style_exp(secids, np.asarray(dates)).to_dataframe()
    industry = (
        DATAVENDOR.risk_industry_exp(secids, np.asarray(dates))
        .to_dataframe()
        .fillna("unknown")
    )
    if len(industry.columns):
        industry = industry.idxmax(axis=1).rename("risk_industry")
        frame = frame.join(industry)
    if len(risk.columns):
        frame = frame.join(risk.add_prefix("style_"))
    frame = frame.reset_index()
    frame["label_end"] = CALENDAR.offset(
        frame.date.to_numpy(), LABEL[0] + LABEL[1] - 1, "td"
    )
    frame["label_mature"] = frame.label_end <= int(CALENDAR.updated())
    if "risk_industry" in frame:
        frame["risk_industry"] = frame.risk_industry.astype("category")
    return frame.sort_values(["date", "secid"]).reset_index(drop=True)


def membership_frame(dates):
    frames = []
    for d in dates:
        path = DB.path("universe", "top-1000", d)
        if not path.exists():
            continue
        frame = Load.df(path).assign(date=d, universe_member=True)
        frames.append(frame)
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["date", "secid"])
        .reset_index(drop=True)
        if frames
        else pd.DataFrame(columns=["date", "secid", "universe_member"])
    )


def build_snapshot(
    dates, root=None, *, force=False, resume=True
) -> SnapshotBuildReport:
    audit_root = PATH.rslt_custom / AUDIT_CATEGORY if root is None else Path(root) / "audit"
    root = Path(root) if root is not None else root_path()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        token = CURRENT.set(None)
        try:
            report = _build(dates, root, audit_root, force=force or not resume, resume=resume)
            try:
                report["removed_generations"] = _prune_generations(root)
            except OSError as error:
                Logger.warning(f"Snapshot cleanup deferred: {error}")
            return report
        finally:
            CURRENT.reset(token)


def _date_summary(dates):
    ordered = sorted(dates)
    if len(ordered) <= 3:
        return ", ".join(map(str, ordered))
    return f"{len(ordered)} 个交易日，首日 {ordered[0]}，末日 {ordered[-1]}"


def _file_changes(old, new):
    groups = defaultdict(list)
    for key in sorted(old.keys() | new.keys()):
        if old.get(key) == new.get(key):
            continue
        path = Path(key)
        before, after = old.get(key), new.get(key)
        if before is None or after is None:
            reason = "新增依赖" if before is None else "移出依赖"
        elif before == [None] or after == [None]:
            reason = "文件出现" if before == [None] else "文件缺失"
        else:
            fields = [name for name, a, b in zip(("大小", "mtime", "ctime"), before, after) if a != b]
            reason = "/".join(fields) + " 变化"
        groups[path.name.split(".")[0], reason].append(key)
    return [f"{name} {reason}（{len(files)} 个，例如 {files[0]}）"
            for (name, reason), files in groups.items()]


def _source_changes(prior, sources, universe_files):
    old = prior["sources"]
    reasons = []
    # Absolute source keys store [path, size, mtime, ctime] or [path, None].
    def files(state):
        return {key: value[1:] for key, value in state.items() if Path(key).is_absolute()}

    reasons.extend(_file_changes(files(old), files(sources)))
    for key, label in (("universe_sources", "股票池成分文件"),
                       ("universe_dependencies", "股票池依赖指纹")):
        if old.get(key) == sources.get(key):
            continue
        before, after = old.get(key, {}), sources.get(key, {})
        changed = [date for date in before.keys() & after.keys() if before[date] != after[date]]
        if changed:
            reasons.append(f"{label}变化（{_date_summary(changed)}）")
            if key == "universe_sources":
                reasons.extend(_file_changes({before[d][0]: before[d][1:] for d in changed},
                                             {after[d][0]: after[d][1:] for d in changed}))
            if key == "universe_dependencies":
                if "universe_files" in prior:
                    reasons.extend(_file_changes(prior["universe_files"], universe_files))
                else:
                    reasons.append("旧版股票池依赖仅记录每日摘要，可定位到上述日期")
    return reasons


def _contract_changes(before, after, prefix=""):
    for key in sorted(before.keys() | after.keys()):
        a, b = before.get(key), after.get(key)
        if a == b:
            continue
        field = f"{prefix}{key}"
        if isinstance(a, dict) and isinstance(b, dict):
            yield from _contract_changes(a, b, field + ".")
        else:
            yield f"组合计算设置 {field} 从 {a} 改为 {b}"


def _year_reasons(old, manifest, year, dates, sources, universe_files, global_reasons):
    reasons = global_reasons.copy()
    prior = old["years"].get(str(year))
    if not prior:
        return reasons or ["此前没有该年度分区"]
    if old.get("dependency_version") != DEPENDENCY_VERSION:
        reasons.append("首次建立审计基线")
    for key, label in (("schema", "快照格式版本"), ("label", "收益标签参数")):
        if old.get(key) != manifest[key]:
            reasons.append(f"{label} {old.get(key)} → {manifest[key]}")
    reasons.extend(_contract_changes(old.get("contract") or {}, manifest["contract"]))
    if prior["dates"] != dates:
        added = set(dates).difference(prior["dates"])
        reasons.append(f"新增覆盖日期（{_date_summary(added)}）" if added else "交易日期列表变化")
    if prior.get("sources") != sources:
        changes = _source_changes(prior, sources, universe_files) if prior.get("sources") else []
        reasons.extend(changes or ["数据源指纹变化"])
    last_label_end = prior.get("last_label_end")
    if last_label_end is None:
        last_label_end = int(CALENDAR.td(dates[-1], LABEL[0] + LABEL[1] - 1))
    cutoff = min(int(CALENDAR.updated()), last_label_end)
    if "maturity_cutoff" not in prior:
        reasons.append("缺少旧版标签成熟截止日")
    elif prior["maturity_cutoff"] < cutoff:
        reasons.append(f"标签成熟截止日从 {prior['maturity_cutoff']} 推进到 {cutoff}")
    return reasons


def _checksum(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _baseline_problems(folder, prior):
    problems = {}
    for kind in TABLES:
        path = folder / f"{kind}.feather"
        try:
            if _checksum(path) != prior.get("hashes", {}).get(kind):
                problems[kind] = f"年度文件校验失败 {path.name}"
        except FileNotFoundError:
            problems[kind] = f"年度文件缺失 {path.name}"
        except OSError as error:
            problems[kind] = f"年度文件读取失败 {path.name}: {error}"
    return problems


def _audit_files(sources, universe_files, info_files):
    files = {key: value[1:] for key, value in sources.items() if Path(key).is_absolute()}
    return files | universe_files | info_files


def _audit_dependencies(prior, sources, universe_files, info_files, reasons):
    previous_info = prior.get("routine_info_files", {})
    previous_sources = prior.get("sources", {})
    previous_universe = prior.get("universe_files", {})
    before = _audit_files(previous_sources, previous_universe, previous_info)
    after = _audit_files(sources, universe_files, info_files)
    universe_paths = set(universe_files) | set(previous_universe)
    for state in (sources, previous_sources):
        universe_paths.update(value[0] for value in state.get("universe_sources", {}).values())
    changes = []
    for path in sorted(before.keys() | after.keys()):
        if before.get(path) == after.get(path):
            continue
        name = Path(path).name.split(".")[0]
        if path in info_files:
            candidates = {
                "calendar": ["market.*", "evaluation.*"],
                "industry": ["market.generator_industry", "market.row"],
                "description": ["membership.*"],
                "change_name": ["membership.*"],
            }[name]
        elif path in universe_paths:
            candidates = ["membership.*"]
        else:
            candidates = {
                "day": ["evaluation.forward_return_10_2_close", "evaluation.row"],
                "adjprice": ["market.close_to_next_*", "market.row"],
                "list_day_price": ["evaluation.forward_return_10_2_close"],
                "tushare_cne5_exp": ["evaluation.risk_industry", "evaluation.style_*"],
                "tushare": ["market.generator_industry", "market.row"],
            }.get(name, [])
        changes.append(dict(path=path, old=before.get(path), new=after.get(path),
                            observed_only=path in info_files, candidate_fields=candidates))
    return dict(reasons=reasons, files=changes,
                legacy_universe_details_missing=bool(prior) and "universe_files" not in prior,
                attribution="candidate_inputs_only")


def _compare_and_save(audit, year, kind, path, frame, old_folder, prior, problem, dates, cutoff):
    summary = {"status": "new", "rows": len(frame)}
    if prior and not problem:
        old_path = old_folder / str(year) / path.name
        try:
            original = pd.read_feather(old_path)
        except (OSError, ValueError) as error:
            problem = f"年度文件无法解码 {path.name}: {error}"
        else:
            with snapshot_audit.open_details(audit, year, kind) as file:
                writer = csv.writer(file)
                writer.writerow(snapshot_audit.FIELDS)
                diff = snapshot_audit.TableDiff(writer, year, kind)
                snapshot_audit.compare_tables(
                    original, frame, diff, set(prior["dates"]), set(dates),
                    prior.get("maturity_cutoff", 0), cutoff,
                )
            if not diff.count:
                (audit / f"{year}-{kind}.csv.gz").unlink()
                os.link(old_path, path)
                Logger.info(f"{year}/{kind} 重算后一致，复用原表")
                return summary | {"status": "equal"}
            summary |= {"status": "changed", "differences": diff.count, "fields": diff.summary()}
            Logger.info(f"{year}/{kind} 发现 {diff.count} 项差异，将替换此表")
            for item in summary["fields"]:
                Logger.info(f"{year}/{kind} {json.dumps(item, ensure_ascii=False)}")
            for sample in diff.samples:
                Logger.info(f"{year}/{kind} 历史差异样例 {json.dumps(sample, ensure_ascii=False)}")
    if problem:
        summary |= {"status": "baseline_unavailable", "reason": problem}
        Logger.warning(f"{year}/{kind} 基准缺失或损坏，将重建 · {problem}")
    if summary["status"] != "changed":
        snapshot_audit.validate_keys(frame, year, kind)
    frame.to_feather(path, compression="lz4")
    return summary


def _build(dates, root, audit_root, force=False, resume=True) -> SnapshotBuildReport:
    start = perf_counter()
    compute = save = 0.0
    pointer = root / "current.json"
    previous = None
    old = {"years": {}}
    global_reasons = []
    unavailable_baseline = None
    try:
        if pointer.exists():
            previous = json.loads(pointer.read_text())
            if not isinstance(previous, dict) or not isinstance(
                previous.get("generation"), str
            ):
                raise ValueError("Invalid snapshot pointer")
            if resume:
                old_folder = root / "generations" / previous["generation"]
                old = json.loads((old_folder / "manifest.json").read_text())
                if not isinstance(old, dict) or not isinstance(old.get("years"), dict):
                    raise ValueError("Invalid previous snapshot manifest")
    except (OSError, ValueError, KeyError) as error:
        Logger.warning(
            f"Previous snapshot unavailable; rebuilding requested dates: {error}"
        )
        previous = None
        old = {"years": {}}
        unavailable_baseline = f"上一版快照无法读取（{error}）"
        global_reasons.append(unavailable_baseline)
    if force:
        global_reasons.append("请求强制重建" if resume else "本次关闭了增量复用")
    if not previous and not global_reasons:
        global_reasons.append("首次构建，尚无已发布快照")
    old_folder = root / "generations" / previous["generation"] if previous else None
    dates = sorted(
        set(map(int, dates))
        | {d for part in old["years"].values() for d in part["dates"]}
    )
    if not dates:
        raise ValueError("empty snapshot dates")
    generation = uuid.uuid4().hex
    folder = root / "generations" / generation
    folder.mkdir(parents=True)
    manifest = {
        "schema": SCHEMA,
        "dependency_version": DEPENDENCY_VERSION,
        "generation": generation,
        "label": list(LABEL),
        "contract": default_contract(),
        "years": {},
    }
    annual_dates = defaultdict(list)
    for date in dates:
        annual_dates[date // 10000].append(date)
    rebuild_reasons = {}
    rebuild_inputs = {}
    reused = []
    rewritten = []
    audit_target = audit_stage = None
    summaries = {}
    dependencies = {}
    try:
        info_files = _routine_info_files()
        Logger.info(f"检查快照覆盖的 {len(annual_dates)} 个年度（{dates[0]} 至 {dates[-1]}）")
        for year, ds in annual_dates.items():
            prior = old["years"].get(str(year))
            universe_files = {}
            before = source_state(ds, universe_files=universe_files)
            reasons = _year_reasons(old, manifest, year, ds, before, universe_files, global_reasons)
            problems = {}
            if prior:
                assert old_folder is not None
                problems = _baseline_problems(old_folder / str(year), prior)
            reasons.extend(problems.values())
            if reasons:
                rebuild_reasons[year] = reasons
                rebuild_inputs[year] = (before, universe_files, problems)
            else:
                reused.append(year)
        Logger.info(f"计划重算年度 {list(rebuild_reasons)}，复用年度 {reused}")
        for year, reasons in rebuild_reasons.items():
            for reason in reasons:
                Logger.info(f"{year} 年重算 · {reason}")
        if not rebuild_reasons and previous:
            shutil.rmtree(folder)
            return {
                "compute_seconds": compute,
                "save_seconds": save,
                "total_seconds": perf_counter() - start,
                "rewritten_years": [],
                "recomputed_years": [],
                "reused_years": reused,
                "rebuild_reasons": {},
                "generation": previous["generation"],
            }
        audit_root.mkdir(parents=True, exist_ok=True)
        audit_stage = Path(tempfile.mkdtemp(prefix=f".{generation}-", dir=audit_root))
        for year, ds in annual_dates.items():
            target = folder / str(year)
            target.mkdir()
            prior = old["years"].get(str(year))
            if year not in rebuild_inputs:
                assert old_folder is not None
                for kind in TABLES:
                    name = f"{kind}.feather"
                    os.link(old_folder / str(year) / name, target / name)
                manifest["years"][str(year)] = prior
                summaries[year] = {kind: {"status": "reused"} for kind in TABLES}
                continue
            before, universe_files, problems = rebuild_inputs[year]
            dependencies[year] = _audit_dependencies(
                prior or {}, before, universe_files, info_files, rebuild_reasons[year]
            )
            Logger.info(f"开始重算 {year} 年，共 {len(ds)} 个交易日")
            at = perf_counter()
            DATAVENDOR.clear_all()
            checksums = {}
            columns = {}
            summaries[year] = {}
            updated = int(CALENDAR.updated())
            last_label_end = int(CALENDAR.td(ds[-1], LABEL[0] + LABEL[1] - 1))
            for kind, make_frame in zip(TABLES, (market_frame, evaluation_frame, membership_frame)):
                frame = make_frame(ds)
                compute += perf_counter() - at
                at = perf_counter()
                path = target / f"{kind}.feather"
                summary = _compare_and_save(
                    audit_stage, year, kind, path, frame,
                    old_folder, prior, problems.get(kind, unavailable_baseline), ds, updated,
                )
                summaries[year][kind] = summary
                if summary["status"] == "equal":
                    assert prior is not None
                    checksums[kind] = prior["hashes"][kind]
                    columns[kind] = prior["columns"][kind]
                else:
                    checksums[kind] = _checksum(path)
                    columns[kind] = list(frame.columns)
                save += perf_counter() - at
                at = perf_counter()
            if any(summary["status"] != "equal" for summary in summaries[year].values()):
                rewritten.append(year)
            manifest["years"][str(year)] = {
                "dates": ds,
                "sources": before,
                "universe_files": universe_files,
                "routine_info_files": info_files,
                "hashes": checksums,
                "maturity_cutoff": updated,
                "last_label_end": last_label_end,
                "columns": columns,
            }
        at = perf_counter()
        snapshot_audit.write_summary(audit_stage, previous["generation"] if previous else None,
                                     generation, summaries, dependencies)
        (folder / "manifest.json").write_text(json.dumps(manifest))
        tmp = root / f".{generation}.json"
        tmp.write_text(
            json.dumps(
                {
                    "generation": generation,
                    "previous": previous["generation"] if previous else None,
                }
            )
        )
        if info_files != _routine_info_files():
            raise RuntimeError("Routine information files changed during snapshot build")
        for year, ds in annual_dates.items():
            if manifest["years"][str(year)]["sources"] != source_state(ds):
                raise RuntimeError(f"{year}: sources changed during snapshot build")
        audit_path = audit_root / generation
        os.replace(audit_stage, audit_path)
        audit_target = audit_path
        os.replace(tmp, pointer)
        save += perf_counter() - at
    except BaseException:
        shutil.rmtree(folder)
        if audit_stage is not None and audit_stage.exists():
            shutil.rmtree(audit_stage)
        if audit_target is not None:
            shutil.rmtree(audit_target)
        (root / f".{generation}.json").unlink(missing_ok=True)
        raise
    return {
        "compute_seconds": compute,
        "save_seconds": save,
        "total_seconds": perf_counter() - start,
        "rewritten_years": rewritten,
        "recomputed_years": list(rebuild_reasons),
        "reused_years": reused,
        "rebuild_reasons": rebuild_reasons,
        "generation": generation,
        "audit_path": str(audit_target),
        "bytes": sum(p.stat().st_size for p in folder.rglob("*.feather")),
    }


class TrainingAnalysisSnapshot:
    def __init__(self, root=None, dates=None, *, allow_partial=False):
        self.allow_partial = allow_partial
        self.unavailable_frames = set()
        self.root = Path(root) if root is not None else root_path()
        # Pin while holding the same lock as publication/GC: no pointer-to-lease race.
        with (self.root / "writer.lock").open("a") as writer:
            fcntl.flock(writer, fcntl.LOCK_SH | fcntl.LOCK_NB)
            pointer = json.loads((self.root / "current.json").read_text())
            if not isinstance(pointer, dict) or not isinstance(
                pointer.get("generation"), str
            ):
                raise ValueError("Invalid snapshot pointer")
            self.generation = pointer["generation"]
            self.folder = self.root / "generations" / self.generation
            self._lease = (self.folder / "reader.lock").open("a")
            fcntl.flock(self._lease, fcntl.LOCK_SH)
        try:
            self._validate(dates, allow_partial)
        except Exception:
            self.close()
            raise

    def close(self):
        lease = getattr(self, "_lease", None)
        if lease is not None:
            lease.close()

    def __del__(self):
        self.close()

    def _validate(self, dates, allow_partial):
        self.meta = json.loads((self.folder / "manifest.json").read_text())
        if not isinstance(self.meta, dict):
            raise ValueError("Invalid snapshot manifest")
        if (
            self.meta.get("label") != list(LABEL)
            or self.meta["schema"] != SCHEMA
            or self.meta.get("dependency_version") != DEPENDENCY_VERSION
            or self.meta["contract"] != json.loads(json.dumps(default_contract()))
        ):
            raise ValueError("snapshot schema/default configuration mismatch")
        self.cache = OrderedDict()
        years = (
            set(self.meta["years"])
            if dates is None
            else {str(int(d) // 10000) for d in dates}
        )
        valid_years = set()
        for year in years:
            part = self.meta["years"].get(year)
            if part is None or source_state(part["dates"]) != part["sources"]:
                if allow_partial:
                    continue
                raise ValueError(f"snapshot missing or revised year: {year}")
            valid_years.add(year)
        if years and not valid_years:
            raise ValueError("snapshot has no valid requested years")
        if dates is not None and not allow_partial:
            covered = {d for year in years for d in self.meta["years"][year]["dates"]}
            if not set(map(int, dates)).issubset(covered):
                raise ValueError("snapshot date coverage incomplete")
        self.validated_years = valid_years

    @contextlib.contextmanager
    def activate(self):
        token = CURRENT.set(self)
        try:
            with return_label_scope():
                yield self
        finally:
            CURRENT.reset(token)
            self.close()

    def _frame(self, kind, year):
        if year not in self.validated_years:
            return None
        key = (year, kind)
        if key in self.unavailable_frames:
            return None
        if key not in self.cache:
            path = self.folder / year / f"{kind}.feather"
            try:
                part = self.meta["years"][year]
                if (
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    != part["hashes"][kind]
                ):
                    raise ValueError("snapshot checksum mismatch")
                frame = pd.read_feather(path)
                if not {"date", "secid"}.issubset(frame.columns):
                    raise ValueError("snapshot missing date or secid")
            except (OSError, ValueError, KeyError) as error:
                if not self.allow_partial:
                    raise
                self.unavailable_frames.add(key)
                Logger.warning(
                    f"Training analysis snapshot {year}/{kind} unavailable: {error}"
                )
                return None
            self.cache[key] = frame
            while len(self.cache) > 3:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def _day(self, kind, date):
        year = str(int(date) // 10000)
        frame = self._frame(kind, year)
        if frame is None or int(date) not in self.meta["years"][year]["dates"]:
            return None
        axis = frame.date.to_numpy()
        return frame.iloc[
            np.searchsorted(axis, date, "left") : np.searchsorted(axis, date, "right")
        ]

    def table(self, kind, dates):
        frames = []
        for year in sorted({str(int(d) // 10000) for d in dates}):
            if year not in self.validated_years:
                return None
            selected = sorted({int(d) for d in dates if str(int(d) // 10000) == year})
            if not set(selected).issubset(set(self.meta["years"][year]["dates"])):
                return None
            frame = self._frame(kind, year)
            if frame is None:
                return None
            axis = frame.date.to_numpy()
            for date in selected:
                frames.append(
                    frame.iloc[
                        np.searchsorted(axis, date, "left") : np.searchsorted(
                            axis, date, "right"
                        )
                    ]
                )
        return pd.concat(frames, ignore_index=True) if frames else None

    def evaluation(
        self,
        kind: EvaluationKind,
        secids: Axis,
        dates: Axis,
        params: LabelParams = LABEL,
    ) -> pd.DataFrame | None:
        if kind == "ret" and tuple(params) != LABEL:
            return None
        frame = self.table("evaluation", dates)
        columns = {
            "ret": ["forward_return_10_2_close"],
            "industry": ["risk_industry"],
            "style": ["style_" + v for v in Const.Factor.RISK.style],
        }[kind]
        if frame is None or any(c not in frame for c in columns):
            return None
        requested = pd.MultiIndex.from_product([secids, dates], names=["secid", "date"])
        frame = frame.set_index(["secid", "date"])[columns]
        if not frame.index.is_unique:
            return None
        positions = frame.index.get_indexer(requested)
        if (positions < 0).any():
            return None
        frame = frame.take(positions).set_axis(requested)
        frame.columns = (
            ["ret"]
            if kind == "ret"
            else ["industry"]
            if kind == "industry"
            else Const.Factor.RISK.style
        )
        if kind == "industry":
            frame = frame.astype(object)
        return frame

    def quote(self, date: int, end: int, price: str) -> Quote | None:
        if end != int(DATAVENDOR.td(date, 1)):
            return None
        frame = self._day("market", date)
        key = f"close_to_next_{price}"
        if (
            frame is None
            or frame.empty
            or key not in frame
            or key + "_present" not in frame
        ):
            return None
        present = frame[key + "_present"].to_numpy(bool)
        return frame.secid.to_numpy(np.int64)[present], frame[key].to_numpy(float)[
            present
        ]

    def industry(self, date: int) -> pd.DataFrame | None:
        frame = self._day("market", date)
        return (
            None
            if frame is None or frame.empty or "generator_industry" not in frame
            else frame.set_index("secid")[["generator_industry"]]
            .rename(columns={"generator_industry": "indus"})
            .astype(object)
        )

    def universe(
        self, dates, exclusions="st_lowprice_bse_loser_warnst", *, name="top-1000"
    ):
        from src.res.factor.util import Universe, Portfolio

        if name != "top-1000":
            return None
        frame = self.table("membership", dates)
        required = exclusions.split("_")
        if frame is None or any(c not in frame for c in ["universe_member", *required]):
            return None
        frame = frame[frame.universe_member]
        if (
            not set(map(int, dates)).issubset(set(frame.date))
            or frame[required].isna().any().any()
        ):
            return None
        frame = Universe._combine_caches_with_exclusion([frame], exclusions)
        return Portfolio.from_dataframe(frame, name="univ")


def weekly_update():
    dates = [
        int(d)
        for d in DB.dates("trade_ts", "day")
        if int(d) >= 20170103 and int(d) <= int(CALENDAR.updated())
    ]
    report = build_snapshot(dates)
    if "audit_path" in report:
        from src.automation.artifact_exchange import ArtifactSpec
        from src.automation.artifact_exchange.research import CURRENT_TASK, publish_safe

        path = Path(report["audit_path"])
        Logger.info(f"快照一致性审计明细 · {path}")
        spec = ArtifactSpec(path, category=AUDIT_CATEGORY, experiment="weekly_update",
                            run_id=report["generation"])
        owner = CURRENT_TASK.get()
        if owner is None:
            publish_safe("weekly_update", report["generation"], [spec])
        else:
            owner.register_artifact(spec)
    return report


def prune_generations(root=None):
    root = Path(root) if root is not None else root_path()
    with (root / "writer.lock").open("a") as writer:
        fcntl.flock(writer, fcntl.LOCK_EX)
        return _prune_generations(root)


def _prune_generations(root):
    pointer = json.loads((root / "current.json").read_text())
    current = pointer["generation"]
    generations = sorted(
        (root / "generations").iterdir(),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    previous = pointer.get("previous")
    if "previous" not in pointer:
        previous = next(
            (
                path.name
                for path in generations
                if path.name != current and (path / "manifest.json").is_file()
            ),
            None,
        )
    removed = []
    for folder in generations:
        if not folder.is_dir() or folder.name in (current, previous):
            continue
        with (folder / "reader.lock").open("a") as reader:
            try:
                fcntl.flock(reader, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            shutil.rmtree(folder)
            removed.append(folder.name)
    for path in root.glob(".*.json"):
        if len(path.stem.lstrip(".")) == 32:
            path.unlink()
    return removed


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inspect", "validate", "rebuild"])
    parser.add_argument("--start", type=int, default=20170103)
    parser.add_argument("--end", type=int, default=99991231)
    args = parser.parse_args()
    dates = [
        int(d)
        for d in DB.dates("trade_ts", "day")
        if args.start <= int(d) <= min(args.end, int(CALENDAR.updated()))
    ]
    if args.action == "rebuild":
        Logger.stdout(json.dumps(build_snapshot(dates, force=True), indent=2), vb_level=0)
    else:
        root = root_path()
        pointer = json.loads((root / "current.json").read_text())
        folder = root / "generations" / pointer["generation"]
        meta = json.loads((folder / "manifest.json").read_text())
        if args.action == "validate":
            with TrainingAnalysisSnapshot(dates=dates).activate() as snapshot:
                for kind in ["market", "evaluation", "membership"]:
                    snapshot.table(kind, dates)
        Logger.stdout(
            json.dumps(
                {
                    "root": str(root),
                    "generation": pointer["generation"],
                    "years": {
                        year: {
                            "dates": len(part["dates"]),
                            "start": min(part["dates"]),
                            "end": max(part["dates"]),
                            "columns": part["columns"],
                        }
                        for year, part in meta["years"].items()
                    },
                },
                indent=2,
            ),
            vb_level=0,
        )


if __name__ == "__main__":
    main()
