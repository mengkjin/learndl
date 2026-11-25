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
import fcntl
import hashlib
import json
import os
import shutil
import uuid
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import NotRequired, TypedDict
from time import perf_counter

import numpy as np
import pandas as pd

from src.data import DATAVENDOR
from src.proj import PATH, DB, CALENDAR, Const, Load, Logger
from src.res.factor.fmp.prepared_market import default_contract, fingerprints
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

SCHEMA = 4


class SnapshotBuildReport(TypedDict):
    compute_seconds: float
    save_seconds: float
    total_seconds: float
    rewritten_years: list[int]
    reused_years: list[int]
    rebuild_reasons: dict[int, list[str]]
    generation: str
    bytes: NotRequired[int]
    removed_generations: NotRequired[list[str]]


def root_path():
    return PATH.interim / "training_analysis_snapshot"


def source_state(dates, *, universe_files=None):
    dates = sorted(set(map(int, dates)))
    paths: dict[str, object] = dict(fingerprints(dates))
    calendar = DB.path("information_ts", "calendar")
    paths[str(calendar)] = PreparedUniverse.fingerprint(calendar)
    paths["next_dates"] = [int(CALENDAR.td(d, 1)) for d in dates]
    paths["label_ends"] = [int(CALENDAR.td(d, 11)) for d in dates]
    extended = DATAVENDOR.td_within(
        int(CALENDAR.td(dates[0], -30)), int(CALENDAR.td(dates[-1], 30))
    )
    for d in extended:
        for src, key in [("trade_ts", "day"), ("models", "tushare_cne5_exp")]:
            p = DB.path(src, key, int(d))
            paths[str(p)] = PreparedUniverse.fingerprint(p)
    universe = PreparedUniverse(root_path() / "unused", "top-1000")
    paths["universe_dependencies"] = universe.dependencies(dates, file_fingerprints=universe_files)
    paths["universe_sources"] = universe.sources(dates)
    return paths


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
    root = Path(root) if root is not None else root_path()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        token = CURRENT.set(None)
        try:
            report = _build(dates, root, force=force or not resume, resume=resume)
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


def _source_changes(prior, sources, dates, universe_files):
    old = prior["sources"]
    reasons = []
    # Absolute source keys store [path, size, mtime, ctime] or [path, None].
    def files(state):
        return {key: value[1:] for key, value in state.items() if Path(key).is_absolute()}

    reasons.extend(_file_changes(files(old), files(sources)))
    for key, label in (("next_dates", "下一交易日"), ("label_ends", "标签结束日"),
                       ("universe_sources", "股票池成分文件"), ("universe_dependencies", "股票池依赖指纹")):
        if old.get(key) == sources.get(key):
            continue
        before, after = old.get(key, {}), sources.get(key, {})
        if key in ("next_dates", "label_ends"):
            before = dict(zip(prior["dates"], before))
            after = dict(zip(dates, after))
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


def _year_reasons(old, manifest, year, dates, sources, universe_files, old_folder, global_reasons):
    reasons = global_reasons.copy()
    prior = old["years"].get(str(year))
    if not prior:
        return reasons or ["此前没有该年度分区"]
    for key, label in (("schema", "快照格式版本"), ("label", "收益标签参数")):
        if old.get(key) != manifest[key]:
            reasons.append(f"{label} {old.get(key)} → {manifest[key]}")
    reasons.extend(_contract_changes(old.get("contract") or {}, manifest["contract"]))
    if prior["dates"] != dates:
        added = set(dates).difference(prior["dates"])
        reasons.append(f"新增覆盖日期（{_date_summary(added)}）" if added else "交易日期列表变化")
    if prior.get("sources") != sources:
        changes = _source_changes(prior, sources, dates, universe_files) if prior.get("sources") else []
        reasons.extend(changes or ["数据源指纹变化"])
    cutoff = min(int(CALENDAR.updated()), int(CALENDAR.td(dates[-1], 11)))
    if "maturity_cutoff" not in prior:
        reasons.append("缺少旧版标签成熟截止日")
    elif prior["maturity_cutoff"] < cutoff:
        reasons.append(f"标签成熟截止日从 {prior['maturity_cutoff']} 推进到 {cutoff}")
    if not reasons:
        for kind, checksum in prior["hashes"].items():
            path = old_folder / str(year) / f"{kind}.feather"
            if not path.is_file():
                reasons.append(f"年度文件缺失 {path.name}")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                reasons.append(f"年度文件校验失败 {path.name}")
    return reasons


def _build(dates, root, force=False, resume=True) -> SnapshotBuildReport:
    start = perf_counter()
    compute = save = 0.0
    pointer = root / "current.json"
    previous = None
    old = {"years": {}}
    global_reasons = []
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
    except (OSError, ValueError, KeyError) as error:
        Logger.warning(
            f"Previous snapshot unavailable; rebuilding requested dates: {error}"
        )
        previous = None
        old = {"years": {}}
        global_reasons.append(f"上一版快照无法读取（{error}）")
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
    try:
        Logger.info(f"检查快照覆盖的 {len(annual_dates)} 个年度（{dates[0]} 至 {dates[-1]}）")
        for year, ds in annual_dates.items():
            universe_files = {}
            before = source_state(ds, universe_files=universe_files)
            reasons = _year_reasons(old, manifest, year, ds, before, universe_files, old_folder, global_reasons)
            if reasons:
                rebuild_reasons[year] = reasons
                rebuild_inputs[year] = (before, universe_files)
            else:
                reused.append(year)
        Logger.info(f"计划重算年度 {list(rebuild_reasons)}，复用年度 {reused}")
        for year, reasons in rebuild_reasons.items():
            for reason in reasons:
                Logger.info(f"{year} 年重算 · {reason}")
        for year, ds in annual_dates.items():
            target = folder / str(year)
            target.mkdir()
            if year not in rebuild_inputs:
                assert old_folder is not None
                for p in (old_folder / str(year)).glob("*.feather"):
                    os.link(p, target / p.name)
                manifest["years"][str(year)] = old["years"][str(year)]
                continue
            before, universe_files = rebuild_inputs.pop(year)
            Logger.info(f"开始重算 {year} 年，共 {len(ds)} 个交易日")
            at = perf_counter()
            DATAVENDOR.clear_all()
            market = market_frame(ds)
            evaluation = evaluation_frame(ds)
            membership = membership_frame(ds)
            compute += perf_counter() - at
            at = perf_counter()
            checksums = {}
            for name, frame in [
                ("market", market),
                ("evaluation", evaluation),
                ("membership", membership),
            ]:
                path = target / f"{name}.feather"
                frame.to_feather(path, compression="lz4")
                checksums[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            save += perf_counter() - at
            if before != source_state(ds):
                raise RuntimeError(f"{year}: sources changed during snapshot build")
            manifest["years"][str(year)] = {
                "dates": ds,
                "sources": before,
                "universe_files": universe_files,
                "hashes": checksums,
                "maturity_cutoff": int(CALENDAR.updated()),
                "columns": {
                    k: list(v.columns)
                    for k, v in [
                        ("market", market),
                        ("evaluation", evaluation),
                        ("membership", membership),
                    ]
                },
            }
        if not rebuild_reasons and previous:
            shutil.rmtree(folder)
            return {
                "compute_seconds": compute,
                "save_seconds": save,
                "total_seconds": perf_counter() - start,
                "rewritten_years": [],
                "reused_years": reused,
                "rebuild_reasons": {},
                "generation": previous["generation"],
            }
        at = perf_counter()
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
        os.replace(tmp, pointer)
        save += perf_counter() - at
    except BaseException:
        shutil.rmtree(folder)
        (root / f".{generation}.json").unlink(missing_ok=True)
        raise
    return {
        "compute_seconds": compute,
        "save_seconds": save,
        "total_seconds": perf_counter() - start,
        "rewritten_years": list(rebuild_reasons),
        "reused_years": reused,
        "rebuild_reasons": rebuild_reasons,
        "generation": generation,
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
    return build_snapshot(dates)


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
