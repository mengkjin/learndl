"""Compare annual snapshot tables and retain the evidence for publication."""

import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_numeric_dtype

RTOL = 1e-6
ATOL = 1e-8
KEYS = ["date", "secid"]
FIELDS = ("year", "table", "classification", "date", "secid", "field",
          "change", "old", "new", "abs_delta")
LABEL_FIELDS = {"label_mature", "forward_return_10_2_close"}


def _json_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if pd.isna(value):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return {"float": str(value)}
    return value


def validate_keys(frame, year, table):
    keys = frame[KEYS]
    duplicates = keys.duplicated(keep=False)
    if duplicates.any() or keys.isna().any().any():
        invalid = keys[duplicates | keys.isna().any(axis=1)].head(10)
        raise ValueError(f"{year}/{table}: invalid (date, secid) keys: {list(invalid.itertuples(index=False, name=None))}")


class TableDiff:
    """Write full details to CSV and keep a small summary in memory."""

    def __init__(self, writer, year, table):
        self.writer = writer
        self.year = year
        self.table = table
        self.groups = defaultdict(Counter)
        self.dates = defaultdict(set)
        self.secids = defaultdict(set)
        self.max_delta = {}
        self.samples = []
        self.count = 0

    def add(self, classification, date, secid, field, change, old, new):
        if date is not None:
            date, secid = int(date), int(secid)
        delta = None
        if change == "value" and type(old) in (int, float) and type(new) in (int, float):
            delta = abs(new - old)
        self.writer.writerow((self.year, self.table, classification, date, secid, field, change,
                              json.dumps(old, ensure_ascii=False), json.dumps(new, ensure_ascii=False), delta))
        self.count += 1
        key = classification, field
        self.groups[key][change] += 1
        if date is not None:
            self.dates[key].add(date)
            self.secids[key].add(secid)
        if delta is not None:
            self.max_delta[key] = max(self.max_delta.get(key, 0), delta)
        if classification == "historical" and len(self.samples) < 10:
            self.samples.append(dict(date=date, secid=secid, field=field, change=change, old=old, new=new))

    def summary(self):
        summaries = []
        for (classification, field), counts in self.groups.items():
            key = classification, field
            dates = self.dates[key]
            summaries.append(dict(
                classification=classification, field=field, counts=dict(counts),
                dates=len(dates), securities=len(self.secids[key]),
                first_date=min(dates, default=None), last_date=max(dates, default=None),
                max_abs_delta=self.max_delta.get(key),
            ))
        return summaries


def compare_tables(old, new, diff, old_dates, new_dates, old_cutoff, new_cutoff):
    """Compare logical rows, keeping the published values as the tolerance reference."""
    validate_keys(old, diff.year, diff.table)
    validate_keys(new, diff.year, diff.table)
    old = old.set_index(KEYS)
    new = new.set_index(KEYS)
    added_dates = new_dates - old_dates
    removed_dates = old_dates - new_dates
    for frame, other, change in ((old, new, "row_removed"), (new, old, "row_added")):
        rows = frame.loc[frame.index.difference(other.index, sort=False)]
        for (date, secid), values in zip(rows.index, rows.itertuples(name=None, index=False)):
            row = {field: _json_value(value) for field, value in zip(rows.columns, values)}
            classification = ("new_coverage" if date in added_dates else
                              "removed_coverage" if date in removed_dates else "historical")
            diff.add(classification, date, secid, "*", change,
                     row if change == "row_removed" else None, row if change == "row_added" else None)

    common = old.index.intersection(new.index, sort=False)
    old, new = old.loc[common], new.loc[common]
    maturing = np.zeros(len(common), dtype=bool)
    if diff.table == "evaluation" and {"label_end", "label_mature"} <= set(old) & set(new):
        ends = new["label_end"]
        maturing = (
            old["label_mature"].eq(False) & new["label_mature"].eq(True)
            & ends.eq(old["label_end"]) & ends.gt(old_cutoff) & ends.le(new_cutoff)
        ).fillna(False).to_numpy(dtype=bool)

    for field in sorted(set(old) | set(new)):
        if field not in old or field not in new:
            side, change = (new, "column_added") if field in new else (old, "column_removed")
            dtype = str(side[field].dtype)
            diff.add("historical", None, None, field, change,
                     dtype if field in old else None, dtype if field in new else None)
            for (date, secid), value in side[field].items():
                value = _json_value(value)
                diff.add("historical", date, secid, field, change,
                         value if field in old else None, value if field in new else None)
            continue
        before, after = old[field], new[field]
        if str(before.dtype) != str(after.dtype):
            diff.add("historical", None, None, field, "dtype", str(before.dtype), str(after.dtype))
        if is_numeric_dtype(before.dtype) and is_numeric_dtype(after.dtype) and (
            is_float_dtype(before.dtype) or is_float_dtype(after.dtype)
        ):
            equal = np.isclose(after.to_numpy(dtype=float, na_value=np.nan),
                               before.to_numpy(dtype=float, na_value=np.nan),
                               rtol=RTOL, atol=ATOL, equal_nan=True)
        else:
            both_present = before.notna() & after.notna()
            equal = (before.isna() & after.isna()).to_numpy(dtype=bool)
            equal[both_present.to_numpy()] = (
                before[both_present].astype(object).eq(after[both_present].astype(object)).to_numpy(dtype=bool)
            )
        for i in np.flatnonzero(~equal):
            date, secid = common[i]
            a, b = _json_value(before.iloc[i]), _json_value(after.iloc[i])
            maturity = field in LABEL_FIELDS and maturing[i] and (
                field == "label_mature" or (a is None and type(b) in (int, float))
            )
            diff.add("label_maturity" if maturity else "historical", date, secid, field, "value", a, b)


def open_details(folder: Path, year: int, table: str):
    return gzip.open(folder / f"{year}-{table}.csv.gz", "wt", encoding="utf-8", newline="")


def write_summary(folder, before, after, years, dependencies):
    with gzip.open(folder / "dependencies.json.gz", "wt", encoding="utf-8") as out:
        json.dump(dependencies, out, ensure_ascii=False)
    (folder / "summary.json").write_text(
        json.dumps(dict(previous_generation=before, generation=after, rtol=RTOL, atol=ATOL,
                        fingerprint_fields=["size_bytes", "mtime_ns", "ctime_ns"],
                        years=years), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
