"""Paired replay of factor evaluation and FMP; never retrains neural weights."""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter
import pandas as pd

from src.res.factor.training_analysis_snapshot import (
    TrainingAnalysisSnapshot,
    return_label_scope,
)
from src.res.model.diagnostics.t50_benchmark import load_inputs
from src.res.factor.util import StockFactor
from src.res.factor.analytic.factor_perf import FactorPerfTest
from src.res.factor.fmp.builder import PortfolioGroupBuilder
from src.res.factor.fmp.generator_array import run_generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["previous", "snapshot"])
    parser.add_argument("--start", type=int, default=20240101)
    parser.add_argument("--end", type=int, default=20241231)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--previous-market",
        default="results/nn_training_review/generator_stage2/year/market",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode:
        alpha, universe, meta = load_inputs(
            "gru_day_V1", "top-1000", args.start, args.end
        )
        dates = alpha.available_dates().dates
        factor = StockFactor(
            pd.concat(
                [
                    alpha.get(int(d))
                    .to_dataframe()
                    .rename(columns={"alpha": "best"})
                    .assign(date=d)
                    for d in dates[::5]
                ],
                ignore_index=True,
            )
        )
        started = perf_counter()
        snapshot = None
        scope = return_label_scope()
        if args.mode == "snapshot":
            snapshot = TrainingAnalysisSnapshot(dates=dates, allow_partial=True)
            scope = snapshot.activate()
        load_seconds = perf_counter() - started
        with scope:
            at = perf_counter()
            if snapshot is None:
                factor.day_returns()
            factor.within_benchmarks()
            preload = perf_counter() - at
            task = FactorPerfTest(start=args.start, end=args.end, vb_level="never")
            at = perf_counter()
            task.calc(factor)
            calc = perf_counter() - at
            group = PortfolioGroupBuilder(
                "top",
                alpha,
                [universe],
                analytic=False,
                attribution=False,
                trade_engine="yale",
                start=args.start,
                end=args.end,
                vb_level="never",
            )
            at = perf_counter()
            account = run_generator(
                group,
                "array",
                prepared_root=args.previous_market if snapshot is None else None,
            )
            fmp = perf_counter() - at
            elapsed = perf_counter() - started
        results = task.get_rslts()
        results["account"] = account
        for key, frame in results.items():
            frame.to_pickle(args.output / f"{args.mode}-{key}.pkl")
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "snapshot_open": load_seconds,
                    "preload": preload,
                    "factor_calc": calc,
                    "fmp": fmp,
                    "fmp_breakdown": group.generator_diagnostics,
                    "total": elapsed,
                    "keys": list(results),
                    "hits": snapshot.hits if snapshot else 0,
                    "snapshot_generation": snapshot.generation if snapshot else None,
                    "date_count": len(dates),
                }
            )
        )
        return
    report = {
        "runs": [],
        "scope": "Factor preparation, factor_perf_test.calc and top FMP replay; common prediction/universe loading, plotting/export and neural network fitting excluded. Previous mode uses stage-2 NPZ market backend; snapshot mode uses annual Feather and default evaluation inputs. Local historical warnst incomplete.",
    }
    for i in range(args.repeats):
        for mode in (
            ["previous", "snapshot"] if i % 2 == 0 else ["snapshot", "previous"]
        ):
            cmd = [
                sys.executable,
                "-m",
                "src.res.model.diagnostics.training_snapshot_benchmark",
                "--output",
                str(args.output),
                "--mode",
                mode,
                "--start",
                str(args.start),
                "--end",
                str(args.end),
                "--previous-market",
                args.previous_market,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode:
                raise RuntimeError(proc.stdout[-3000:] + proc.stderr[-5000:])
            row = json.loads(proc.stdout.strip().splitlines()[-1])
            row["repeat"] = i
            report["runs"].append(row)
            print(row, flush=True)
        for key in row["keys"]:
            pd.testing.assert_frame_equal(
                pd.read_pickle(args.output / f"previous-{key}.pkl"),
                pd.read_pickle(args.output / f"snapshot-{key}.pkl"),
                check_exact=False,
                rtol=1e-6,
                atol=1e-8,
            )
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
    report["equivalent"] = True
    report["fmp_medians"] = {
        mode: {
            key: statistics.median(
                row["fmp_breakdown"][key]
                for row in report["runs"]
                if row["mode"] == mode
            )
            for key in (
                "setup_seconds",
                "preparation_seconds",
                "generation_seconds",
                "accounting_seconds",
                "output_seconds",
            )
        }
        for mode in ("previous", "snapshot")
    }
    report["medians"] = {
        m: {
            k: statistics.median(r[k] for r in report["runs"] if r["mode"] == m)
            for k in ["snapshot_open", "preload", "factor_calc", "fmp", "total"]
        }
        for m in ["previous", "snapshot"]
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(report["medians"])


if __name__ == "__main__":
    main()
