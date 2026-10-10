from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .data import PanelData, date_split, make_synthetic_panel


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Experimental RL portfolio construction")
    sub = result.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-real", help="audit and export a point-in-time learndl snapshot")
    prepare.add_argument("--alpha", required=True, help="source@name or source@name@column")
    prepare.add_argument("--start", type=int, required=True)
    prepare.add_argument("--end", type=int, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--alpha-direction", type=int, choices=(-1, 1), default=1)
    prepare.add_argument("--alpha-lag", type=int, default=0)
    prepare.add_argument("--max-alpha-staleness", type=int, default=0)
    prepare.add_argument("--min-listing-days", type=int, default=63)
    prepare.add_argument("--alpha-sample-status", choices=("unknown", "out-of-sample", "in-sample"), default="unknown")
    prepare.add_argument("--dry-run", action="store_true")

    for name in ("demo", "train", "train-suite"):
        command = sub.add_parser(name)
        command.add_argument("--output", type=Path, default=Path("results/rl_experimental"))
        command.add_argument("--timesteps", type=int, default=100_000 if name == "train-suite" else 4_096)
        command.add_argument("--device", default="cpu")
        command.add_argument("--seed", type=int, default=7)
        command.add_argument("--eval-freq", type=int, default=0, help="0: validate after every PPO update; positive: environment-step interval")
        command.add_argument("--train-end-date", type=int)
        command.add_argument("--valid-end-date", type=int)
        if name == "demo":
            command.add_argument("--days", type=int, default=320)
            command.add_argument("--stocks", type=int, default=200)
        else:
            command.add_argument("data", type=Path)
        if name == "train-suite":
            command.add_argument("--seeds", type=int, nargs="+", default=(7, 17, 29))

    baseline = sub.add_parser("replay-baselines", help="run both no-training baselines")
    baseline.add_argument("data", type=Path)
    baseline.add_argument("--output", type=Path, required=True)
    baseline.add_argument("--train-end-date", type=int)
    baseline.add_argument("--valid-end-date", type=int)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("data", type=Path)
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("--output", type=Path, default=Path("results/rl_experimental/evaluation"))

    run = sub.add_parser("run-experiment", help="prepare, train, report, bundle and optionally email one experiment")
    run.add_argument("config", type=Path)
    run.add_argument("--no-email", action="store_true")
    run.add_argument("--recipient")

    analyze = sub.add_parser("analyze-bundle", help="verify and extract a portable result bundle")
    analyze.add_argument("bundle", type=Path)
    analyze.add_argument("--output", type=Path, required=True)

    resend = sub.add_parser("resend-bundle", help="send only bundle parts not previously delivered")
    resend.add_argument("metadata", type=Path)
    resend.add_argument("--recipient")

    report = sub.add_parser("generate-report", help="generate the offline Chinese report for a completed run")
    report.add_argument("run", type=Path)
    report.add_argument("--output", type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "run-experiment":
        from .experiment import run_experiment

        print(json.dumps(run_experiment(args.config, no_email=args.no_email, recipient=args.recipient), indent=2, ensure_ascii=False))
        return

    if args.command == "analyze-bundle":
        from .bundle import analyze_bundle

        print(analyze_bundle(args.bundle, args.output))
        return

    if args.command == "resend-bundle":
        from .bundle import send_bundle

        print(json.dumps(send_bundle(args.metadata, args.recipient, only_failed=True), indent=2, ensure_ascii=False))
        return

    if args.command == "generate-report":
        from .report import generate_report

        print(generate_report(args.run, args.output))
        return
    if args.command == "prepare-real":
        # Kept lazy so this command does not import Gymnasium, SB3, or Torch.
        from .real_data import RealDataConfig, prepare_real_data

        config = RealDataConfig(
            alpha=args.alpha,
            start=args.start,
            end=args.end,
            output=args.output,
            alpha_direction=args.alpha_direction,
            alpha_lag=args.alpha_lag,
            max_alpha_staleness=args.max_alpha_staleness,
            min_listing_days=args.min_listing_days,
            alpha_sample_status=args.alpha_sample_status,
        )
        print(json.dumps(prepare_real_data(config, dry_run=args.dry_run), indent=2, ensure_ascii=False))
        return

    if args.command == "evaluate":
        from .training import evaluate_checkpoint

        metrics = evaluate_checkpoint(PanelData.load_npz(args.data), args.checkpoint, args.output)
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "replay-baselines":
        from .baselines import history_metrics, replay_baseline, write_history
        from .portfolio import PortfolioConstraints

        panel = PanelData.load_npz(args.data)
        split = date_split(panel, args.train_end_date, args.valid_end_date)["test"]
        args.output.mkdir(parents=True, exist_ok=True)
        metrics = {}
        for kind in ("alpha_equal_weight", "robust_top50"):
            history = replay_baseline(panel, split, PortfolioConstraints(), kind)
            write_history(args.output / f"{kind}_history.csv", history)
            metrics[kind] = history_metrics(history)
        (args.output / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        return

    from .training import ExperimentConfig, train_experiment, train_seed_suite

    panel = make_synthetic_panel(n_steps=args.days, n_stocks=args.stocks, seed=args.seed) if args.command == "demo" else PanelData.load_npz(args.data)
    config = replace(
        ExperimentConfig(),
        total_timesteps=args.timesteps,
        device=args.device,
        seed=args.seed,
        eval_freq=args.eval_freq,
        train_end_date=args.train_end_date,
        valid_end_date=args.valid_end_date,
    )
    if args.command == "train-suite":
        metrics = train_seed_suite(panel, args.output, config, tuple(args.seeds))
        printable = metrics
    else:
        metrics = train_experiment(panel, args.output, config)
        printable = {
            "test": metrics["test"],
            "alpha_equal_weight_baseline": metrics["alpha_equal_weight_baseline"],
            "robust_top50_baseline": metrics["robust_top50_baseline"],
            "timing": metrics["timing"],
        }
    print(json.dumps(printable, indent=2, default=lambda value: value.tolist()))


if __name__ == "__main__":
    main()
