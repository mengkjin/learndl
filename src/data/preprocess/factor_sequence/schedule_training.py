"""One ordinary schedule task and model directory per candidate across all historical dates."""
from __future__ import annotations

__all__ = [
    'schedule_config',
    'variants',
    'train',
    'export_folds',
    'main',
]

from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
import pandas as pd
import yaml
from .bundle import load, write_json, file_digest
from .pipeline import train_overrides, read_predictions, compare


def schedule_config(dataset, name, model, folds, research=False, seed=17, variant=None):
    first = folds[0]
    config = train_overrides(
        dataset, name, model, first, first["model_date"], research, seed
    )
    config.update(
        {
            "model.end_date": folds[-1]["test"][-1],
            "model.interval": 63,
            "train.factor_sequence.train_dates": [],
            "train.factor_sequence.valid_dates": [],
            "train.factor_sequence.folds": folds,
        }
    )
    for key, value in (variant or {}).items():
        config[f"algo.{model}.{key}"] = [value]
    return config


def variants(ablations=False):
    from src.res.algo.nn.model.factor_sequence import MODELS

    values = [(model, {}) for model in MODELS]
    if ablations:
        values += [
            ("factorseq_m1", {"use_performance": False}),
            ("factorseq_m1", {"use_market": False}),
            ("factorseq_m2", {"modulation": False}),
        ]
    return values


def train(
    dataset,
    suite_folder,
    model,
    seed,
    variant,
    research=False,
    email=True,
    recipient=None,
    resume_suite=None,
):
    from src.proj import PATH, CALENDAR, Save
    from src.proj.util.cli.script_session import as_script_main
    from src.proj.util.filesys.dynamic_import import dynamic_modules
    from src.res.model.util.schedule_resume import find_schedule_runs

    suite_folder = Path(suite_folder)
    plan = json.loads((suite_folder / "walk_forward.json").read_text())
    folds = [
        {**f, "model_date": int(CALENDAR.td(f["test"][0], -1))} for f in plan["folds"]
    ]
    suffix = "_".join(variant) if variant else "full"
    name = f"fs_{suite_folder.name}_{model}_{suffix}_{seed}"
    previous_result = None
    if resume_suite:
        matches = [
            json.loads(p.read_text())
            for p in (Path(resume_suite) / "models").glob("*/run.json")
        ]
        matches = [
            r
            for r in matches
            if (r["model"], r["seed"], r["variant"]) == (model, seed, variant)
        ]
        if len(matches) > 1:
            raise ValueError("Ambiguous previous schedule identity")
        previous_result = matches[0] if matches else None
        if previous_result:
            name = previous_result["name"]
    folder = suite_folder / "models" / name
    folder.mkdir(parents=True, exist_ok=False)
    config = schedule_config(dataset, name, model, folds, research, seed, variant)
    config_text = yaml.safe_dump(config, sort_keys=False)
    (folder / "schedule.yaml").write_text(config_text)
    schedule = PATH.sched_shared / f"{name}.yaml"
    schedule.parent.mkdir(parents=True, exist_ok=True)
    existing = find_schedule_runs(name, short_test=False)
    if previous_result:
        if (
            len(existing) != 1
            or not schedule.is_file()
            or schedule.read_text() != config_text
        ):
            raise ValueError(
                "Resume requires one existing model root and unchanged schedule config"
            )
        if previous_result["data_version"] != load(dataset)["version"]:
            raise ValueError("Resume dataset changed")
    else:
        if schedule.exists() or existing:
            raise FileExistsError(
                f"Schedule already exists: {name}; use normal schedule resume with its model path"
            )
        with schedule.open("x") as stream:
            stream.write(config_text)
    result = {
        "model": model,
        "seed": seed,
        "variant": variant,
        "name": name,
        "schedule_name": name,
        "status": "running",
        "folds": len(folds),
        "data_version": load(dataset)["version"],
        "schedule_sha256": file_digest(schedule),
    }
    write_json(folder / "run.json", result)
    script = PATH.scpt / "4_train/2_schedule_model.py"
    started = time.perf_counter()
    previous = sys.argv
    try:
        sys.argv = [str(script)]
        with as_script_main(script):
            module = next(iter(dynamic_modules(script)))
            if not previous_result or previous_result["status"] != "passed":
                # Normal schedule resume selects the sole existing root verified above.
                task = module.main(
                    schedule_name=name,
                    short_test=False,
                    resume=bool(previous_result),
                    email=email,
                    email_recipient=recipient,
                )
                if not task.success or task.error_messages:
                    raise RuntimeError(
                        task.error_message or "Schedule training did not succeed"
                    )
            else:
                result["reused_from"] = str(resume_suite)
        Save.async_wait_all(caller_name="factor_sequence_schedule")
        paths = find_schedule_runs(name, short_test=False)
        if len(paths) != 1:
            raise ValueError(f"Expected exactly one model root, got {len(paths)}")
        result.update(
            status="passed",
            model_path=str(paths[0].base),
            seconds=time.perf_counter() - started,
        )
        pred = read_predictions(paths[0].base)
        expected = [d for f in folds for d in f["test"]]
        if (
            sorted(pred.date.unique()) != expected
            or not np.isfinite(pred[["pred", "label"]]).all().all()
        ):
            raise ValueError("Schedule OOS predictions do not match the frozen plan")
        pred.to_csv(folder / "predictions.csv", index=False)
    except BaseException:
        import traceback

        result.update(
            status="failed",
            error=traceback.format_exc(),
            seconds=time.perf_counter() - started,
        )
        raise
    finally:
        sys.argv = previous
        write_json(folder / "run.json", result)
    return result


def export_folds(dataset, suite_folder, plan, results):
    """Adapt ordinary trainer predictions to the existing cross-model comparison reports."""
    from src.data import DATAVENDOR
    from src.res.algo.nn.model.factor_sequence import CATALOG

    suite_folder = Path(suite_folder)
    with np.load(Path(dataset) / "factorseq.npz", allow_pickle=False) as stock:
        secids = stock["secids"].copy()
    children = []
    for index, fold in enumerate(plan["folds"]):
        child = (
            suite_folder.parent / f"{suite_folder.name}_f{index:03d}_{fold['test'][0]}"
        )
        training = child / "training"
        training.mkdir(parents=True, exist_ok=False)
        children.append(str(child))
        write_json(training / "split.json", fold)
    # Read each model's all-period predictions once, then partition only report files.
    for result in results:
        pred = pd.read_csv(suite_folder / "models" / result["name"] / "predictions.csv")
        for child, fold in zip(children, plan["folds"]):
            part = pred.loc[pred.date.isin(fold["test"])].copy()
            part["valid"] = True
            part["data_version"] = result["data_version"]
            part["model_version"] = CATALOG[result["model"]]["version"]
            part.to_csv(Path(child) / "training" / f"{result['name']}.csv", index=False)
    for child, fold in zip(children, plan["folds"]):
        child = Path(child)
        training = child / "training"
        future = DATAVENDOR.nday_fut_ret(
            secids, np.asarray(fold["test"]), nday=10, lag=2
        )
        np.savez_compressed(
            training / "realized_returns.npz",
            values=future.values.cpu().numpy(),
            dates=future.date,
            secids=future.secid,
        )
        runs = [
            {**result, "model_version": CATALOG[result["model"]]["version"]}
            for result in results
        ]
        write_json(training / "runs.json", runs)
        write_json(
            child / "smoke.json",
            {"status": "passed", "source": "ordinary_schedule_training"},
        )
        write_json(child / "compare.json", compare(training))
    return children


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--variant", default="{}")
    parser.add_argument("--research", action="store_true")
    parser.add_argument("--no-email", action="store_true")
    parser.add_argument("--recipient")
    parser.add_argument("--resume-suite", type=Path)
    args = parser.parse_args()
    train(
        args.dataset,
        args.suite,
        args.model,
        args.seed,
        json.loads(args.variant),
        args.research,
        not args.no_email,
        args.recipient,
        args.resume_suite,
    )


if __name__ == "__main__":
    main()
