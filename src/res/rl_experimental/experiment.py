"""One-command server experiment orchestration."""
from __future__ import annotations

import json
import html
import platform
import shutil
import subprocess
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch

from .bundle import create_bundle, send_bundle
from .data import PanelData
from .real_data import RealDataConfig, prepare_real_data
from .report import generate_report
from .reward import RewardConfig
from .training import ExperimentConfig, train_experiment


DEFAULT_RUN_CONFIG: dict[str, Any] = {
    "alpha": "pred@gru_day_V1",
    "start": 20250102,
    "end": 20250228,
    "alpha_sample_status": "unknown",
    "snapshot_dir": "results/rl_experimental/snapshots/pred_gru_day_V1_20250102_20250228",
    "output_root": "results/rl_experimental/server_runs",
    "rebuild_snapshot": False,
    "bundle_max_mb": 10,
    "recipient": None,
    "training": {
        "total_timesteps": 4096,
        "seed": 7,
        "device": "cuda",
        "eval_freq": 0,
        "reward": {
            "turnover_penalty": 0.0,
            "downside_penalty": 0.0,
            "concentration_penalty": 0.0,
            "custom_function": None,
            "custom_params": {},
        },
        "selection_metric": "nav",
    },
}


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _git_revision() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() or None


def _system_info() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_revision": _git_revision(),
    }


def load_run_config(path: str | Path) -> dict[str, Any]:
    supplied = json.loads(Path(path).read_text(encoding="utf-8"))
    config = DEFAULT_RUN_CONFIG | supplied
    config["training"] = DEFAULT_RUN_CONFIG["training"] | supplied.get("training", {})
    config["training"]["reward"] = DEFAULT_RUN_CONFIG["training"]["reward"] | supplied.get("training", {}).get("reward", {})
    required = ("alpha", "start", "end", "snapshot_dir", "output_root")
    if any(config.get(key) in (None, "") for key in required):
        raise ValueError(f"run config requires {required}")
    return config


def _snapshot(config: dict[str, Any]) -> tuple[PanelData, Path]:
    directory = Path(config["snapshot_dir"]).resolve()
    data = RealDataConfig(
        alpha=config["alpha"], start=int(config["start"]), end=int(config["end"]), output=directory,
        alpha_direction=int(config.get("alpha_direction", 1)),
        alpha_lag=int(config.get("alpha_lag", 0)),
        max_alpha_staleness=int(config.get("max_alpha_staleness", 0)),
        min_listing_days=int(config.get("min_listing_days", 63)),
        alpha_sample_status=config.get("alpha_sample_status", "unknown"),
    )
    required = [directory / name for name in ("panel.npz", "manifest.json", "quality_report.json")]
    if config.get("rebuild_snapshot") or not any(path.exists() for path in required):
        prepare_real_data(data)
    elif not all(path.is_file() for path in required):
        raise ValueError("snapshot is incomplete; enable rebuild_snapshot after checking the directory")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    actual = manifest["config"]
    expected = asdict(data)
    for key, value in expected.items():
        expected_value = str(value) if isinstance(value, Path) else value
        if actual.get(key) != expected_value:
            raise ValueError(f"snapshot config mismatch for {key}: {actual.get(key)!r} != {expected_value!r}")
    panel = PanelData.load_npz(directory / "panel.npz")
    if panel.contract() != manifest["panel_contract"]:
        raise ValueError("snapshot panel contract does not match manifest")
    return panel, directory


def _experiment_config(config: dict[str, Any]) -> ExperimentConfig:
    values = asdict(ExperimentConfig()) | config["training"]
    values["reward"] = RewardConfig.from_value(values["reward"])
    experiment = ExperimentConfig(**values)
    if experiment.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("server config requests CUDA but torch.cuda.is_available() is false")
    return experiment


def run_experiment(config_path: str | Path, *, no_email: bool = False, recipient: str | None = None) -> dict[str, Any]:
    config = load_run_config(config_path)
    run_id = f"{config['alpha'].replace('@', '_')}_{config['start']}_{config['end']}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    root = Path(config["output_root"]).resolve()
    run = root / "runs" / run_id
    run.mkdir(parents=True, exist_ok=False)
    _write_json(run / "experiment_config.json", config)
    _write_json(run / "system_info.json", _system_info())
    phases: list[dict[str, Any]] = []
    status: dict[str, Any] = {"run_id": run_id, "status": "running", "phases": phases}
    _write_json(run / "status.json", status)
    error: Exception | None = None
    current_phase = "startup"
    with (run / "stdout.log").open("w", encoding="utf-8") as stdout_file, (run / "stderr.log").open("w", encoding="utf-8") as stderr_file:
        with redirect_stdout(_Tee(sys.stdout, stdout_file)), redirect_stderr(_Tee(sys.stderr, stderr_file)):
            try:
                current_phase = "snapshot"
                panel, snapshot = _snapshot(config)
                phases.append({"phase": "snapshot", "status": "success", "path": str(snapshot)})
                shutil.copy2(snapshot / "manifest.json", run / "data_manifest.json")
                shutil.copy2(snapshot / "quality_report.json", run / "data_quality_report.json")
                current_phase = "training"
                experiment = _experiment_config(config)
                metrics = train_experiment(panel, run, experiment)
                phases.append({"phase": "training", "status": "success", "updates": metrics["completed_updates"]})
                current_phase = "report"
                report = generate_report(run)
                phases.append({"phase": "report", "status": "success", "path": report.name})
                status.update({"status": "success", "completed_at": datetime.now(timezone.utc).isoformat()})
            except Exception as exc:
                error = exc
                trace = traceback.format_exc()
                stderr_file.write(trace)
                phases.append({"phase": current_phase, "status": "failed", "error": str(exc)})
                status.update({
                    "status": "failed", "completed_at": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(exc).__name__, "error": str(exc),
                })
                (run / "failure_report.html").write_text(
                    "<!doctype html><meta charset='utf-8'><title>RL experiment failed</title>"
                    f"<h1>Experiment failed</h1><p>{html.escape(type(exc).__name__)}: {html.escape(str(exc))}</p>"
                    "<p>See stderr.log and status.json in this bundle.</p>", encoding="utf-8",
                )
            finally:
                _write_json(run / "status.json", status)
    metadata = create_bundle(run, root / "bundles", int(float(config["bundle_max_mb"]) * 1024 * 1024))
    status["bundle_metadata"] = str(metadata)
    if not no_email:
        status["delivery"] = send_bundle(metadata, recipient or config.get("recipient"))
    _write_json(root / "bundles" / f"{run_id}.run_status.json", status)
    if error is not None:
        raise RuntimeError(f"experiment {run_id} failed; diagnostic bundle: {metadata}") from error
    return status
