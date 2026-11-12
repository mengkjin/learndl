"""Isolated synthetic GRU training memory experiments.

# Synthetic GRU CUDA memory benchmark

Run from the repository root on the CUDA server:

```bash
.venv/bin/python scripts/5_test/benchmark_gru_memory.py
```

The default run measures 22 cases: 5,000 stocks × 30 days × 16 bars × 6 features,
intraday hidden sizes 32 and 64, two GRU layers, dropout 0.1, FP32, and three
complete training steps per case. The day GRU always has hidden size 64.
Weights and data are random. The head mirrors the default `gru_gru` linear /
BatchNorm / linear / BatchNorm structure, using synthetic MSE and Adam.
The script does not load production data or models, change worklists, or run
the training scheduler. Run on an otherwise idle GPU for comparable results.

Each case uses a fresh subprocess; an OOM is recorded and the next case starts.
Cases have a 1,200-second timeout, configurable with `--timeout`. Default modes:

- `full`: all stock/day sequences in one GRU call.
- `full_checkpoint`: one checkpoint around that entire intraday encoder.
- `chunk`: ordinary chunks of 50,000, 25,000 or 12,500 stock/day sequences.
- `chunk_outer_checkpoint`: one checkpoint around the entire chunk loop.
- `chunk_checkpoint`: a separate checkpoint around each chunk.

All modes preserve stock/day ordering and the full-stock day GRU, BatchNorm and
loss. Checkpoint uses `use_reentrant=False, preserve_rng_state=True`; it is active
immediately in the selected cases (not waiting for an OOM callback).
The day GRU is uncheckpointed in all cases by default. `--day-checkpoint` enables
it equally across all modes, allowing a second comparison closer to production
recovery. Dropout assignments may differ between chunk sizes, so outputs are not
expected to be bitwise equal across these performance cases.

Encoder results use a compact clone of the last time step in all default modes.
This avoids an output view retaining full sequence storage and does not detach
gradients. `--retain-output-view` runs the corresponding original-view experiment;
use this with `full` / `full_checkpoint` when examining the existing encoder.

## Larger workload and resident data simulation

```bash
.venv/bin/python scripts/5_test/benchmark_gru_memory.py \
  --stocks 5000 6000 --hidden 64 --resident-gib 8 --day-checkpoint
```

`--resident-gib 8` allocates and touches an additional 8 GiB GPU buffer, held
throughout each case. This simulates a fixed baseline, not production allocator
fragmentation or data pipeline behavior. It is included in reported peaks. Set
it using observed resident allocations, not the total training peak (which would
double-count activations). Avoid concurrent training: external GPU use affects
available capacity but is not included in this process's allocated-memory peak.

Useful options: `--steps 5`, `--recipient ADDRESS`, `--device cuda:1`, `--no-email`,
`--output-dir /absolute/new/directory`. The output directory must not already exist.
To keep running after disconnecting, use a terminal multiplexer or:

```bash
nohup .venv/bin/python -u scripts/5_test/benchmark_gru_memory.py > /tmp/gru_memory_benchmark.log 2>&1 &
```

## Results and email

Results default to `logs/gru_memory/<UTC timestamp>/`:

- `summary.txt`: readable results, including OOM/error stage.
- `results.csv`: comparison rows.
- `results.json`: configuration, environment, per-stage allocations and timings.
- `case_*.json`, `case_*.result.json`, `case_*.log`: isolated worker inputs/results/logs.
- `mail/`: existing project `TaskResultMail` delivery receipts and prepared payloads.

After the experiment, the script sends the summary and three report attachments
using the server's existing email credentials. The default recipient is the
configured sender's own address. No credentials are embedded in this script.
Partial results are also mailed after handled errors or interruption. SIGKILL,
power loss, or host failure cannot guarantee a final email; completed cases are
saved incrementally. Disabled email or an unconfirmed send is explicitly reported
and returns a nonzero exit status. For an SMTP failure with a prepared receipt,
retry only delivery, without rerunning the benchmark:

```bash
.venv/bin/python -m src.proj.util.web.task_mail /absolute/results/directory/mail
```

The benchmark exits zero after a completed experiment even if some/all cases OOM;
OOM is an expected measurement outcome. Worker errors, timeouts and email failures
return nonzero. A successful mail means SMTP acceptance, not verified inbox delivery.

## Interpreting measurements

Peak allocated/reserved memory includes setup and all steps, including the cold
first step and lazy Adam state initialization. Each step resets peak statistics
after clearing gradients; stage peaks are cumulative within that step. Allocated
memory at forward completion helps reveal retained autograd state, but includes
the fixed baseline and all other live tensors. It is not a direct measurement of
cuDNN reserve space. Warm step time is the median excluding step 0 and includes
explicit GPU synchronization at each phase.

An OOM peak is only the allocation achieved before failure: the failed request is
not included. Do not label a smaller OOM peak as a successful memory reduction.
Compare successful cases at identical dimensions, baseline and day checkpoint
settings. Lower reserved memory is not necessarily lower live memory. JSON also
records device free/total memory, PyTorch/CUDA/cuDNN versions and allocator config.

This experiment measures synthetic capacity and the memory/time tradeoff. It does
not prove that production training fits, nor find an exact maximum batch size.
Use `--stocks` with several values to bracket the capacity threshold, then validate
the chosen implementation against the real failing workload.

"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any

import torch
import torch.version as torch_version
from torch import nn
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[2]
GIB = 1024**3
MODES = ("full", "full_checkpoint", "chunk", "chunk_outer_checkpoint", "chunk_checkpoint")


class SyntheticGRU(nn.Module):
    """Two GRUs and the default gru_gru head, with random weights and MSE loss.

Keep this independent of project imports so data/config initialization cannot
contaminate GPU measurements. BatchNorm and loss always see all stocks.
"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        h, day_h = config["hidden"], config["day_hidden"]
        kwargs = dict(num_layers=config["layers"], batch_first=True,
                      dropout=config["dropout"] if config["layers"] > 1 else 0.)
        self.intraday = nn.GRU(config["features"], h, **kwargs)
        self.days = nn.GRU(h, day_h, **kwargs)
        self.head = nn.Sequential(nn.Linear(day_h, day_h), nn.BatchNorm1d(day_h),
                                  nn.Linear(day_h, 1), nn.BatchNorm1d(1))

    def encode(self, x):
        last = self.intraday(x)[0][:, -1]
        # Keep the returned checkpoint boundary compact rather than retaining
        # full sequence storage through a view. This does not detach gradients.
        return last.clone() if self.config["compact_output"] else last

    def encode_chunks(self, flat):
        outputs = []
        for part in flat.split(self.config["chunk_size"], dim=0):
            if self.config["mode"] == "chunk_checkpoint":
                outputs.append(checkpoint(self.encode, part, use_reentrant=False,
                                          preserve_rng_state=True))
            else:
                outputs.append(self.encode(part))
        return torch.cat(outputs, dim=0)

    def encode_days(self, encoded):
        return self.days(encoded)[0][:, -1]

    def forward(self, x):
        stocks, days, bars, features = x.shape
        flat = x.reshape(-1, bars, features)
        mode = self.config["mode"]
        if mode == "full":
            encoded = self.encode(flat)
        elif mode == "full_checkpoint":
            encoded = checkpoint(self.encode, flat, use_reentrant=False, preserve_rng_state=True)
        elif mode == "chunk_outer_checkpoint":
            encoded = checkpoint(self.encode_chunks, flat, use_reentrant=False, preserve_rng_state=True)
        else:
            encoded = self.encode_chunks(flat)
        assert isinstance(encoded, torch.Tensor)
        encoded = encoded.reshape(stocks, days, -1)
        # Fixed across modes to isolate the intraday strategy.
        if self.config["day_checkpoint"]:
            hidden = checkpoint(self.encode_days, encoded, use_reentrant=False, preserve_rng_state=True)
        else:
            hidden = self.encode_days(encoded)
        return self.head(hidden)


def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def memory(device):
    if device.type != "cuda":
        return {}
    free, total = torch.cuda.mem_get_info(device)
    return dict(allocated_gib=torch.cuda.memory_allocated(device) / GIB,
                reserved_gib=torch.cuda.memory_reserved(device) / GIB,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device) / GIB,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device) / GIB,
                device_free_gib=free / GIB, device_total_gib=total / GIB)


def run_case(config, device_name="cuda:0"):
    """One process per case; include cold step and initialized Adam steps."""
    result: dict[str, Any] = dict(config=config, status="running", stage="setup", steps=[])
    device = torch.device(device_name)
    try:
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; no CPU fallback for memory measurements")
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
            result["gpu"] = torch.cuda.get_device_name(device)
        torch.manual_seed(config["seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        result["environment"] = dict(torch=torch.__version__, cuda=torch_version.cuda,
                                     cudnn=torch.backends.cudnn.version(),
                                     allocator=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                                     tf32=False, cudnn_benchmark=False)
        result["initial_memory"] = memory(device)
        # Deliberately retained until the entire case finishes.
        resident = torch.empty(int(config["resident_gib"] * GIB), dtype=torch.uint8, device=device)
        resident.zero_()
        model = SyntheticGRU(config).to(device).train()
        x = torch.randn(config["stocks"], config["days"], config["bars"],
                        config["features"], device=device)
        target = torch.randn(config["stocks"], 1, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        result["setup_memory"] = memory(device)
        for index in range(config["steps"]):
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            step: dict[str, Any] = dict(index=index, phases={}, baseline=memory(device))
            result["steps"].append(step)
            total_start = time.perf_counter()
            for phase in ("forward", "backward", "optimizer"):
                result["stage"] = phase
                start = time.perf_counter()
                if phase == "forward":
                    prediction = model(x)
                    loss = (prediction - target).square().mean()
                elif phase == "backward":
                    loss.backward()
                else:
                    optimizer.step()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                step["phases"][phase] = dict(seconds=time.perf_counter() - start, **memory(device))
            step["seconds"] = time.perf_counter() - total_start
            step["loss"] = loss.item()
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite synthetic loss")
            del prediction, loss
        result.update(status="ok", stage="complete")
    except torch.cuda.OutOfMemoryError as error:
        result.update(status="oom", error=str(error))
    except Exception as error:
        result.update(status="error", error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc())
    finally:
        # This is achieved allocation, NOT the amount needed to finish an OOM case.
        try:
            result["final_memory"] = memory(device) if device.type != "cuda" or torch.cuda.is_initialized() else {}
        except Exception:
            result["final_memory"] = {}
    return result


def make_cases(args):
    for stocks in args.stocks:
        for hidden in args.hidden:
            for mode in args.modes:
                sizes = args.chunks if mode.startswith("chunk") else [0]
                for size in sizes:
                    yield dict(stocks=stocks, hidden=hidden, mode=mode, chunk_size=size,
                               days=args.days, bars=args.bars, features=args.features,
                               layers=args.layers, day_hidden=args.day_hidden, dropout=args.dropout,
                               steps=args.steps, seed=args.seed, resident_gib=args.resident_gib,
                               compact_output=not args.retain_output_view,
                               day_checkpoint=args.day_checkpoint)


def summarize(result):
    row = {key: result["config"][key] for key in ("stocks", "hidden", "mode", "chunk_size")}
    row.update(status=result["status"], stage=result.get("stage", "unknown"))
    samples = [result.get("setup_memory", {}), result.get("final_memory", {})]
    for step in result.get("steps", []):
        samples.extend(step["phases"].values())
    for key in ("peak_allocated_gib", "peak_reserved_gib"):
        values = [sample[key] for sample in samples if key in sample]
        row[key] = max(values) if values else None
    steps = result.get("steps", [])
    times = [step["seconds"] for step in steps[1:] if "seconds" in step]
    row["warm_step_seconds"] = statistics.median(times) if times else None
    forward = [step["phases"]["forward"]["allocated_gib"] for step in steps
               if "allocated_gib" in step["phases"].get("forward", {})]
    row["forward_retained_gib"] = max(forward) if forward else None
    row["error"] = result.get("error", "")
    return row


def write_reports(directory, report):
    save_json(directory / "results.json", report)
    rows = [summarize(case) for case in report["cases"]]
    if rows:
        with (directory / "results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["Synthetic GRU memory benchmark", f"Host: {report['host']}",
             f"Status: {report['status']}", f"Output: {directory}",
             "FP32, random weights/data, two GRUs + full-stock BatchNorm head + MSE + Adam.",
             "Not production acceptance: no real data cache, production loss or trainer.",
             "OOM peaks are achieved allocations (lower bounds), not required capacity.",
             "Phase peaks in JSON are cumulative within each step; times include CUDA synchronization.",
             "Warm times exclude step 0; peaks include setup and ALL steps.", ""]
    for row in rows:
        def fmt(key):
            value = row[key]
            return "n/a" if value is None else f"{value:.3f}"
        lines.append(f"stocks={row['stocks']} H={row['hidden']} {row['mode']} chunk={row['chunk_size']}: "
                     f"{row['status']} ({row['stage']}), peak={fmt('peak_allocated_gib')} GiB, "
                     f"reserved={fmt('peak_reserved_gib')} GiB, "
                     f"forward retained={fmt('forward_retained_gib')} GiB, "
                     f"warm step={fmt('warm_step_seconds')} s")
        if row["error"]:
            lines.append("  " + row["error"])
    if report.get("error"):
        lines.append(report["error"])
    (directory / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def send_results(directory, recipient):
    # Project imports (and SMTP initialization) only happen in the coordinator,
    # AFTER every GPU worker has exited. Use the existing durable mail receipts.
    sys.path.insert(0, str(ROOT))
    from src.proj.env import MACHINE
    from src.proj.util.web.task_mail import TaskResultMail
    if not MACHINE.emailable:
        raise RuntimeError("Email disabled by machine policy; results saved but NOT emailed")
    body = (directory / "summary.txt").read_text(encoding="utf-8")
    attachments = [directory / name for name in ("results.json", "results.csv", "summary.txt")
                   if (directory / name).exists()]
    delivery = directory / "mail"
    if not TaskResultMail(delivery)("GRU CUDA memory benchmark", body, recipient, attachments=attachments):
        raise RuntimeError(f"Mail failed; retry delivery only: .venv/bin/python -m "
                           f"src.proj.util.web.task_mail {delivery}")


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--stocks", type=int, nargs="+", default=[5000])
    cli.add_argument("--hidden", type=int, nargs="+", default=[32, 64])
    cli.add_argument("--chunks", type=int, nargs="+", default=[50000, 25000, 12500])
    cli.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    for name, default in (("days", 30), ("bars", 16), ("features", 6), ("layers", 2),
                          ("day-hidden", 64), ("steps", 3), ("seed", 20261105)):
        cli.add_argument(f"--{name}", type=int, default=default)
    cli.add_argument("--dropout", type=float, default=.1)
    cli.add_argument("--resident-gib", type=float, default=0.)
    cli.add_argument("--day-checkpoint", action="store_true", help="Same day-GRU checkpoint policy for all cases")
    cli.add_argument("--retain-output-view", action="store_true", help="Use original last-step view instead of compact clone")
    cli.add_argument("--device", default="cuda:0")
    cli.add_argument("--timeout", type=float, default=1200., help="Seconds allowed per isolated case")
    cli.add_argument("--output-dir", type=Path)
    cli.add_argument("--recipient", help="Default: project configured sender's own address")
    cli.add_argument("--no-email", action="store_true")
    cli.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    return cli


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.worker_config:
        config = json.loads(args.worker_config.read_text())
        result = run_case(config, args.device)
        save_json(args.worker_config.with_suffix(".result.json"), result)
        return 0
    dimensions = args.stocks + args.hidden + args.chunks + [args.days, args.bars, args.features,
                                                          args.layers, args.day_hidden]
    if min(dimensions) < 1 or min(args.stocks) < 2 or args.steps < 2:
        cli.error("Positive dimensions, at least 2 stocks (BatchNorm) and 2 steps are required")
    if not 0 <= args.dropout < 1 or args.resident_gib < 0 or args.timeout <= 0:
        cli.error("Invalid dropout, resident memory or timeout")
    if not args.device.startswith("cuda"):
        cli.error("Memory benchmark requires CUDA; CPU is only used by unit tests")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = (args.output_dir or ROOT / "logs" / "gru_memory" / stamp).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = dict(host=socket.gethostname(), created_utc=stamp, status="running", cases=[],
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    exit_code = 0
    try:
        cases = list(make_cases(args))
        for index, config in enumerate(cases):
            print(f"[{index + 1}/{len(cases)}] {config}", flush=True)
            path = directory / f"case_{index:03d}.json"
            save_json(path, config)
            command = [sys.executable, str(Path(__file__).resolve()), "--worker-config", str(path),
                       "--device", args.device]
            with path.with_suffix(".log").open("w") as log:
                try:
                    completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                               timeout=args.timeout, check=False)
                    result_path = path.with_suffix(".result.json")
                    if completed.returncode == 0 and result_path.exists():
                        result = json.loads(result_path.read_text())
                    else:
                        result = dict(config=config, status="error", stage="worker",
                                      error=f"Worker exit {completed.returncode}; see {log.name}")
                except subprocess.TimeoutExpired:
                    result = dict(config=config, status="timeout", stage="worker",
                                  error=f"Exceeded {args.timeout} seconds; worker killed")
            report["cases"].append(result)
            write_reports(directory, report)
            print(summarize(result), flush=True)
        exit_code = int(any(case["status"] in ("error", "timeout") for case in report["cases"]))
        report["status"] = "completed_with_errors" if exit_code else "completed"
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        exit_code = 1
    finally:
        write_reports(directory, report)
        print(f"Results: {directory}", flush=True)
        if not args.no_email:
            try:
                send_results(directory, args.recipient)
                print("Result email accepted by SMTP.", flush=True)
            except Exception as error:
                print(f"Email NOT confirmed: {error}", file=sys.stderr, flush=True)
                exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
