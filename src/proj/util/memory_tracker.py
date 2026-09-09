"""Opt-in process/cgroup memory telemetry with an OOM-surviving email watcher.

The sampler writes one JSON object per line and flushes every record, so the file
remains useful when the observed process is killed with SIGKILL.  On Linux, an
optional tiny ``systemd --user`` service watches the process from a sibling
cgroup and emails the report after an abnormal exit.  Normal task completion
uses the project's existing attachment/email path instead. Set
``LEARNDL_MEMORY_TRACK=1`` to enable it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

__all__ = ["ProcessMemoryTracker", "collect_memory_sample"]

_KIB_FIELDS = {
    "VmPeak", "VmSize", "VmHWM", "VmRSS", "RssAnon", "RssFile", "RssShmem",
    "VmData", "VmStk", "VmExe", "VmLib", "VmPTE", "VmSwap", "HugetlbPages",
}
_MEMINFO_FIELDS = {
    "MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "SwapCached",
    "Active", "Inactive", "Dirty", "Writeback", "AnonPages", "Mapped",
    "Shmem", "Slab", "SReclaimable", "SUnreclaim", "PageTables", "SwapTotal",
    "SwapFree", "Committed_AS",
}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return ""


def _parse_kib_file(text: str, fields: set[str]) -> dict[str, int | str]:
    parsed: dict[str, int | str] = {}
    for raw in text.splitlines():
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        if key not in fields:
            continue
        match = re.match(r"\s*(\d+)\s*kB\s*$", value)
        parsed[key] = int(match.group(1)) * 1024 if match else value.strip()
    return parsed


def _parse_psi(text: str, prefix: str) -> dict[str, float | int]:
    parsed: dict[str, float | int] = {}
    for raw in text.splitlines():
        parts = raw.split()
        if not parts:
            continue
        kind = parts[0]
        for token in parts[1:]:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            name = f"{prefix}_{kind}_{key}"
            try:
                parsed[name] = int(value) if key == "total" else float(value)
            except ValueError:
                continue
    return parsed


def _parse_scalar(text: str) -> int | str | None:
    value = text.strip()
    if not value:
        return None
    if value == "max":
        return value
    try:
        return int(value)
    except ValueError:
        return value


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _process_identity(pid: int) -> tuple[str | None, str | None]:
    """Return ``(state, start_ticks)`` from ``/proc/<pid>/stat``."""
    stat = _read_text(Path(f"/proc/{pid}/stat"))
    if not stat:
        return None, None
    # The command name may contain spaces or parentheses; fields after the last
    # ')' begin with state (field 3).  starttime is field 22, hence index 19 here.
    tail = stat.rsplit(")", 1)[-1].strip().split()
    if len(tail) <= 19:
        return None, None
    return tail[0], tail[19]


def _process_alive(pid: int, start_ticks: str | None = None) -> bool:
    state, current_start = _process_identity(pid)
    if state is None or state == "Z":
        return False
    return start_ticks is None or current_start == start_ticks


def _cgroup_dir(pid: int) -> tuple[str | None, Path | None]:
    cgroup_text = _read_text(Path(f"/proc/{pid}/cgroup"))
    for raw in cgroup_text.splitlines():
        fields = raw.split(":", 2)
        if len(fields) == 3 and fields[0] == "0":
            relative = fields[2]
            return relative, Path("/sys/fs/cgroup") / relative.lstrip("/")
    return None, None


def _add_cgroup_metrics(sample: dict[str, Any], cgroup_dir: Path, prefix: str) -> None:
    metric_files = {
        f"{prefix}_memory_current_bytes": "memory.current",
        f"{prefix}_memory_peak_bytes": "memory.peak",
        f"{prefix}_memory_low_bytes": "memory.low",
        f"{prefix}_memory_high_bytes": "memory.high",
        f"{prefix}_memory_max_bytes": "memory.max",
        f"{prefix}_swap_current_bytes": "memory.swap.current",
        f"{prefix}_swap_peak_bytes": "memory.swap.peak",
        f"{prefix}_swap_high_bytes": "memory.swap.high",
        f"{prefix}_swap_max_bytes": "memory.swap.max",
    }
    for name, filename in metric_files.items():
        value = _parse_scalar(_read_text(cgroup_dir / filename))
        if value is not None:
            sample[name] = value
    events = _read_text(cgroup_dir / "memory.events")
    for raw in events.splitlines():
        parts = raw.split()
        if len(parts) == 2 and parts[1].isdigit():
            sample[f"{prefix}_event_{parts[0]}"] = int(parts[1])
    sample.update(_parse_psi(_read_text(cgroup_dir / "memory.pressure"), f"{prefix}_psi"))


def _add_numa_metrics(sample: dict[str, Any]) -> None:
    node_root = Path("/sys/devices/system/node")
    for node_dir in sorted(node_root.glob("node[0-9]*")):
        node = node_dir.name
        for raw in _read_text(node_dir / "meminfo").splitlines():
            match = re.match(
                r"Node\s+\d+\s+(MemTotal|MemFree|MemUsed|Active|Inactive|FilePages|AnonPages|Slab|SReclaimable):\s+(\d+)\s+kB$",
                raw,
            )
            if match:
                name = match.group(1).lower()
                sample[f"numa_{node}_{name}_bytes"] = int(match.group(2)) * 1024


def collect_memory_sample(pid: int) -> dict[str, Any]:
    """Collect flat, JSON-serialisable memory metrics for ``pid``."""
    now = time.time()
    sample: dict[str, Any] = {
        "event": "sample",
        "timestamp": datetime.fromtimestamp(now).astimezone().isoformat(timespec="milliseconds"),
        "unix_time": now,
        "pid": pid,
    }

    status_text = _read_text(Path(f"/proc/{pid}/status"))
    for key, value in _parse_kib_file(status_text, _KIB_FIELDS).items():
        sample[f"process_{key.lower()}_bytes"] = value
    for raw in status_text.splitlines():
        if raw.startswith(("Threads:", "Cpus_allowed_list:", "Mems_allowed_list:")):
            key, value = raw.split(":", 1)
            name = key.lower().replace("_allowed_list", "_allowed")
            sample[f"process_{name}"] = _parse_scalar(value)
    for name in ("oom_score", "oom_score_adj"):
        value = _parse_scalar(_read_text(Path(f"/proc/{pid}/{name}")))
        if value is not None:
            sample[f"process_{name}"] = value

    for key, value in _parse_kib_file(_read_text(Path("/proc/meminfo")), _MEMINFO_FIELDS).items():
        sample[f"system_{key.lower()}_bytes"] = value
    sample.update(_parse_psi(_read_text(Path("/proc/pressure/memory")), "system_psi"))
    _add_numa_metrics(sample)

    cgroup_path, cgroup_dir = _cgroup_dir(pid)
    if cgroup_path is not None:
        sample["cgroup_path"] = cgroup_path
    if cgroup_dir is not None:
        _add_cgroup_metrics(sample, cgroup_dir, "cgroup")
        relative_parts = Path((cgroup_path or "").lstrip("/")).parts
        for index, part in enumerate(relative_parts):
            if re.fullmatch(r"user@\d+\.service", part):
                user_service_dir = Path("/sys/fs/cgroup").joinpath(*relative_parts[:index + 1])
                sample["user_service_cgroup_path"] = "/" + "/".join(relative_parts[:index + 1])
                _add_cgroup_metrics(sample, user_service_dir, "user_service_cgroup")
                break
    return sample


def _format_bytes(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value / 1024 ** 3:.2f} GiB"


def _summarize_report(report_path: Path) -> str:
    records: list[dict[str, Any]] = []
    for raw in _read_text(report_path).splitlines():
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    samples = [record for record in records if record.get("event") in {"sample", "mark", "final"}]
    if not samples:
        return "No complete memory samples were recovered."

    def maximum(key: str) -> Any:
        values = [record[key] for record in samples if isinstance(record.get(key), (int, float))]
        return max(values) if values else None

    def minimum(key: str) -> Any:
        values = [record[key] for record in samples if isinstance(record.get(key), (int, float))]
        return min(values) if values else None

    last = samples[-1]
    return "\n".join([
        f"Samples recovered: {len(samples)}",
        f"Last timestamp: {last.get('timestamp', 'unknown')}",
        f"Last phase: {last.get('phase', 'unknown')}",
        f"Peak process RSS: {_format_bytes(maximum('process_vmrss_bytes'))}",
        f"Peak process high-water RSS: {_format_bytes(maximum('process_vmhwm_bytes'))}",
        f"Peak cgroup memory: {_format_bytes(maximum('cgroup_memory_current_bytes'))}",
        f"Minimum system available memory: {_format_bytes(minimum('system_memavailable_bytes'))}",
        f"Peak system PSI some avg10: {maximum('system_psi_some_avg10')}",
        f"Peak cgroup PSI some avg10: {maximum('cgroup_psi_some_avg10')}",
        f"Peak user-service PSI some avg10: {maximum('user_service_cgroup_psi_some_avg10')}",
        f"Last cgroup oom_kill count: {last.get('cgroup_event_oom_kill', 'unavailable')}",
        f"Process NUMA nodes allowed: {last.get('process_mems_allowed', 'unavailable')}",
    ])


def _current_task_wants_email() -> bool:
    try:
        from src.proj.util.script.autorun import AutoRunTask
        return bool(AutoRunTask._instances and AutoRunTask._instances[-1].email)
    except Exception:
        return False


class ProcessMemoryTracker:
    """Continuously sample process, host, cgroup and PSI memory state."""

    def __init__(
        self,
        label: str,
        *,
        interval: float | None = None,
        stdout_interval: float | None = None,
        output_dir: Path | None = None,
        launch_watcher: bool | None = None,
        register_output: bool = True,
        enabled: bool | None = None,
    ) -> None:
        self.label = label
        self.enabled = _env_flag("LEARNDL_MEMORY_TRACK") if enabled is None else enabled
        self.interval = max(0.2, float(interval or os.getenv("LEARNDL_MEMORY_TRACK_INTERVAL", "5")))
        default_stdout = float(os.getenv("LEARNDL_MEMORY_TRACK_STDOUT_INTERVAL", "30"))
        self.stdout_interval = default_stdout if stdout_interval is None else stdout_interval
        self.output_dir = output_dir
        self.launch_watcher = launch_watcher
        self.register_output = register_output
        self.pid = os.getpid()
        self.path: Path | None = None
        self.done_path: Path | None = None
        self.phase = "initializing"
        self._stop_event = threading.Event()
        self._write_lock = threading.Lock()
        self._file: TextIO | None = None
        self._thread: threading.Thread | None = None
        self._last_stdout = 0.0

    def __enter__(self) -> ProcessMemoryTracker:
        return self.start()

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.stop("error" if exc_type is not None else "complete")

    def _default_output_dir(self) -> Path:
        from src.proj import PATH
        return PATH.runtime.joinpath("memory_telemetry")

    def start(self) -> ProcessMemoryTracker:
        if not self.enabled or self._file is not None:
            return self
        output_dir = self.output_dir or self._default_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.label).strip("_") or "training"
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        self.path = output_dir / f"{stamp}.{safe_label}.pid{self.pid}.memory.jsonl"
        self.done_path = Path(f"{self.path}.done")
        self.done_path.unlink(missing_ok=True)
        self._file = self.path.open("w", encoding="utf-8", buffering=1)
        self._write_record({
            "event": "start",
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "pid": self.pid,
            "label": self.label,
            "interval_seconds": self.interval,
            "python": sys.executable,
            "argv": sys.argv,
        })
        self._sample("sample")
        if self.register_output:
            from src.proj import Proj
            Proj.exit_files.append(self.path)
        print(f"Memory telemetry started: {self.path}", flush=True)
        self._thread = threading.Thread(target=self._run, name="memory-telemetry", daemon=True)
        self._thread.start()
        self._maybe_launch_watcher()
        return self

    def mark(self, phase: str, **details: Any) -> None:
        if not self.enabled or self._file is None:
            return
        self.phase = phase
        self._sample("mark", details)

    def stop(self, outcome: str = "complete") -> None:
        if self._file is None or self.path is None or self.done_path is None:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 1)
        self._sample("final", {"outcome": outcome})
        self._write_record({
            "event": "stop",
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "pid": self.pid,
            "label": self.label,
            "phase": self.phase,
            "outcome": outcome,
        })
        with self._write_lock:
            self._file.close()
            self._file = None
        self.done_path.write_text(json.dumps({"outcome": outcome}), encoding="utf-8")
        print(f"Memory telemetry finished: {self.path}", flush=True)

    def _write_record(self, record: dict[str, Any]) -> None:
        with self._write_lock:
            if self._file is None:
                return
            self._file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            self._file.flush()

    def _sample(self, event: str, details: dict[str, Any] | None = None) -> None:
        sample = collect_memory_sample(self.pid)
        sample["event"] = event
        sample["label"] = self.label
        sample["phase"] = self.phase
        if details:
            sample["details"] = details
        self._write_record(sample)

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval):
            self._sample("sample")
            now = time.monotonic()
            if self.stdout_interval and now - self._last_stdout >= self.stdout_interval:
                sample = collect_memory_sample(self.pid)
                print(
                    "[memory] "
                    f"phase={self.phase} rss={_format_bytes(sample.get('process_vmrss_bytes'))} "
                    f"cgroup={_format_bytes(sample.get('cgroup_memory_current_bytes'))} "
                    f"available={_format_bytes(sample.get('system_memavailable_bytes'))} "
                    f"psi10={sample.get('cgroup_psi_some_avg10', sample.get('system_psi_some_avg10', 'n/a'))}",
                    flush=True,
                )
                self._last_stdout = now

    def _maybe_launch_watcher(self) -> None:
        if self.path is None or self.done_path is None:
            return
        launch = self.launch_watcher
        if launch is None:
            try:
                from src.proj import MACHINE
                launch = bool(sys.platform.startswith("linux") and MACHINE.platform_server and _current_task_wants_email())
            except Exception:
                launch = False
        if not launch or shutil.which("systemd-run") is None:
            return
        _, start_ticks = _process_identity(self.pid)
        token = f"{self.pid}-{int(time.time() * 1000) % 1_000_000_000}"
        unit = f"learndl-memory-watch-{token}"
        args = [
            "systemd-run", "--user", "--quiet", "--collect", f"--unit={unit}",
            f"--working-directory={Path.cwd()}", sys.executable, str(Path(__file__).resolve()),
            "--watch-pid", str(self.pid), "--report", str(self.path),
            "--done", str(self.done_path), "--label", self.label,
        ]
        if start_ticks is not None:
            args.extend(["--start-ticks", start_ticks])
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=10, check=False)
            if result.returncode == 0:
                self._write_record({"event": "watcher_started", "unit": unit})
                print(f"Memory telemetry kill watcher started: {unit}.service", flush=True)
            else:
                self._write_record({
                    "event": "watcher_start_failed", "returncode": result.returncode,
                    "stderr": result.stderr.strip(),
                })
                print(f"Memory telemetry kill watcher failed: {result.stderr.strip()}", flush=True)
        except (OSError, subprocess.SubprocessError) as exc:
            self._write_record({"event": "watcher_start_failed", "error": repr(exc)})
            print(f"Memory telemetry kill watcher failed: {exc!r}", flush=True)


def _watch_process(args: argparse.Namespace) -> int:
    report_path = Path(args.report)
    done_path = Path(args.done)
    while True:
        if done_path.exists():
            return 0
        if not _process_alive(args.watch_pid, args.start_ticks):
            break
        time.sleep(2)

    # Give line-buffered writers a moment to finish their final kernel write.
    time.sleep(1)
    summary = _summarize_report(report_path)
    watcher_event = {
        "event": "abnormal_exit_detected",
        "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "pid": args.watch_pid,
        "label": args.label,
    }
    try:
        with report_path.open("a", encoding="utf-8") as report:
            report.write(json.dumps(watcher_event, separators=(",", ":")) + "\n")
    except OSError:
        pass

    body = "\n".join([
        "The monitored training process ended without completing memory-telemetry cleanup.",
        "This is consistent with SIGKILL/systemd-oomd. The JSONL attachment is intended for post-mortem analysis.",
        "",
        f"Label: {args.label}",
        f"PID: {args.watch_pid}",
        f"Report: {report_path}",
        "",
        summary,
    ])
    try:
        from src.proj.util.web.emailer import Email
        Email.send(
            f"Killed - Memory Telemetry - {args.label}", body,
            attachments=[report_path], confirmation_message="memory telemetry",
        )
    except Exception as exc:
        try:
            with report_path.open("a", encoding="utf-8") as report:
                report.write(json.dumps({"event": "watcher_email_failed", "error": repr(exc)}) + "\n")
        except OSError:
            pass
        return 1
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learndl memory telemetry OOM watcher")
    parser.add_argument("--watch-pid", type=int, required=True)
    parser.add_argument("--start-ticks")
    parser.add_argument("--report", required=True)
    parser.add_argument("--done", required=True)
    parser.add_argument("--label", default="training")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(_watch_process(_parse_args()))
