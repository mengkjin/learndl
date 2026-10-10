"""Portable result bundles, safe local extraction and resumable email delivery."""
from __future__ import annotations

import hashlib
import json
import shutil
import stat
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_files(run: Path) -> list[Path]:
    result = []
    for path in sorted(run.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"result bundle refuses symlink: {path}")
        if path.is_file():
            if path.name == "panel.npz":
                continue
            result.append(path)
    return result


def create_bundle(run_dir: str | Path, bundle_dir: str | Path, max_part_bytes: int = 10 * 1024 * 1024) -> Path:
    run = Path(run_dir).resolve()
    target = Path(bundle_dir).resolve()
    if max_part_bytes <= 0:
        raise ValueError("max_part_bytes must be positive")
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = run / "package_manifest.json"
    files = [path for path in _run_files(run) if path != manifest_path]
    manifest = {
        "schema_version": 1,
        "run_id": run.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contains_panel": False,
        "files": [
            {"path": path.relative_to(run).as_posix(), "size": path.stat().st_size, "sha256": _sha256(path)}
            for path in files
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    archive = target / f"{run.name}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in _run_files(run):
            bundle.write(path, path.relative_to(run).as_posix())
    parts = []
    if archive.stat().st_size <= max_part_bytes:
        parts = [archive]
    else:
        with archive.open("rb") as source:
            index = 1
            while chunk := source.read(max_part_bytes):
                part = target / f"{archive.name}.part{index:03d}"
                part.write_bytes(chunk)
                parts.append(part)
                index += 1
        archive.unlink()
    archive_digest = hashlib.sha256()
    for part in parts:
        with part.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                archive_digest.update(chunk)
    parts_manifest = {
        "schema_version": 1,
        "run_id": run.name,
        "archive_name": archive.name,
        "archive_sha256": archive_digest.hexdigest(),
        "parts": [
            {"name": path.name, "size": path.stat().st_size, "sha256": _sha256(path)} for path in parts
        ],
    }
    metadata = target / f"{run.name}.parts.json"
    metadata.write_text(json.dumps(parts_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def _parts(metadata: Path) -> tuple[dict[str, Any], list[Path]]:
    definition = json.loads(metadata.read_text(encoding="utf-8"))
    parts = [metadata.parent / item["name"] for item in definition["parts"]]
    for item, path in zip(definition["parts"], parts, strict=True):
        if not path.is_file() or path.stat().st_size != item["size"] or _sha256(path) != item["sha256"]:
            raise ValueError(f"bundle part failed size/hash verification: {path}")
    return definition, parts


def _resolve_metadata(source: Path) -> Path:
    if source.name.endswith(".parts.json"):
        return source
    if source.suffix == ".zip" or ".zip.part" in source.name:
        run_id = source.name.split(".zip", 1)[0]
        candidate = source.parent / f"{run_id}.parts.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("pass the .parts.json file, or keep it beside the ZIP/part")


def analyze_bundle(source: str | Path, output_dir: str | Path) -> Path:
    metadata = _resolve_metadata(Path(source).resolve())
    definition, parts = _parts(metadata)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / definition["archive_name"]
    if len(parts) == 1 and parts[0].suffix == ".zip":
        shutil.copy2(parts[0], archive)
    else:
        with archive.open("wb") as target:
            for part in parts:
                with part.open("rb") as handle:
                    shutil.copyfileobj(handle, target)
    if _sha256(archive) != definition["archive_sha256"]:
        raise ValueError("reassembled archive SHA-256 does not match")
    extract_root = destination / definition["run_id"]
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            relative = PurePosixPath(info.filename)
            mode = info.external_attr >> 16
            if relative.is_absolute() or ".." in relative.parts or stat.S_ISLNK(mode):
                raise ValueError(f"unsafe bundle member: {info.filename}")
            target = (extract_root / Path(*relative.parts)).resolve()
            if extract_root not in target.parents and target != extract_root:
                raise ValueError(f"bundle member escapes output directory: {info.filename}")
        bundle.extractall(extract_root)
    package = json.loads((extract_root / "package_manifest.json").read_text(encoding="utf-8"))
    if package.get("contains_panel") is not False:
        raise ValueError("bundle incorrectly claims to contain a panel")
    for item in package["files"]:
        path = extract_root / item["path"]
        if not path.is_file() or path.stat().st_size != item["size"] or _sha256(path) != item["sha256"]:
            raise ValueError(f"extracted file failed verification: {item['path']}")
    status = json.loads((extract_root / "status.json").read_text(encoding="utf-8"))
    analysis = {
        "run_id": definition["run_id"],
        "status": status["status"],
        "report": str(extract_root / "report.html") if (extract_root / "report.html").is_file() else None,
        "tensorboard": str(extract_root / "tensorboard"),
        "metrics": str(extract_root / "metrics.json") if (extract_root / "metrics.json").is_file() else None,
        "checkpoint_replay_requires_panel": True,
    }
    (extract_root / "local_analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    if status["status"] != "success":
        raise RuntimeError(f"bundle contains a failed run; inspect {extract_root}")
    return extract_root


def bundle_attachments(metadata_path: str | Path) -> list[Path]:
    """Verified archive parts and their manifest for a host task's email."""
    metadata = Path(metadata_path).resolve()
    _, parts = _parts(metadata)
    return [*parts, metadata]


def send_bundle(
    metadata_path: str | Path,
    recipient: str | None = None,
    *,
    only_failed: bool = False,
    transport: Callable[[str, str, str | None, list[Path]], bool] | None = None,
) -> dict[str, Any]:
    metadata = Path(metadata_path).resolve()
    definition, parts = _parts(metadata)
    delivery_path = metadata.with_name(f"{definition['run_id']}.delivery.json")
    previous = json.loads(delivery_path.read_text(encoding="utf-8")) if delivery_path.is_file() else {"parts": {}}

    if transport is None:
        def transport(title: str, body: str, address: str | None, attachments: list[Path]) -> bool:
            from src.proj.util.web.emailer import Email

            message = Email.message(title, body, address, attachments=attachments)
            return Email.send_with_smtplib(message, address, confirmation_message="RL experiment bundle")

    states = dict(previous.get("parts", {}))
    total = len(parts)
    for index, part in enumerate(parts, 1):
        if only_failed and states.get(part.name, {}).get("sent") is True:
            continue
        title = f"RL experiment {definition['run_id']} ({index}/{total})"
        body = (
            f"Run: {definition['run_id']}\nAttachment: {part.name}\n"
            "Keep the .parts.json file beside all parts, then run analyze-bundle locally."
        )
        sent = bool(transport(title, body, recipient, [part, metadata]))
        states[part.name] = {"sent": sent, "attempted_at": datetime.now(timezone.utc).isoformat()}
        delivery = {"run_id": definition["run_id"], "recipient": recipient, "parts": states}
        delivery_path.write_text(json.dumps(delivery, ensure_ascii=False, indent=2), encoding="utf-8")
    return json.loads(delivery_path.read_text(encoding="utf-8"))
