"""Staged maintenance installation with rollback journal and source checks."""
import hashlib
import json
import os
import shutil
from pathlib import Path
from uuid import uuid4

def digest(path):
    path = Path(path)
    if not path.exists():
        return None
    files = sorted(path.rglob("*")) if path.is_dir() else [path]
    result = hashlib.sha256()
    for file in files:
        if file.is_file():
            result.update(
                str(file.relative_to(path) if path.is_dir() else file.name).encode()
            )
            with file.open("rb") as stream:
                result.update(hashlib.file_digest(stream, "sha256").digest())
    return result.hexdigest()


def copy_path(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)


def remove_path(path):
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def install(replacements, baseline, folder):
    """Persist rollback instructions before changing any output. Requires idle writers."""
    entries = []
    for i, (target, staged) in enumerate(replacements):
        if digest(target) != baseline[target]:
            raise RuntimeError(
                f"Output changed during repair: {target}; rerun while idle"
            )
        backup = folder / "backups" / str(i)
        if target.exists():
            copy_path(target, backup)
        entries.append(
            dict(
                target=str(target),
                backup=str(backup),
                existed=target.exists(),
                staged=str(staged) if staged else None,
            )
        )
    journal = folder / "transaction.json"
    journal.write_text(json.dumps(dict(status="installing", entries=entries), indent=2))
    touched = []
    try:
        for (target, staged), entry in zip(replacements, entries):
            if digest(target) != baseline[target]:
                raise RuntimeError(f"Concurrent output change: {target}")
            temporary = target.with_name(target.name + ".repair-" + uuid4().hex)
            try:
                if staged is not None:
                    copy_path(staged, temporary)
                touched.append(entry)
                if target.is_dir() or staged is None:
                    remove_path(target)
                if staged is not None:
                    os.replace(temporary, target)
            finally:
                remove_path(temporary)
        journal.write_text(
            json.dumps(dict(status="complete", entries=entries), indent=2)
        )
    except BaseException:
        for entry in reversed(touched):
            target = Path(entry["target"])
            remove_path(target)
            if entry["existed"]:
                copy_path(Path(entry["backup"]), target)
        journal.write_text(
            json.dumps(dict(status="rolled_back", entries=entries), indent=2)
        )
        raise
