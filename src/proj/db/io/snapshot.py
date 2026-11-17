"""Opt-in, persistent copy-on-read snapshots for long-running serial research jobs."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import sqlite3
from uuid import uuid4

__all__ = ["ReadSnapshot", "snapshot_read_path", "snapshot_active"]

_active: ContextVar[ReadSnapshot | None] = ContextVar("db_read_snapshot", default=None)


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot_active():
    """Whether IO accelerators must remain in this snapshot's calling context."""
    return _active.get() is not None


def snapshot_read_path(path):
    """Return an immutable private copy only inside an explicit snapshot scope."""
    active = _active.get()
    return active.resolve(path) if active is not None else path


class ReadSnapshot:
    """Freeze each source at first read, not an atomic database-wide point in time.

    Source files are copied (never hardlinked). The persisted catalog is reused on
    restart. Output/export paths outside ``source_root`` retain normal semantics.
    Use serial_execution alongside this scope so worker processes cannot bypass it.
    """

    def __init__(self, folder, source_root):
        self.folder = Path(folder)
        self.source_root = Path(source_root).resolve()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.folder / "catalog.sqlite")
        self.db.execute(
            "create table if not exists files "
            "(source text primary key, target text, digest text, captured text)"
        )
        self.db.execute("create table if not exists metadata (root text)")
        prior = self.db.execute("select root from metadata").fetchone()
        if prior and prior[0] != str(self.source_root):
            self.db.close()
            raise ValueError("Snapshot source root changed")
        if not prior:
            self.db.execute("insert into metadata values (?)", (str(self.source_root),))
        self.db.commit()
        self.checked = set()

    def resolve(self, path):
        if not isinstance(path, (str, Path)):
            return path
        source = Path(path).resolve()
        if not source.is_relative_to(self.source_root):
            return path
        row = self.db.execute(
            "select target,digest from files where source=?", (str(source),)
        ).fetchone()
        if row:
            target, digest = Path(row[0]), row[1]
            if str(source) not in self.checked:
                if digest is None:
                    if target.exists():
                        raise ValueError(
                            f"Frozen missing source unexpectedly exists: {target}"
                        )
                elif not target.exists() or _digest(target) != digest:
                    raise ValueError(f"Corrupt private source snapshot: {target}")
                self.checked.add(str(source))
            return target
        target = self.folder / "files" / source.relative_to(self.source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = None
        if source.exists():
            temporary = target.with_name(target.name + "." + uuid4().hex + ".tmp")
            try:
                for _ in range(3):
                    before = source.stat()
                    shutil.copyfile(source, temporary)
                    after = source.stat()
                    if (before.st_ino, before.st_size, before.st_mtime_ns) == (
                        after.st_ino,
                        after.st_size,
                        after.st_mtime_ns,
                    ):
                        digest = _digest(temporary)
                        temporary.replace(target)
                        break
                else:
                    raise ValueError(
                        f"Source changed repeatedly while snapshotting: {source}"
                    )
            finally:
                temporary.unlink(missing_ok=True)
        else:
            # Remove a private orphan left between copy and catalog commit.
            target.unlink(missing_ok=True)
        self.db.execute(
            "insert into files values (?,?,?,?)",
            (str(source), str(target), digest, datetime.now(timezone.utc).isoformat()),
        )
        self.db.commit()
        self.checked.add(str(source))
        return target

    def summary(self):
        digest = hashlib.sha256()
        count = missing = 0
        for source, sha in self.db.execute(
            "select source,digest from files order by source"
        ):
            digest.update(f"{source}:{sha}\n".encode())
            count += 1
            missing += sha is None
        return dict(
            policy="persistent-first-read-copy-v1",
            files=count,
            missing=missing,
            sha256=digest.hexdigest(),
            catalog=str(self.folder / "catalog.sqlite"),
            limitation="Per-file first-read snapshot; not an atomic historical database snapshot",
        )

    @contextmanager
    def activate(self):
        if _active.get() is not None:
            raise RuntimeError("Nested read snapshots are not supported")
        token = _active.set(self)
        try:
            yield self
        finally:
            _active.reset(token)

    def close(self):
        self.db.close()
