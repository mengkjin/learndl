"""Continue a data-stage failure using verified factor artifacts and private raw inputs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

from .bundle import file_digest, write_json
from .rebuild_data import REPAIR_VERSION

__all__ = ["RecoverySource", "run_recovery"]


class RecoverySource:
    """An explicit artifact adoption, not certification of the old raw database."""

    def __init__(self, parent):
        self.parent = Path(parent)
        self.db = sqlite3.connect(
            (self.parent / "checkpoint.sqlite").as_uri() + "?mode=ro", uri=True
        )
        self.identity = json.loads(
            self.db.execute("select value from identity").fetchone()[0]
        )
        incomplete = self.db.execute(
            "select key from steps where status != 'passed'"
        ).fetchall()
        if (
            self.identity.get("version") != REPAIR_VERSION
            or self.identity.get("mode") != "rebuild"
            or incomplete != [("data",)]
        ):
            self.db.close()
            raise ValueError(
                "Recovery requires a compatible rebuild with only its data step incomplete"
            )

    def adopt(self, key, path, checkpoint):
        """Adopt only an exact, completed daily factor artifact; never old downstream stages."""
        if not key.startswith("factor:"):
            return False
        passed = self.db.execute(
            "select status from steps where key=?", (key,)
        ).fetchone()
        output = self.db.execute(
            "select digest from outputs where path=?", (str(path),)
        ).fetchone()
        if not passed or passed[0] != "passed" or not output or not path.exists():
            return False
        if file_digest(path) != output[0]:
            return False
        # Adoption does not modify the file; no duplicate backup of 100k outputs.
        checkpoint.db.execute(
            "insert or replace into outputs values (?,?)", (str(path), output[0])
        )
        checkpoint.db.execute(
            "insert or replace into steps values (?,?,?)",
            (key, "passed", "adopted unchanged factor artifact"),
        )
        checkpoint.db.commit()
        return True

    def close(self):
        self.db.close()


def run_recovery(
    start=20110101,
    end=None,
    run_id=None,
    dry_run=False,
    batch_size=32,
    source_run="rebuild_20261116T064711Z_a33e87ad",
):
    """Freeze the old cutoff; supplement missing history and rebuild all downstream stages."""
    from src.proj import PATH, Logger
    from src.proj.db.io.snapshot import ReadSnapshot
    from src.proj.util.functional.parallel import serial_execution
    from src.data import DATAVENDOR
    from .repair import run_repair

    run_id = (
        run_id
        or f"recovery_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    )
    if not all(
        re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in (run_id, source_run)
    ):
        raise ValueError("Unsafe run ID")
    if run_id == source_run:
        raise ValueError(
            "Use a new recovery run ID; the original checkpoint is preserved"
        )
    root = PATH.result / "factor_sequence_fixes"
    source = RecoverySource(root / source_run)
    snapshot = None
    try:
        if start != source.identity["start"]:
            raise ValueError("Recovery start must match original rebuild start")
        end = int(end or source.identity["affected_end"])
        if end > source.identity["affected_end"] or end < start:
            raise ValueError(
                "Backfill interval must lie within the original rebuild range"
            )
        folder = root / run_id
        reports = folder / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        provenance = dict(
            source_run=source_run,
            source_identity=source.identity,
            cutoff=source.identity["affected_end"],
            limitation=(
                "Exact factor artifacts adopted from prior rebuild; its raw historical "
                "consistency cannot be reconstructed. Newly read raw inputs use persistent "
                "per-file first-read copies, not a database-wide atomic snapshot."
            ),
        )
        existing = reports / "recovery_source.json"
        if existing.exists() and json.loads(existing.read_text()) != provenance:
            raise ValueError("Recovery source changed; do not mix checkpoints")
        write_json(existing, provenance)
        Logger.stdout(
            f"[recovery] source={source_run}; frozen cutoff={provenance['cutoff']}",
            flush=True,
        )
        if not dry_run:
            if (folder / "checkpoint.sqlite").exists() and not (
                folder / "raw_snapshot/catalog.sqlite"
            ).exists():
                raise ValueError(
                    "Recovery input snapshot is missing; cannot safely resume"
                )
            snapshot = ReadSnapshot(folder / "raw_snapshot", PATH.db)
        from contextlib import nullcontext

        with serial_execution(), snapshot.activate() if snapshot else nullcontext():
            # Do not use tables loaded before entering the snapshot scope.
            for name in (
                "INFO",
                "TRADE",
                "RISK",
                "INDI",
                "BS",
                "IS",
                "CF",
                "FINA",
                "ANALYST",
            ):
                loader = getattr(DATAVENDOR, name)
                loader.clear_all()
            return run_repair(
                "recover",
                start,
                end,
                run_id,
                dry_run,
                batch_size,
                recovery=source,
                snapshot=snapshot,
            )
    finally:
        source.close()
        if snapshot is not None:
            write_json(
                root / run_id / "reports" / "raw_snapshot.json", snapshot.summary()
            )
            snapshot.close()
