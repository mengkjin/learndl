"""Explicit publication recovery for completed factor-sequence experiments."""
from __future__ import annotations
import json
from pathlib import Path

__all__ = ['recover_reports']

def recover_reports(roots, run_id):
    """Retry only these experiments; preserve outbox identities and server originals."""
    from src.automation.artifact_exchange.core import (
        Config,
        ArtifactSpec,
        locked,
        _send,
        sandboxed,
    )
    from src.automation.artifact_exchange.research import publish_safe
    import portalocker

    c = Config.load()
    if sandboxed() or not c.enabled or c.role != "publisher":
        return [
            dict(status="skipped", reason="not an enabled publisher or autofix context")
        ]
    output = []
    for root in map(Path, roots.values()):
        name = "nn@" + root.name
        matches = []
        for ledger in (c.state / "outbox").glob("*/delivery.json"):
            record = json.loads(ledger.read_text())
            if record.get("manifest", {}).get("experiment") == name:
                matches.append((ledger, record))
        if matches:
            for ledger, record in matches:
                if record.get("status") == "published":
                    output.append(
                        dict(
                            status="published",
                            id=record["id"],
                            experiment=name,
                            reason="existing receipt; no duplicate publication",
                        )
                    )
                    continue
                try:
                    with locked(c, "publish"):
                        record = json.loads(ledger.read_text())
                        output.append(_send(record, ledger, c))
                except Exception as exc:
                    output.append(
                        dict(status="failed", experiment=name, error=str(exc))
                    )
        elif (root / "results").is_dir():
            try:
                with portalocker.Lock(str(root / ".lifecycle.lock"), timeout=0):
                    if not any(p.is_file() for p in (root / "results").rglob("*")):
                        raise ValueError("No completed report files")
                    output.extend(
                        publish_safe(
                            "model_report_recovery",
                            run_id,
                            [
                                ArtifactSpec(
                                    root / "results",
                                    category="model_training",
                                    experiment=name,
                                    run_id=run_id,
                                )
                            ],
                            "historical_reports",
                        )
                    )
            except Exception as exc:
                output.append(dict(status="failed", experiment=name, error=str(exc)))
        else:
            output.append(
                dict(
                    status="unregistered",
                    experiment=name,
                    reason="exact model results absent",
                )
            )
    return output
