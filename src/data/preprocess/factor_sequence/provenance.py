"""Logical component names for reproducible source fingerprints."""
from __future__ import annotations
from pathlib import Path

__all__ = ['COMPONENTS', 'component_paths']

COMPONENTS = {
    '__init__.py': 'src/data/preprocess/factor_sequence/__init__.py',
    'audit_runner.py': 'src/res/model/experiment/factor_sequence/audit_runner.py',
    'bundle.py': 'src/data/preprocess/factor_sequence/bundle.py',
    'context_audit.py': 'src/data/preprocess/factor_sequence/context_audit.py',
    'context_diagnostics.py': 'src/res/model/analytic/factor_sequence/context_diagnostics.py',
    'dividend.py': 'src/data/preprocess/factor_sequence/dividend.py',
    'numeric.py': 'src/data/preprocess/factor_sequence/numeric.py',
    'pipeline.py': 'src/res/model/experiment/factor_sequence/pipeline.py',
    'processors.py': 'src/data/preprocess/factor_sequence/processors.py',
    'rebuild_data.py': 'src/data/preprocess/factor_sequence/repaired_bundle.py',
    'recovery.py': 'src/data/maintenance/factor_sequence/recovery.py',
    'repair.py': 'src/data/maintenance/factor_sequence/repair.py',
    'repaired_schedule.py': 'src/res/model/experiment/factor_sequence/repaired_schedule.py',
    'resume.py': 'src/res/model/experiment/factor_sequence/resume.py',
    'risk_audit.py': 'src/data/preprocess/factor_sequence/risk_audit.py',
    'schedule_training.py': 'src/res/model/experiment/factor_sequence/schedule_training.py',
    'schema.py': 'src/data/preprocess/factor_sequence/schema.py',
    'server_suite.py': 'src/res/model/experiment/factor_sequence/server_suite.py',
    'sources.py': 'src/res/model/experiment/factor_sequence/sources.py',
    'training_mail.py': 'src/res/model/experiment/factor_sequence/training_mail.py',
    'worklist.py': 'src/res/model/experiment/factor_sequence/worklist.py',
}


def component_paths() -> dict[str, Path]:
    root = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file())
    return {name: root / path for name, path in COMPONENTS.items()}
