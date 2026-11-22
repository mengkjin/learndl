"""Pair completed v4 histories with an explicit Temporal model root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from .context_diagnostics import epoch_curves, paired_days, residual_snapshot

__all__ = ["compare_v4", "main"]


def compare_v4(temporal, shared, private, graph, output):
    roots = {
        key: Path(path)
        for key, path in dict(
            temporal=temporal,
            shared=shared,
            private=private,
            graph=graph,
        ).items()
    }
    names = {
        "temporal": "factorseq_v3_temporal",
        **{key: f"factorseq_v4_{key}" for key in ("shared", "private", "graph")},
    }
    configs = {
        key: yaml.safe_load((root / "configs/model.yaml").read_text())
        for key, root in roots.items()
    }
    for key, config in configs.items():
        if config["model.name"] != names[key]:
            raise ValueError(f"{key}: expected schedule {names[key]}")
        for field in (
            "input.factor_sequence.dataset",
            "train.factor_sequence.folds",
            "model.labels",
        ):
            if not config.get(field) or config[field] != configs["temporal"].get(field):
                raise ValueError(f"{key}: missing or different {field}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for key in ("shared", "private", "graph"):
        paired = paired_days(
            residual_snapshot(roots["temporal"]), residual_snapshot(roots[key])
        )
        paired.to_csv(output / f"{key}_daily.csv", index=False)
        paired.assign(year=paired.date // 10000).groupby(["year", "group"]).mean(
            numeric_only=True,
        ).to_csv(output / f"{key}_annual.csv")
    for key, root in roots.items():
        report = output / key
        report.mkdir()
        fit = sorted((root / "results/factor_context_v4/fit").glob("*.json"))
        if fit:
            pd.DataFrame([json.loads(path.read_text()) for path in fit]).to_csv(
                report / "selected_epochs.csv",
                index=False,
            )
        # The baseline predates the selected-epoch summary. Its existing workbook
        # diagnostic explicitly identifies maxima across attempts as such.
        epoch_curves(root, report)
    (output / "model_roots.json").write_text(
        json.dumps({key: str(root.resolve()) for key, root in roots.items()}, indent=2)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("temporal", "shared", "private", "graph", "output"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    compare_v4(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
