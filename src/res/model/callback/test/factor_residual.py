"""Daily residual/coverage diagnostics without retaining the full test panel in RAM."""

from __future__ import annotations

import json
import resource
import sys
import time
import pandas as pd
import torch
from src.res.model.util import BaseCallBack

__all__ = ["FactorResidualDiagnostics"]


class FactorResidualDiagnostics(BaseCallBack):
    def __init__(self, trainer, comparison_roots=None, **kwargs):
        super().__init__(trainer, **kwargs)
        self.started = time.monotonic()
        self.comparison_roots = comparison_roots

    @property
    def context_report(self):
        path = self.config.base_path.rslt("factor_context_v4")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def on_fit_model_end(self):
        if self.config.model_module != "factorseq_contextual_v4":
            return
        metrics = self.metrics.model_metrics
        best = metrics.best_attempt_metrics()
        row = dict(
            model_date=int(self.model_date),
            model_num=int(self.model_num),
            best_attempt=metrics.best_attempt(),
            best_epoch=int(best.epoch),
            phase=int(best.phase),
            train_ic=float(best.train_rankic),
            valid_ic=float(best.valid_rankic),
            train_valid_gap=float(best.train_rankic - best.valid_rankic),
        )
        folder = self.context_report / "fit"
        folder.mkdir(exist_ok=True)
        (folder / f"{self.model_date}.{self.model_num}.json").write_text(
            json.dumps(row, indent=2)
        )

    @staticmethod
    def context_summary(date, other):
        ready = other["performance_ready"]
        row: dict[str, int | float | None] = dict(
            date=int(date), ready_factors=int(ready.sum().item())
        )
        for key in (
            "film_gamma",
            "film_beta",
            "context_correction",
            "private_delta",
            "identity_delta",
            "graph_delta",
        ):
            if key in other:
                value = other[key].detach().float()
                if key != "context_correction":
                    value = value[ready]
                row[key + "_rms"] = (
                    float(value.square().mean().sqrt().item()) if value.numel() else 0.0
                )
                row[key + "_max_abs"] = (
                    float(value.abs().max().item()) if value.numel() else 0.0
                )
        if "factor_attention" in other:
            weights = other["factor_attention"].detach().float()
            rows = weights[weights.sum(-1) > 0]
            row["attention_rows"] = len(rows)
            row["attention_entropy"] = (
                float(-(rows * rows.clamp_min(1e-12).log()).sum(-1).mean().item())
                if len(rows)
                else None
            )
        return row

    @property
    def folder(self):
        path = self.config.base_path.snapshot("factor_residual")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def on_test_batch_end(self):
        batch = self.batch_input
        out = self.batch_output
        if "linear_score" not in out.other:
            return
        x = batch.x[0] if isinstance(batch.x, (tuple, list)) else batch.x
        x = x.squeeze(2) if x.ndim == 4 else x
        k = x.shape[-1] // 4
        frame = pd.DataFrame(
            dict(
                secid=batch.secid,
                label=batch.label0,
                pred=out.pred.detach().cpu().numpy().reshape(-1),
                linear=out.other["linear_score"].detach().cpu().numpy().reshape(-1),
                residual=out.other["residual_score"].detach().cpu().numpy().reshape(-1),
                coverage=x[:, -1, k : 2 * k].float().mean(-1).detach().cpu().numpy(),
            )
        )
        if "film_gamma" in out.other:
            for key in ("temporal_score", "context_correction"):
                frame[key] = out.other[key].detach().cpu().numpy().reshape(-1)
        frame.to_feather(self.folder / f"{batch.date0}.feather")
        if "film_gamma" in out.other:
            folder = self.folder / "context"
            folder.mkdir(exist_ok=True)
            row = self.context_summary(batch.date0, out.other)
            (folder / f"{batch.date0}.json").write_text(json.dumps(row))

    def on_test_end(self):
        rows = []
        for path in sorted(self.folder.glob("*.feather")):
            frame = pd.read_feather(path)
            for group, part in [
                ("all", frame),
                ("low", frame[frame.coverage < 0.5]),
                ("medium", frame[(frame.coverage >= 0.5) & (frame.coverage < 0.9)]),
                ("high", frame[frame.coverage >= 0.9]),
            ]:
                if len(part) < 2:
                    continue
                rows.append(
                    dict(
                        date=int(path.stem),
                        group=group,
                        n=len(part),
                        rank_ic=part.pred.corr(part.label, method="spearman"),
                        linear_std=part.linear.std(),
                        residual_std=part.residual.std(),
                        branch_corr=part.linear.corr(part.residual),
                    )
                )
        daily = pd.DataFrame(rows)
        daily.to_csv(self.folder / "daily.csv", index=False)
        report = dict(
            elapsed_seconds=time.monotonic() - self.started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()
            if torch.cuda.is_available()
            else None,
        )
        (self.folder / "resources.json").write_text(json.dumps(report, indent=2))
        context_rows = [
            json.loads(path.read_text())
            for path in (self.folder / "context").glob("*.json")
            if (self.folder / f"{path.stem}.feather").is_file()
        ]
        if context_rows:
            pd.DataFrame(context_rows).sort_values("date").to_csv(
                self.context_report / "daily_context.csv", index=False
            )
            daily.to_csv(self.context_report / "daily_coverage.csv", index=False)
            (self.context_report / "resources.json").write_text(
                json.dumps(report, indent=2)
            )
        self.logger.stdout(
            f"Factor residual diagnostics: {self.folder}; resources: {report}"
        )
        # Do not silently resolve an unnumbered sibling when this run is @2.
        # Historical pairing belongs to the explicit-root audit, not root guessing.
        if self.comparison_roots:
            from src.res.model.analytic.factor_sequence.context_diagnostics import (
                diagnose_roots,
            )

            diagnose_roots(self.comparison_roots, self.folder / "comparison")
        else:
            self.logger.stdout(
                "Paired comparison not run: explicit comparison_roots required; use factor_sequence_coverage audit"
            )
