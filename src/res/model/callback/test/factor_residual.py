"""Daily residual/coverage diagnostics without retaining the full test panel in RAM."""
from __future__ import annotations

import json
import resource
import sys
import time
import numpy as np
import pandas as pd
import torch
from src.res.model.util import BaseCallBack

__all__ = ["FactorResidualDiagnostics"]


class FactorResidualDiagnostics(BaseCallBack):
    def __init__(self, trainer, **kwargs):
        super().__init__(trainer, **kwargs)
        self.started = time.monotonic()

    @property
    def folder(self):
        path = self.config.base_path.rslt("factor_residual")
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
        frame.to_feather(self.folder / f"{batch.date0}.feather")

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
        pd.DataFrame(rows).to_csv(self.folder / "daily.csv", index=False)
        report = dict(
            elapsed_seconds=time.monotonic() - self.started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()
            if torch.cuda.is_available()
            else None,
        )
        (self.folder / "resources.json").write_text(json.dumps(report, indent=2))
        self.logger.stdout(
            f"Factor residual diagnostics: {self.folder}; resources: {report}"
        )
        from src.proj import PATH

        comparisons = []
        for base, candidate in [("snapshot", "temporal"), ("temporal", "contextual")]:
            left = (
                PATH.model_nn
                / f"factorseq_residual@factorseq_v2_{base}"
                / "results"
                / "factor_residual"
            )
            right = (
                PATH.model_nn
                / f"factorseq_residual@factorseq_v2_{candidate}"
                / "results"
                / "factor_residual"
            )
            left_files = {p.name: p for p in left.glob("*.feather")}
            right_files = {p.name: p for p in right.glob("*.feather")}
            if not left_files or left_files.keys() != right_files.keys():
                continue
            for name in sorted(left_files):
                a = pd.read_feather(left_files[name])
                b = pd.read_feather(right_files[name])
                joined = a.merge(
                    b,
                    on="secid",
                    suffixes=("_base", "_candidate"),
                    validate="one_to_one",
                )
                if (
                    len(joined) != len(a)
                    or len(joined) != len(b)
                    or not np.allclose(
                        joined.label_base, joined.label_candidate, equal_nan=True
                    )
                ):
                    raise ValueError("Residual comparison samples/labels differ")
                ia = joined.pred_base.corr(joined.label_base, method="spearman")
                ib = joined.pred_candidate.corr(
                    joined.label_candidate, method="spearman"
                )
                comparisons.append(
                    dict(
                        pair=f"{candidate}-{base}",
                        date=int(name.split(".")[0]),
                        delta_ic=ib - ia,
                        prediction_corr=joined.pred_base.corr(
                            joined.pred_candidate, method="spearman"
                        ),
                    )
                )
        if comparisons:
            daily = pd.DataFrame(comparisons)
            daily.to_csv(self.folder / "paired_daily.csv", index=False)
            annual = (
                daily.assign(year=daily.date // 10000)
                .groupby(["pair", "year"])[["delta_ic", "prediction_corr"]]
                .mean()
            )
            annual.to_csv(self.folder / "paired_annual.csv")
            self.logger.stdout(f"Paired residual comparisons:\n{annual.to_string()}")
