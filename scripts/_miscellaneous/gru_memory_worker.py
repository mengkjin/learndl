"""PyTorch worker for the isolated synthetic GRU memory benchmark."""
from __future__ import annotations

import os
import time
import traceback
from typing import Any

import torch
import torch.version as torch_version
from torch import nn
from torch.utils.checkpoint import checkpoint

GIB = 1024**3


class SyntheticGRU(nn.Module):
    """Two GRUs and the default gru_gru head, with random weights and MSE loss.

Keep this independent of project imports so data/config initialization cannot
contaminate GPU measurements. BatchNorm and loss always see all stocks.
"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        h, day_h = config["hidden"], config["day_hidden"]
        kwargs = dict(num_layers=config["layers"], batch_first=True,
                      dropout=config["dropout"] if config["layers"] > 1 else 0.)
        self.intraday = nn.GRU(config["features"], h, **kwargs)
        self.days = nn.GRU(h, day_h, **kwargs)
        self.head = nn.Sequential(nn.Linear(day_h, day_h), nn.BatchNorm1d(day_h),
                                  nn.Linear(day_h, 1), nn.BatchNorm1d(1))

    def encode(self, x):
        last = self.intraday(x)[0][:, -1]
        # Keep the returned checkpoint boundary compact rather than retaining
        # full sequence storage through a view. This does not detach gradients.
        return last.clone() if self.config["compact_output"] else last

    def encode_chunks(self, flat):
        outputs = []
        for part in flat.split(self.config["chunk_size"], dim=0):
            if self.config["mode"] == "chunk_checkpoint":
                outputs.append(checkpoint(self.encode, part, use_reentrant=False,
                                          preserve_rng_state=True))
            else:
                outputs.append(self.encode(part))
        return torch.cat(outputs, dim=0)

    def encode_days(self, encoded):
        return self.days(encoded)[0][:, -1]

    def forward(self, x):
        stocks, days, bars, features = x.shape
        flat = x.reshape(-1, bars, features)
        mode = self.config["mode"]
        if mode == "full":
            encoded = self.encode(flat)
        elif mode == "full_checkpoint":
            encoded = checkpoint(self.encode, flat, use_reentrant=False, preserve_rng_state=True)
        elif mode == "chunk_outer_checkpoint":
            encoded = checkpoint(self.encode_chunks, flat, use_reentrant=False, preserve_rng_state=True)
        else:
            encoded = self.encode_chunks(flat)
        assert isinstance(encoded, torch.Tensor)
        encoded = encoded.reshape(stocks, days, -1)
        # Fixed across modes to isolate the intraday strategy.
        if self.config["day_checkpoint"]:
            hidden = checkpoint(self.encode_days, encoded, use_reentrant=False, preserve_rng_state=True)
        else:
            hidden = self.encode_days(encoded)
        return self.head(hidden)


def memory(device):
    if device.type != "cuda":
        return {}
    free, total = torch.cuda.mem_get_info(device)
    return dict(allocated_gib=torch.cuda.memory_allocated(device) / GIB,
                reserved_gib=torch.cuda.memory_reserved(device) / GIB,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device) / GIB,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device) / GIB,
                device_free_gib=free / GIB, device_total_gib=total / GIB)


def run_case(config, device_name="cuda:0"):
    """One process per case; include cold step and initialized Adam steps."""
    result: dict[str, Any] = dict(config=config, status="running", stage="setup", steps=[])
    device = torch.device(device_name)
    try:
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; no CPU fallback for memory measurements")
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
            result["gpu"] = torch.cuda.get_device_name(device)
        torch.manual_seed(config["seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        result["environment"] = dict(torch=torch.__version__, cuda=torch_version.cuda,
                                     cudnn=torch.backends.cudnn.version(),
                                     allocator=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                                     tf32=False, cudnn_benchmark=False)
        result["initial_memory"] = memory(device)
        # Deliberately retained until the entire case finishes.
        resident = torch.empty(int(config["resident_gib"] * GIB), dtype=torch.uint8, device=device)
        resident.zero_()
        model = SyntheticGRU(config).to(device).train()
        x = torch.randn(config["stocks"], config["days"], config["bars"],
                        config["features"], device=device)
        target = torch.randn(config["stocks"], 1, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        result["setup_memory"] = memory(device)
        for index in range(config["steps"]):
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            step: dict[str, Any] = dict(index=index, phases={}, baseline=memory(device))
            result["steps"].append(step)
            total_start = time.perf_counter()
            for phase in ("forward", "backward", "optimizer"):
                result["stage"] = phase
                start = time.perf_counter()
                if phase == "forward":
                    prediction = model(x)
                    loss = (prediction - target).square().mean()
                elif phase == "backward":
                    loss.backward()
                else:
                    optimizer.step()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                step["phases"][phase] = dict(seconds=time.perf_counter() - start, **memory(device))
            step["seconds"] = time.perf_counter() - total_start
            step["loss"] = loss.item()
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite synthetic loss")
            del prediction, loss
        result.update(status="ok", stage="complete")
    except torch.cuda.OutOfMemoryError as error:
        result.update(status="oom", error=str(error))
    except Exception as error:
        result.update(status="error", error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc())
    finally:
        # This is achieved allocation, NOT the amount needed to finish an OOM case.
        try:
            result["final_memory"] = memory(device) if device.type != "cuda" or torch.cuda.is_initialized() else {}
        except Exception:
            result["final_memory"] = {}
    return result
