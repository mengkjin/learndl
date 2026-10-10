from __future__ import annotations

import csv
import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import gymnasium
import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure as TensorBoardFigure
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from .baselines import replay_baseline
from .data import PanelData, Standardizer, date_split
from .diagnostics import compare_metrics, comparison_figure
from .env import EpisodeRange, PortfolioEnv
from .policy import MaskedStockActorCriticPolicy
from .portfolio import PortfolioConstraints
from .reward import RewardConfig


@dataclass(frozen=True)
class ExperimentConfig:
    candidate_count: int = 100
    max_holdings: int = 50
    max_weight: float = 0.03
    fee_rate: float = 0.00035
    selection_scale: float = 1.0
    reward_scale: float = 100.0
    reward: RewardConfig | dict[str, Any] = field(default_factory=RewardConfig)
    selection_metric: str = "nav"
    n_envs: int = 4
    rollout_steps: int = 128
    batch_size: int = 128
    update_epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    hidden_dim: int = 64
    total_timesteps: int = 4_096
    eval_freq: int = 0  # 0: every completed PPO update; positive: environment-step interval
    seed: int = 7
    device: str = "cpu"
    train_end_date: int | None = None
    valid_end_date: int | None = None

    def constraints(self) -> PortfolioConstraints:
        return PortfolioConstraints(
            candidate_count=self.candidate_count,
            max_holdings=self.max_holdings,
            max_weight=self.max_weight,
            fee_rate=self.fee_rate,
            selection_scale=self.selection_scale,
        )


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    keys = list(dict.fromkeys(key for row in history for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow({k: json.dumps(v, default=_json_default) if isinstance(v, (list, np.ndarray)) else v for k, v in row.items()})


def history_metrics(history: list[dict[str, Any]], gamma: float | None = None) -> dict[str, float]:
    if not history:
        return {"total_return": 0.0, "annualized_volatility": 0.0, "max_drawdown": 0.0, "mean_turnover": 0.0}
    nav = np.asarray([float(row["nav"]) for row in history])
    returns = np.asarray([float(row["period_return"]) for row in history])
    peak = np.maximum.accumulate(np.concatenate([[1.0], nav]))
    drawdown = np.concatenate([[1.0], nav]) / peak - 1.0
    metrics = {
        "total_return": float(nav[-1] - 1.0),
        "annualized_volatility": float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else 0.0,
        "max_drawdown": float(drawdown.min()),
        "mean_turnover": float(np.mean([row["turnover"] for row in history])),
        "final_nav": float(nav[-1]),
        "total_fees": float(np.sum([row.get("fee_estimate", 0.0) for row in history])),
    }
    if "frozen_holdings" in history[0]:
        metrics["mean_frozen_holdings"] = float(np.mean([row["frozen_holdings"] for row in history]))
        metrics["rejected_orders"] = float(
            np.sum([row["rejected_buys"] + row["rejected_sells"] for row in history])
        )
    if gamma is not None:
        metrics["discounted_reward"] = float(sum((gamma ** index) * row["reward"] for index, row in enumerate(history)))
    return metrics


def evaluate_model(model: BaseAlgorithm, env: PortfolioEnv) -> tuple[dict[str, float], list[dict[str, Any]]]:
    obs, _ = env.reset()
    terminated = False
    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if truncated:
            raise RuntimeError("unexpected truncation in deterministic evaluation")
    return history_metrics(env.history), list(env.history)


def evaluate_alpha_baseline(env: PortfolioEnv) -> tuple[dict[str, float], list[dict[str, Any]]]:
    obs, _ = env.reset()
    terminated = False
    while not terminated:
        # Zero preferences retain alpha ranking and give equal softplus allocation scores.
        action_shape = env.action_space.shape
        if action_shape is None:
            raise RuntimeError("portfolio action space must declare a shape")
        action = np.zeros(action_shape, dtype=np.float32)
        obs, _, terminated, truncated, _ = env.step(action)
        if truncated:
            raise RuntimeError("unexpected truncation in baseline evaluation")
    return history_metrics(env.history), list(env.history)


class ValidationCallback(BaseCallback):
    """Flush metrics after train(), never from inside a partially collected rollout."""

    ROLLOUT_FIELDS = (
        "reward", "turnover", "fee_estimate", "cash_weight", "holdings",
        "frozen_holdings", "rejected_buys", "rejected_sells",
    )

    def __init__(
        self, eval_env: PortfolioEnv, output_dir: Path, eval_freq: int,
        equal_metrics: dict[str, float], gamma: float, selection_metric: str, verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        if eval_freq < 0:
            raise ValueError("eval_freq must be nonnegative")
        self.eval_env = eval_env
        self.output_dir = output_dir
        self.eval_freq = int(eval_freq)
        self.equal_metrics = equal_metrics
        self.gamma = float(gamma)
        self.selection_metric = selection_metric
        self.best_score = -np.inf
        self.best_metrics: dict[str, float] | None = None
        self.best_history: list[dict[str, Any]] = []
        self.best_update = 0
        self.update_index = 0
        self.updates: list[dict[str, Any]] = []
        self.validations: list[dict[str, Any]] = []
        self.next_evaluation = self.eval_freq
        self.rollout_seconds = 0.0
        self.update_seconds = 0.0
        self.validation_seconds = 0.0
        self._rollout_started = 0.0
        self._rollout_ended: float | None = None
        self._last_rollout_seconds = 0.0
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def _validate(self, *, eligible_for_selection: bool) -> dict[str, float]:
        started = time.perf_counter()
        metrics, history = evaluate_model(self.model, self.eval_env)
        metrics["discounted_reward"] = history_metrics(history, self.gamma)["discounted_reward"]
        seconds = time.perf_counter() - started
        self.validation_seconds += seconds
        values = {f"validation/ppo/{key}": value for key, value in metrics.items()}
        values.update({f"validation/equal_weight/{key}": value for key, value in self.equal_metrics.items()})
        values.update({
            f"validation/comparison/{key}": value
            for key, value in compare_metrics(metrics, self.equal_metrics).items()
        })
        score = metrics["final_nav"] if self.selection_metric == "nav" else metrics["discounted_reward"]
        values["validation/ppo/selection_score"] = score
        if eligible_for_selection and score > self.best_score:
            self.best_score = score
            self.best_metrics, self.best_history = metrics, history
            self.best_update = self.update_index
            self.model.save(self.output_dir / "best_model")
        values["time/validation_seconds"] = seconds
        for key, value in values.items():
            self.logger.record(key, value)
        self.validations.append({
            "update": self.update_index, "environment_steps": self.num_timesteps, **values,
        })
        write_history(self.output_dir / "validation_updates.csv", self.validations)
        return values

    def _on_training_start(self) -> None:
        self._validate(eligible_for_selection=False)
        self.logger.record("train/update_index", 0)
        self.logger.dump(step=0)

    def _finish_update(self, *, final: bool = False) -> None:
        if self._rollout_ended is None:
            return
        # SB3 called train() after on_rollout_end(). Its scalar metrics are still
        # buffered because learn(log_interval=None) disables the default pre-update dump.
        seconds = time.perf_counter() - self._rollout_ended
        self.update_seconds += seconds
        self._rollout_ended = None
        self.update_index += 1
        values = {
            key: float(value) for key, value in self.logger.name_to_value.items()
            if key.startswith("train/") and isinstance(value, (int, float, np.number))
        }
        # SB3 n_updates counts epochs, not PPO rounds; expose our own round index.
        values.pop("train/n_updates", None)
        values["train/update_index"] = self.update_index
        values["time/rollout_seconds"] = self._last_rollout_seconds
        values["time/update_seconds"] = seconds
        values.update({f"rollout/{key}_mean": value / self._counts[key] for key, value in self._sums.items()})
        evaluated = final or self.eval_freq == 0 or self.num_timesteps >= self.next_evaluation
        if evaluated:
            values.update(self._validate(eligible_for_selection=True))
            if self.eval_freq:
                self.next_evaluation = (self.num_timesteps // self.eval_freq + 1) * self.eval_freq
        else:
            values["time/validation_seconds"] = 0.0
        model = self.model
        if not isinstance(model, OnPolicyAlgorithm):
            raise TypeError("validation callback requires an on-policy algorithm")
        values["time/fps"] = model.n_steps * model.n_envs / max(
            self._last_rollout_seconds + seconds + values["time/validation_seconds"], 1e-9,
        )
        for key, value in values.items():
            self.logger.record(key, value)
        self.updates.append({
            "update": self.update_index, "environment_steps": self.num_timesteps,
            "validated": evaluated, **values,
        })
        write_history(self.output_dir / "update_history.csv", self.updates)
        self.logger.dump(step=self.num_timesteps)

    def _on_rollout_start(self) -> None:
        self._finish_update()
        self._sums, self._counts = {}, {}
        self._rollout_started = time.perf_counter()

    def _on_rollout_end(self) -> None:
        now = time.perf_counter()
        self._last_rollout_seconds = now - self._rollout_started
        self.rollout_seconds += self._last_rollout_seconds
        self._rollout_ended = now

    def _on_training_end(self) -> None:
        self._finish_update(final=True)

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            fields = set(self.ROLLOUT_FIELDS) | {key for key in info if key.startswith("reward_component/")}
            for key in sorted(fields):
                if key in info:
                    self._sums[key] = self._sums.get(key, 0.0) + float(info[key])
                    self._counts[key] = self._counts.get(key, 0) + 1
        return True


def _env(panel: PanelData, interval: tuple[int, int], config: ExperimentConfig, *, random_start: bool) -> PortfolioEnv:
    return PortfolioEnv(
        panel,
        EpisodeRange(*interval),
        config.constraints(),
        reward_scale=config.reward_scale,
        reward_config=RewardConfig.from_value(config.reward),
        random_start=random_start,
        minimum_episode_steps=min(config.rollout_steps, interval[1] - interval[0]),
    )


def train_experiment(panel: PanelData, output_dir: str | Path, config: ExperimentConfig) -> dict[str, Any]:
    if config.total_timesteps <= 0 or config.eval_freq < 0:
        raise ValueError("total_timesteps must be positive and eval_freq nonnegative")
    if config.selection_metric not in {"nav", "discounted_reward"}:
        raise ValueError("selection_metric must be nav or discounted_reward")
    RewardConfig.from_value(config.reward).validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits = date_split(panel, config.train_end_date, config.valid_end_date)
    standardizer = Standardizer.fit(panel, *splits["train"])
    normalized = standardizer.transform(panel)
    standardizer.save_npz(output / "normalizer.npz")

    def _train_env(offset: int) -> PortfolioEnv:
        return _seeded_env(normalized, splits["train"], config, offset)

    def _make_train_env(offset: int):
        def _fn() -> PortfolioEnv:
            return _train_env(offset)
        return _fn

    train_env = DummyVecEnv([_make_train_env(i) for i in range(config.n_envs)])
    valid_env = _env(normalized, splits["valid"], config, random_start=False)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    model = PPO(
        cast(type[ActorCriticPolicy], MaskedStockActorCriticPolicy),
        train_env,
        learning_rate=config.learning_rate,
        n_steps=config.rollout_steps,
        batch_size=config.batch_size,
        n_epochs=config.update_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        policy_kwargs={"hidden_dim": config.hidden_dim},
        tensorboard_log=str(output / "tensorboard"),
        device=config.device,
        seed=config.seed,
        verbose=0,
    )
    valid_equal_history = replay_baseline(panel, splits["valid"], config.constraints(), "alpha_equal_weight")
    valid_equal_metrics = history_metrics(valid_equal_history)
    write_history(output / "validation_equal_weight_history.csv", valid_equal_history)
    callback = ValidationCallback(
        valid_env, output, config.eval_freq, valid_equal_metrics, config.gamma, config.selection_metric,
    )
    initial_parameters = [parameter.detach().cpu().clone() for parameter in model.policy.parameters()]
    started = time.perf_counter()
    # None turns off SB3's dump; the stub still annotates log_interval as int.
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callback,
        progress_bar=False,
        log_interval=cast(int, None),
    )
    elapsed = time.perf_counter() - started
    parameter_change_l2 = float(
        np.sqrt(
            sum(
                torch.square(after.detach().cpu() - before).sum().item()
                for before, after in zip(initial_parameters, model.policy.parameters(), strict=True)
            )
        )
    )
    model.save(output / "final_model")
    write_history(output / "validation_history.csv", callback.best_history)

    test_env = _env(normalized, splits["test"], config, random_start=False)
    best = PPO.load(output / "best_model", env=test_env, device=config.device)
    test_metrics, test_history = evaluate_model(best, test_env)
    test_metrics["discounted_reward"] = history_metrics(test_history, config.gamma)["discounted_reward"]
    baseline_history = replay_baseline(panel, splits["test"], config.constraints(), "alpha_equal_weight")
    baseline_metrics = history_metrics(baseline_history)
    write_history(output / "test_history.csv", test_history)
    write_history(output / "alpha_equal_weight_history.csv", baseline_history)
    robust_metrics = None
    test_histories = {"ppo": test_history, "equal_weight": baseline_history}
    if panel.is_execution_panel:
        robust_history = replay_baseline(panel, splits["test"], config.constraints(), "robust_top50")
        robust_metrics = history_metrics(robust_history)
        test_histories["robust_top50"] = robust_history
        write_history(output / "robust_top50_history.csv", robust_history)

    test_comparison = compare_metrics(test_metrics, baseline_metrics)
    for name, history in test_histories.items():
        for key, value in history_metrics(history).items():
            model.logger.record(f"test/{name}/{key}", value)
    for key, value in test_comparison.items():
        model.logger.record(f"test/comparison/{key}", value)
    figure = comparison_figure(test_histories)
    figure.savefig(output / "test_comparison.png", dpi=130)
    model.logger.record("test/comparison/curves", TensorBoardFigure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
    model.logger.dump(step=model.num_timesteps)
    model.logger.close()

    metadata = {
        "config": asdict(config),
        "splits": splits,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
        },
        "timing": {
            "training_seconds": elapsed,
            "rollout_seconds": callback.rollout_seconds,
            "update_seconds": callback.update_seconds,
            "validation_seconds": callback.validation_seconds,
            "steps_per_second": model.num_timesteps / max(elapsed, 1e-9),
        },
        "parameter_change_l2": parameter_change_l2,
        "reward_definition": valid_env.reward_evaluator.metadata,
        "selection_metric": config.selection_metric,
        "validation": callback.best_metrics,
        "validation_equal_weight_baseline": valid_equal_metrics,
        "best_update": callback.best_update,
        "completed_updates": callback.update_index,
        "actual_timesteps": model.num_timesteps,
        "test_comparison": test_comparison,
        "test": test_metrics,
        "alpha_equal_weight_baseline": baseline_metrics,
        "robust_top50_baseline": robust_metrics,
        "panel_contract": panel.contract(),
        "stock_ids": panel.stock_ids,
    }
    (output / "metrics.json").write_text(json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8")
    train_env.close()
    valid_env.close()
    test_env.close()
    return metadata


def evaluate_checkpoint(
    panel: PanelData,
    checkpoint: str | Path,
    output_dir: str | Path,
) -> dict[str, float]:
    """Load experiment metadata and export deterministic test-period weights."""
    checkpoint = Path(checkpoint)
    experiment_dir = checkpoint.parent
    metadata = json.loads((experiment_dir / "metrics.json").read_text(encoding="utf-8"))
    if metadata.get("panel_contract") != panel.contract():
        raise ValueError("checkpoint panel contract does not match the supplied snapshot")
    config = ExperimentConfig(**metadata["config"])
    split = tuple(metadata["splits"]["test"])
    with np.load(experiment_dir / "normalizer.npz", allow_pickle=False) as arrays:
        standardizer = Standardizer(arrays["mean"], arrays["scale"])
    normalized = standardizer.transform(panel)
    env = _env(normalized, split, config, random_start=False)
    model = PPO.load(checkpoint, env=env, device=config.device)
    metrics, history = evaluate_model(model, env)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_history(output / "deterministic_weights.csv", history)
    (output / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    env.close()
    return metrics


def train_seed_suite(
    panel: PanelData,
    output_dir: str | Path,
    config: ExperimentConfig,
    seeds: tuple[int, ...] = (7, 17, 29),
) -> dict[str, Any]:
    """Run independent seeds and report validation/test dispersion."""
    from dataclasses import replace

    root = Path(output_dir)
    runs: dict[str, Any] = {}
    for seed in seeds:
        runs[str(seed)] = train_experiment(panel, root / f"seed-{seed}", replace(config, seed=seed))
    test_nav = np.asarray([run["test"]["final_nav"] for run in runs.values()], dtype=float)
    summary = {
        "seeds": list(seeds),
        "test_final_nav_mean": float(test_nav.mean()),
        "test_final_nav_std": float(test_nav.std(ddof=1)) if len(test_nav) > 1 else 0.0,
        "runs": {seed: {"validation": run["validation"], "test": run["test"]} for seed, run in runs.items()},
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "seed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _seeded_env(panel: PanelData, interval: tuple[int, int], config: ExperimentConfig, offset: int) -> PortfolioEnv:
    env = _env(panel, interval, config, random_start=True)
    env.reset(seed=config.seed + offset)
    return env
