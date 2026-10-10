from __future__ import annotations

from typing import Any, Callable, cast

import torch
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy


class MaskedDiagonalGaussian:
    """Independent Gaussian whose loss ignores unavailable stock slots."""

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor, mask: torch.Tensor) -> None:
        self.mean = mean
        self.std = log_std.exp().expand_as(mean)
        self.mask = mask.unsqueeze(-1).expand_as(mean)
        self.distribution = torch.distributions.Normal(mean, self.std)

    def sample(self) -> torch.Tensor:
        return self.distribution.rsample()

    def mode(self) -> torch.Tensor:
        return self.mean

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return (self.distribution.log_prob(actions) * self.mask).sum(dim=(-1, -2))

    def entropy(self) -> torch.Tensor:
        return (self.distribution.entropy() * self.mask).sum(dim=(-1, -2))


class SharedStockNetwork(torch.nn.Module):
    """EIIE-style parameter sharing with masked portfolio summaries."""

    def __init__(self, feature_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        stock_input = feature_dim + 4  # features, alpha, weight, investable, present
        self.actor_encoder = torch.nn.Sequential(
            torch.nn.Linear(stock_input, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.critic_encoder = torch.nn.Sequential(
            torch.nn.Linear(stock_input, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        combined = hidden_dim * 3 + 2
        self.actor_head = torch.nn.Sequential(
            torch.nn.Linear(combined, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, 2)
        )
        self.critic_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _summaries(encoded: torch.Tensor, present: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        present3 = present.unsqueeze(-1)
        count = present3.sum(dim=1).clamp_min(1.0)
        mean = (encoded * present3).sum(dim=1) / count
        holding = (encoded * weights.unsqueeze(-1)).sum(dim=1)
        return mean, holding

    @staticmethod
    def _stock_input(obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [
                obs["features"],
                obs["alpha"].unsqueeze(-1),
                obs["current_weights"].unsqueeze(-1),
                obs["investable_mask"].unsqueeze(-1),
                obs["present_mask"].unsqueeze(-1),
            ],
            dim=-1,
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._stock_input(obs)
        present = obs["present_mask"]
        weights = obs["current_weights"]
        actor_encoded = self.actor_encoder(x) * present.unsqueeze(-1)
        critic_encoded = self.critic_encoder(x) * present.unsqueeze(-1)
        actor_mean, actor_holding = self._summaries(actor_encoded, present, weights)
        critic_mean, critic_holding = self._summaries(critic_encoded, present, weights)
        slots = x.shape[1]
        context = torch.cat([actor_mean, actor_holding, obs["account"]], dim=-1)
        context = context.unsqueeze(1).expand(-1, slots, -1)
        action_mean = self.actor_head(torch.cat([actor_encoded, context], dim=-1))
        value = self.critic_head(torch.cat([critic_mean, critic_holding, obs["account"]], dim=-1))
        return action_mean, value


class MaskedStockActorCriticPolicy(BasePolicy):
    """Minimal SB3 policy implementing a masked, shared per-stock actor-critic."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *,
        hidden_dim: int = 64,
        log_std_init: float = -0.5,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        use_sde: bool = False,
        **_: Any,
    ) -> None:
        if use_sde:
            raise ValueError("gSDE is not supported by the masked stock policy")
        if not isinstance(observation_space, spaces.Dict) or not isinstance(action_space, spaces.Box):
            raise TypeError("masked stock policy requires Dict observations and Box actions")
        super().__init__(observation_space, action_space, normalize_images=False)
        features_space = observation_space.spaces["features"]
        if not isinstance(features_space, spaces.Box) or features_space.shape is None:
            raise TypeError("features observation must be a Box with a declared shape")
        feature_dim = int(features_space.shape[-1])
        self.slots = int(features_space.shape[0])
        if action_space.shape != (self.slots * 2,):
            raise ValueError("action space must contain two values per stock slot")
        self.network = SharedStockNetwork(feature_dim, hidden_dim)
        self.log_std = torch.nn.Parameter(torch.full((1, 1, 2), float(log_std_init)))
        optimizer_kwargs = optimizer_kwargs or {}
        # torch.optim.Optimizer does not declare lr; Adam and the other SB3 defaults do.
        optimizer_factory = cast(Callable[..., torch.optim.Optimizer], optimizer_class)
        self.optimizer = optimizer_factory(self.parameters(), lr=lr_schedule(1), **optimizer_kwargs)

    def _distribution(self, obs: dict[str, torch.Tensor]) -> tuple[MaskedDiagonalGaussian, torch.Tensor]:
        mean, value = self.network(obs)
        mask = obs["investable_mask"]
        return MaskedDiagonalGaussian(mean, self.log_std, mask), value

    def forward(self, obs: dict[str, torch.Tensor], deterministic: bool = False):
        distribution, value = self._distribution(obs)
        actions = distribution.mode() if deterministic else distribution.sample()
        log_prob = distribution.log_prob(actions)
        return actions.flatten(start_dim=1), value, log_prob

    def evaluate_actions(self, obs: dict[str, torch.Tensor], actions: torch.Tensor):
        distribution, value = self._distribution(obs)
        actions = actions.reshape(-1, self.slots, 2)
        return value, distribution.log_prob(actions), distribution.entropy()

    def predict_values(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.network(obs)[1]

    def get_distribution(self, obs: dict[str, torch.Tensor]) -> MaskedDiagonalGaussian:
        return self._distribution(obs)[0]

    def _predict(self, observation: dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        distribution = self.get_distribution(observation)
        actions = distribution.mode() if deterministic else distribution.sample()
        return actions.flatten(start_dim=1)

    def reset_noise(self, n_envs: int = 1) -> None:
        del n_envs

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(lr_schedule=self._dummy_schedule)
        return data
