"""
SimbaV2 Agent with Gradient Scaled Decay (GSD) optimizer.

This variant of the SimbaV2 agent uses the GSD optimizer which applies
per-parameter gradient-scaled weight decay instead of standard Adam.
"""

import functools
from dataclasses import dataclass
from typing import Dict, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax

from scale_rl.agents.base_agent import BaseAgent
from scale_rl.agents.jax_utils.network import Network, PRNGKey
from scale_rl.agents.jax_utils.gsd_optax import gsd
from scale_rl.agents.simbaV2.simbaV2_network import (
    SimbaV2Actor,
    SimbaV2Critic,
    SimbaV2DoubleCritic,
    SimbaV2Temperature,
)
from scale_rl.agents.simbaV2.simbaV2_update import (
    l2normalize_network,
    update_actor,
    update_critic,
    update_target_network,
    update_temperature,
)
from scale_rl.buffers.base_buffer import Batch


@dataclass(frozen=True)
class SimbaV2GSDConfig:
    """Configuration for SimbaV2 with Gradient Scaled Decay."""
    seed: int
    normalize_observation: bool
    normalize_reward: bool
    normalized_g_max: float

    load_only_param: bool
    load_param_key: bool
    load_observation_normalizer: bool
    load_reward_normalizer: bool

    learning_rate_init: float
    learning_rate_end: float
    learning_rate_decay_rate: float
    learning_rate_decay_step: int

    # GSD specific
    weight_decay: float
    decay_beta: float
    min_grad_norm: float
    min_decay: float
    max_decay: float

    actor_num_blocks: int
    actor_hidden_dim: int
    actor_c_shift: float
    actor_scaler_init: float
    actor_scaler_scale: float
    actor_alpha_init: float
    actor_alpha_scale: float
    actor_bc_alpha: float

    critic_use_cdq: bool
    critic_num_blocks: int
    critic_hidden_dim: int
    critic_c_shift: float
    critic_num_bins: int
    critic_min_v: float
    critic_max_v: float
    critic_scaler_init: float
    critic_scaler_scale: float
    critic_alpha_init: float
    critic_alpha_scale: float

    target_tau: float

    temp_initial_value: float
    temp_target_entropy: float
    temp_target_entropy_coef: float

    gamma: float
    n_step: int


@functools.partial(
    jax.jit,
    static_argnames=(
        "observation_dim",
        "action_dim",
        "cfg",
    ),
)
def _init_simbav2_gsd_networks(
    observation_dim: int,
    action_dim: int,
    cfg: SimbaV2GSDConfig,
) -> Tuple[PRNGKey, Network, Network, Network, Network]:
    """Initialize networks with GSD optimizer."""
    fake_observations = jnp.zeros((1, observation_dim))
    fake_actions = jnp.zeros((1, action_dim))

    rng = jax.random.PRNGKey(cfg.seed)
    rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

    # Create learning rate schedule
    lr_schedule = optax.linear_schedule(
        init_value=cfg.learning_rate_init,
        end_value=cfg.learning_rate_end,
        transition_steps=cfg.learning_rate_decay_step,
    )

    # Actor network with GSD optimizer
    actor = Network.create(
        network_def=SimbaV2Actor(
            num_blocks=cfg.actor_num_blocks,
            hidden_dim=cfg.actor_hidden_dim,
            action_dim=action_dim,
            c_shift=cfg.actor_c_shift,
            scaler_init=cfg.actor_scaler_init,
            scaler_scale=cfg.actor_scaler_scale,
            alpha_init=cfg.actor_alpha_init,
            alpha_scale=cfg.actor_alpha_scale,
        ),
        network_inputs={"rngs": actor_key, "observations": fake_observations},
        tx=gsd(
            learning_rate=lr_schedule,
            weight_decay=cfg.weight_decay,
            decay_beta=cfg.decay_beta,
            min_grad_norm=cfg.min_grad_norm,
            min_decay=cfg.min_decay,
            max_decay=cfg.max_decay,
        ),
    )

    # Critic network definition
    if cfg.critic_use_cdq:
        critic_network_def = SimbaV2DoubleCritic(
            num_blocks=cfg.critic_num_blocks,
            hidden_dim=cfg.critic_hidden_dim,
            c_shift=cfg.critic_c_shift,
            num_bins=cfg.critic_num_bins,
            min_v=cfg.critic_min_v,
            max_v=cfg.critic_max_v,
            scaler_init=cfg.critic_scaler_init,
            scaler_scale=cfg.critic_scaler_scale,
            alpha_init=cfg.critic_alpha_init,
            alpha_scale=cfg.critic_alpha_scale,
        )
    else:
        critic_network_def = SimbaV2Critic(
            num_blocks=cfg.critic_num_blocks,
            hidden_dim=cfg.critic_hidden_dim,
            c_shift=cfg.critic_c_shift,
            num_bins=cfg.critic_num_bins,
            min_v=cfg.critic_min_v,
            max_v=cfg.critic_max_v,
            scaler_init=cfg.critic_scaler_init,
            scaler_scale=cfg.critic_scaler_scale,
            alpha_init=cfg.critic_alpha_init,
            alpha_scale=cfg.critic_alpha_scale,
        )

    # Critic network with GSD optimizer
    critic = Network.create(
        network_def=critic_network_def,
        network_inputs={
            "rngs": critic_key,
            "observations": fake_observations,
            "actions": fake_actions,
        },
        tx=gsd(
            learning_rate=lr_schedule,
            weight_decay=cfg.weight_decay,
            decay_beta=cfg.decay_beta,
            min_grad_norm=cfg.min_grad_norm,
            min_decay=cfg.min_decay,
            max_decay=cfg.max_decay,
        ),
    )

    # Target critic (no optimizer needed)
    target_network_def = critic_network_def
    target_critic = Network.create(
        network_def=target_network_def,
        network_inputs={
            "rngs": critic_key,
            "observations": fake_observations,
            "actions": fake_actions,
        },
        tx=None,
    )

    # Temperature network with GSD optimizer
    temperature = Network.create(
        network_def=SimbaV2Temperature(cfg.temp_initial_value),
        network_inputs={
            "rngs": temp_key,
        },
        tx=gsd(
            learning_rate=lr_schedule,
            weight_decay=cfg.weight_decay,
            decay_beta=cfg.decay_beta,
            min_grad_norm=cfg.min_grad_norm,
            min_decay=cfg.min_decay,
            max_decay=cfg.max_decay,
        ),
    )

    # L2-normalize networks after initialization
    actor = l2normalize_network(actor)
    critic = l2normalize_network(critic)
    target_critic = l2normalize_network(target_critic)

    return rng, actor, critic, target_critic, temperature


@jax.jit
def _sample_simbav2_gsd_actions(
    rng: PRNGKey,
    actor: Network,
    observations: jnp.ndarray,
    temperature: float = 1.0,
) -> Tuple[PRNGKey, jnp.ndarray]:
    rng, key = jax.random.split(rng)
    dist, _ = actor(observations=observations, temperature=temperature)
    actions = dist.sample(seed=key)
    return rng, actions


@functools.partial(
    jax.jit,
    static_argnames=(
        "gamma",
        "n_step",
        "critic_use_cdq",
        "critic_min_v",
        "critic_max_v",
        "critic_num_bins",
        "target_tau",
        "temp_target_entropy",
        "actor_bc_alpha",
    ),
)
def _update_simbav2_gsd_networks(
    rng: PRNGKey,
    actor: Network,
    critic: Network,
    target_critic: Network,
    temperature: Network,
    batch: Batch,
    gamma: float,
    n_step: int,
    actor_bc_alpha: float,
    critic_use_cdq: bool,
    critic_min_v: float,
    critic_max_v: float,
    critic_num_bins: int,
    target_tau: float,
    temp_target_entropy: float,
) -> Tuple[PRNGKey, Network, Network, Network, Network, Dict[str, float]]:
    rng, actor_key, critic_key = jax.random.split(rng, 3)

    new_actor, actor_info = update_actor(
        key=actor_key,
        actor=actor,
        critic=critic,
        temperature=temperature,
        batch=batch,
        use_cdq=critic_use_cdq,
        bc_alpha=actor_bc_alpha,
    )

    new_temperature, temperature_info = update_temperature(
        temperature=temperature,
        entropy=actor_info["actor/entropy"],
        target_entropy=temp_target_entropy,
    )

    new_critic, critic_info = update_critic(
        key=critic_key,
        actor=new_actor,
        critic=critic,
        target_critic=target_critic,
        temperature=new_temperature,
        batch=batch,
        use_cdq=critic_use_cdq,
        min_v=critic_min_v,
        max_v=critic_max_v,
        num_bins=critic_num_bins,
        gamma=gamma,
        n_step=n_step,
    )

    new_target_critic, target_critic_info = update_target_network(
        network=new_critic,
        target_network=target_critic,
        target_tau=target_tau,
    )

    info = {
        **actor_info,
        **critic_info,
        **target_critic_info,
        **temperature_info,
    }

    return (rng, new_actor, new_critic, new_target_critic, new_temperature, info)


class SimbaV2AgentGSD(BaseAgent):
    """
    SimbaV2 Agent with Gradient Scaled Decay (GSD) optimizer.

    This agent is identical to SimbaV2Agent but uses the GSD optimizer
    which applies per-parameter gradient-scaled weight decay based on
    each parameter's gradient magnitude.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        cfg: dict,
    ):
        self._observation_dim = observation_space.shape[-1]
        self._action_dim = action_space.shape[-1]

        cfg["temp_target_entropy"] = cfg["temp_target_entropy_coef"] * self._action_dim

        # Set default GSD parameters if not provided
        if "weight_decay" not in cfg:
            cfg["weight_decay"] = 1e-4
        if "decay_beta" not in cfg:
            cfg["decay_beta"] = 1.0  # No additional EMA by default
        if "min_grad_norm" not in cfg:
            cfg["min_grad_norm"] = 1e-8
        if "min_decay" not in cfg:
            cfg["min_decay"] = 0.0
        if "max_decay" not in cfg:
            cfg["max_decay"] = 1.0

        super(SimbaV2AgentGSD, self).__init__(
            observation_space,
            action_space,
            cfg,
        )

        # Map dictionary to dataclass
        self._cfg = SimbaV2GSDConfig(**cfg)

        # Initialize networks with GSD optimizer
        (
            self._rng,
            self._actor,
            self._critic,
            self._target_critic,
            self._temperature,
        ) = _init_simbav2_gsd_networks(
            self._observation_dim, self._action_dim, self._cfg
        )

    def sample_actions(
        self,
        interaction_step: int,
        prev_timestep: Dict[str, np.ndarray],
        training: bool,
    ) -> np.ndarray:
        if training:
            temperature = 1.0
        else:
            temperature = 0.0

        observations = jnp.asarray(prev_timestep["next_observation"])

        self._rng, actions = _sample_simbav2_gsd_actions(
            self._rng, self._actor, observations, temperature
        )
        actions = np.array(actions)

        return actions

    def update(self, update_step: int, batch: Dict[str, np.ndarray]) -> Dict:
        for key, value in batch.items():
            batch[key] = jnp.asarray(value)

        (
            self._rng,
            self._actor,
            self._critic,
            self._target_critic,
            self._temperature,
            update_info,
        ) = _update_simbav2_gsd_networks(
            rng=self._rng,
            actor=self._actor,
            critic=self._critic,
            target_critic=self._target_critic,
            temperature=self._temperature,
            batch=batch,
            gamma=self._cfg.gamma,
            n_step=self._cfg.n_step,
            critic_use_cdq=self._cfg.critic_use_cdq,
            critic_min_v=self._cfg.critic_min_v,
            critic_max_v=self._cfg.critic_max_v,
            critic_num_bins=self._cfg.critic_num_bins,
            target_tau=self._cfg.target_tau,
            temp_target_entropy=self._cfg.temp_target_entropy,
            actor_bc_alpha=self._cfg.actor_bc_alpha,
        )

        for key, value in update_info.items():
            if isinstance(value, dict):
                continue
            if isinstance(value, str):
                continue
            update_info[key] = float(value)

        return update_info

    def save(self, path: str) -> None:
        self._actor.save(path + "/actor")
        self._critic.save(path + "/critic")
        self._target_critic.save(path + "/target_critic")
        self._temperature.save(path + "/temperature")

    def load(self, path: str) -> None:
        only_param = self._cfg.load_only_param
        param_key = self._cfg.load_param_key

        self._actor = self._actor.load(path + "/actor", param_key, only_param)
        self._critic = self._critic.load(path + "/critic", param_key, only_param)
        self._target_critic = self._target_critic.load(
            path + "/target_critic", param_key, only_param
        )
        self._temperature = self._temperature.load(
            path + "/temperature", None, only_param
        )
