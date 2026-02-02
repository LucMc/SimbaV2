"""
Scheduled Weight Decay Optimizer (AdamS) for JAX/Optax

Implements Adam with Scheduled Weight Decay as described in
"Stable Weight Decay Regularization".

The key insight: weight decay is scaled by the inverse of global gradient
magnitude. When gradients are large (active learning), decay is reduced.
When gradients are small (stable weights), decay is stronger.

Weight decay formula: p *= (1 - weight_decay * lr / exp_avg_mean_sqrt)
where exp_avg_mean_sqrt is the sqrt of the mean of all exp_avg_sq_hat values.
"""

from typing import NamedTuple, Union

import chex
import jax
import jax.numpy as jnp
import optax
from optax._src import base


class AdamSState(NamedTuple):
    """State for the AdamS optimizer (combined Adam + Scheduled Weight Decay)."""
    count: chex.Array
    mu: base.Updates  # First moment estimates
    nu: base.Updates  # Second moment estimates


def adams(
    learning_rate: Union[float, optax.Schedule] = 1e-3,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 1e-4,
) -> base.GradientTransformation:
    """
    AdamS optimizer: Adam with Scheduled Weight Decay.

    This is a memory-efficient implementation that combines Adam and
    scheduled weight decay into a single transformation, sharing the
    second moment estimates.

    Args:
        learning_rate: Learning rate (scalar or schedule).
        b1: Decay rate for first moment estimates.
        b2: Decay rate for second moment estimates.
        eps: Term added to denominator for numerical stability.
        weight_decay: Base weight decay coefficient.

    Returns:
        A GradientTransformation implementing AdamS.

    Example:
        >>> optimizer = adams(learning_rate=1e-3, weight_decay=1e-4)
        >>> opt_state = optimizer.init(params)
        >>> updates, opt_state = optimizer.update(grads, opt_state, params)
        >>> params = optax.apply_updates(params, updates)
    """

    def init_fn(params):
        mu = jax.tree_util.tree_map(jnp.zeros_like, params)
        nu = jax.tree_util.tree_map(jnp.zeros_like, params)
        return AdamSState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError("AdamS requires params to be passed for weight decay.")

        count_inc = optax.safe_int32_increment(state.count)

        # Get learning rate (handle schedules)
        if callable(learning_rate):
            lr = learning_rate(count_inc)
        else:
            lr = learning_rate

        # Update first and second moment estimates
        mu = jax.tree_util.tree_map(
            lambda m, g: b1 * m + (1 - b1) * g, state.mu, updates
        )
        nu = jax.tree_util.tree_map(
            lambda v, g: b2 * v + (1 - b2) * jnp.square(g), state.nu, updates
        )

        # Bias correction
        bias_correction1 = 1 - b1 ** count_inc
        bias_correction2 = 1 - b2 ** count_inc

        mu_hat = jax.tree_util.tree_map(lambda m: m / bias_correction1, mu)
        nu_hat = jax.tree_util.tree_map(lambda v: v / bias_correction2, nu)

        # Compute global gradient magnitude for scheduled weight decay
        # This is the sqrt of the mean of all nu_hat values
        leaves = jax.tree_util.tree_leaves(nu_hat)
        total_sum = sum(jnp.sum(leaf) for leaf in leaves)
        total_count = sum(leaf.size for leaf in leaves)
        exp_avg_mean_sqrt = jnp.sqrt(total_sum / total_count)

        # Prevent division by zero
        exp_avg_mean_sqrt = jnp.maximum(exp_avg_mean_sqrt, 1e-10)

        # Compute Adam updates: -lr * mu_hat / (sqrt(nu_hat) + eps)
        def compute_update(m, v, p):
            # Adam update component
            adam_update = -lr * m / (jnp.sqrt(v) + eps)

            # Scheduled weight decay: -lr * weight_decay * p / exp_avg_mean_sqrt
            if weight_decay > 0:
                decay_update = -lr * weight_decay * p / exp_avg_mean_sqrt
                return adam_update + decay_update
            return adam_update

        new_updates = jax.tree_util.tree_map(
            compute_update, mu_hat, nu_hat, params
        )

        return new_updates, AdamSState(count=count_inc, mu=mu, nu=nu)

    return base.GradientTransformation(init_fn, update_fn)


def adams_with_schedule(
    learning_rate_schedule: optax.Schedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 1e-4,
) -> base.GradientTransformation:
    """
    Convenience function for AdamS with a learning rate schedule.

    Args:
        learning_rate_schedule: An optax learning rate schedule.
        b1: Decay rate for first moment estimates.
        b2: Decay rate for second moment estimates.
        eps: Term added to denominator for numerical stability.
        weight_decay: Base weight decay coefficient.

    Returns:
        A GradientTransformation implementing AdamS with the given schedule.
    """
    return adams(
        learning_rate=learning_rate_schedule,
        b1=b1,
        b2=b2,
        eps=eps,
        weight_decay=weight_decay,
    )
