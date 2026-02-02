"""
Gradient Scaled Decay Optimizer (GSD) for JAX/Optax

Per-parameter gradient-scaled weight decay optimizer that applies inverse
gradient scaling: less decay when gradient norm is high, more when low.

Key differences from AdamS (SWD):
- Per-parameter granularity (each tensor has its own gradient magnitude)
- AdamS uses global statistics across all parameters
- Optional EMA smoothing via decay_beta parameter
- Configurable min/max decay bounds

Algorithm for each parameter tensor `p`:
    1. Use Adam's bias-corrected second moment for gradient magnitude:
       param_grad_mag = sqrt(mean(exp_avg_sq_hat))
    2. Optional EMA smoothing if decay_beta < 1.0
    3. Inverse scaling: effective_decay = weight_decay * lr / max(grad_mag, min_grad_norm)
    4. Apply decoupled decay: p *= (1 - clamp(effective_decay, min_decay, max_decay))
    5. Proceed with standard Adam update
"""

from typing import NamedTuple, Optional, Union

import chex
import jax
import jax.numpy as jnp
import optax
from optax._src import base


class GSDState(NamedTuple):
    """State for the GSD optimizer."""
    count: chex.Array
    mu: base.Updates  # First moment estimates
    nu: base.Updates  # Second moment estimates
    grad_mag_ema: Optional[base.Updates]  # Per-parameter gradient magnitude EMA (if decay_beta < 1.0)


def gsd(
    learning_rate: Union[float, optax.Schedule] = 1e-3,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 1e-4,
    decay_beta: float = 1.0,
    min_grad_norm: float = 1e-8,
    min_decay: float = 0.0,
    max_decay: float = 1.0,
) -> base.GradientTransformation:
    """
    GSD optimizer: Adam with per-parameter Gradient Scaled Decay.

    This optimizer applies weight decay scaled by the inverse of each parameter's
    gradient magnitude. Parameters with high gradient magnitude (actively learning)
    receive less decay, while parameters with low gradient magnitude receive more decay.

    Args:
        learning_rate: Learning rate (scalar or schedule).
        b1: Decay rate for first moment estimates.
        b2: Decay rate for second moment estimates.
        eps: Term added to denominator for numerical stability.
        weight_decay: Base weight decay coefficient.
        decay_beta: EMA coefficient for gradient magnitude smoothing.
                    1.0 = use Adam's beta2 smoothing only (recommended for supervised)
                    <1.0 = additional EMA layer (recommended for RL: 0.9)
        min_grad_norm: Minimum gradient norm to avoid division by zero.
        min_decay: Minimum effective decay after scaling.
        max_decay: Maximum effective decay after scaling.

    Returns:
        A GradientTransformation implementing GSD.

    Example:
        >>> optimizer = gsd(learning_rate=1e-3, weight_decay=1e-4)
        >>> opt_state = optimizer.init(params)
        >>> updates, opt_state = optimizer.update(grads, opt_state, params)
        >>> params = optax.apply_updates(params, updates)
    """
    use_ema = decay_beta < 1.0

    def init_fn(params):
        mu = jax.tree_util.tree_map(jnp.zeros_like, params)
        nu = jax.tree_util.tree_map(jnp.zeros_like, params)
        if use_ema:
            # Initialize EMA as None (will be set on first update)
            grad_mag_ema = jax.tree_util.tree_map(lambda p: jnp.zeros(()), params)
        else:
            grad_mag_ema = None
        return GSDState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu, grad_mag_ema=grad_mag_ema)

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError("GSD requires params to be passed for weight decay.")

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

        # Compute per-parameter gradient magnitude
        # For each parameter tensor, compute sqrt(mean(nu_hat))
        def compute_param_grad_mag(v):
            return jnp.sqrt(jnp.mean(v))

        param_grad_mags = jax.tree_util.tree_map(compute_param_grad_mag, nu_hat)

        # Optional EMA smoothing of gradient magnitudes
        if use_ema:
            def update_ema(ema, grad_mag):
                # On first step (ema is zero), initialize with current grad_mag
                is_first = state.count == 0
                new_ema = jnp.where(
                    is_first,
                    grad_mag,
                    decay_beta * ema + (1 - decay_beta) * grad_mag
                )
                return new_ema

            new_grad_mag_ema = jax.tree_util.tree_map(
                update_ema, state.grad_mag_ema, param_grad_mags
            )
            effective_grad_mags = new_grad_mag_ema
        else:
            new_grad_mag_ema = None
            effective_grad_mags = param_grad_mags

        # Compute updates with per-parameter gradient-scaled weight decay
        def compute_update(m, v, p, grad_mag):
            # Adam update component
            adam_update = -lr * m / (jnp.sqrt(v) + eps)

            # Per-parameter gradient-scaled weight decay
            if weight_decay > 0:
                # Inverse scaling: high grad -> low decay, low grad -> high decay
                safe_grad_mag = jnp.maximum(grad_mag, min_grad_norm)
                scaled_decay = weight_decay * lr / safe_grad_mag
                # Clamp to bounds
                effective_decay = jnp.clip(scaled_decay, min_decay, max_decay)
                decay_update = -effective_decay * p
                return adam_update + decay_update
            return adam_update

        new_updates = jax.tree_util.tree_map(
            compute_update, mu_hat, nu_hat, params, effective_grad_mags
        )

        new_state = GSDState(count=count_inc, mu=mu, nu=nu, grad_mag_ema=new_grad_mag_ema)
        return new_updates, new_state

    return base.GradientTransformation(init_fn, update_fn)


def gsd_with_schedule(
    learning_rate_schedule: optax.Schedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 1e-4,
    decay_beta: float = 1.0,
    min_grad_norm: float = 1e-8,
    min_decay: float = 0.0,
    max_decay: float = 1.0,
) -> base.GradientTransformation:
    """
    Convenience function for GSD with a learning rate schedule.

    Args:
        learning_rate_schedule: An optax learning rate schedule.
        b1: Decay rate for first moment estimates.
        b2: Decay rate for second moment estimates.
        eps: Term added to denominator for numerical stability.
        weight_decay: Base weight decay coefficient.
        decay_beta: EMA coefficient for gradient magnitude smoothing.
        min_grad_norm: Minimum gradient norm to avoid division by zero.
        min_decay: Minimum effective decay after scaling.
        max_decay: Maximum effective decay after scaling.

    Returns:
        A GradientTransformation implementing GSD with the given schedule.
    """
    return gsd(
        learning_rate=learning_rate_schedule,
        b1=b1,
        b2=b2,
        eps=eps,
        weight_decay=weight_decay,
        decay_beta=decay_beta,
        min_grad_norm=min_grad_norm,
        min_decay=min_decay,
        max_decay=max_decay,
    )
