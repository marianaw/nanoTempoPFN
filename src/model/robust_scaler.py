"""This is a JAXified version of TempoPFN's scaler"""
import jax
import jax.numpy as jnp
import chex
from flax import linen as nn


class RobustScaler:

    def __init__(self, epsilon, min_scale):
        self.epsilon = epsilon
        self.min_scale = min_scale

    def _compute_stats(
        self, x: chex.Array, mask: chex.Array | None = None
    ) -> tuple[chex.Array, chex.Array]:
        """
        Compute median and IQR statistics
        """
        # Recall that x will have the last dimension =1, as input_dim is 1. x is of shape (B, seq_len, num_channels, dim)
        # Median is seq_length-wise, per channel, per dimension.
        reduction_axis = 1
        # (B, 1, num_channels, dim)
        medians = jnp.median(x, axis=reduction_axis, keepdims=True)
        # also (B, 1, num_channels, dim)
        q75 = jnp.quantile(x, 0.75, axis=reduction_axis, keepdims=True)
        # also (B, 1, num_channels, dim)
        q25 = jnp.quantile(x, 0.25, axis=reduction_axis, keepdims=True)
        iqrs = q75 - q25
        iqrs = jnp.where(iqrs >= self.min_scale, iqrs, self.min_scale)
        return medians, iqrs

    def _scale(self, x: chex.Array, medians: chex.Array, iqrs: chex.Array) -> chex.Array:
        dd = iqrs + self.epsilon
        dd = jnp.where(dd >= self.min_scale, dd, self.min_scale)
        scaled = (x - medians)/dd
        scaled = jax.lax.clamp(min=-50.0, x=scaled, max=50.0)
        return scaled

    def inverse_scale(self, x: chex.Array, medians: chex.Array, iqrs: chex.Array) -> chex.Array:
        dd = iqrs + self.epsilon
        dd = jnp.where(dd >= self.min_scale, dd, self.min_scale)
        # Only expand dims if x has more dimensions AND last dim > 1 (e.g., quantiles)
        if x.ndim == 4 and x.shape[-1] > 1:
            dd = jnp.expand_dims(dd, -1)
            medians = jnp.expand_dims(medians, -1)
        return x * dd + medians

    def scale(self, x: chex.Array,
               mask: chex.Array | None = None,
               stats: tuple[chex.Array, chex.Array] = None):
        """Applies layer normalization on the input.

        Args:
        x: the inputs

        Returns:
        Normalized inputs (the same shape as inputs).
        """
        if stats is None:
            medians, iqrs = self._compute_stats(x, mask)
        else:
            medians, iqrs = stats

        scaled_x = self._scale(x, medians, iqrs)

        return scaled_x, (medians, iqrs)
