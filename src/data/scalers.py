import jax.numpy as jnp
import chex


class RobustScaler:
    """
    Robust scaler using median and IQR for normalization.
    """

    def __init__(self, epsilon: float = 1e-6, min_scale: float = 1e-3):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if min_scale <= 0:
            raise ValueError("min_scale must be positive")
        self.epsilon = epsilon
        self.min_scale = min_scale

    def compute_statistics(
        self, history_values: chex.Array, history_mask: chex.Array | None = None
    ) -> dict[str, chex.Array]:
        """
        Compute median and IQR statistics from historical data with improved numerical stability.
        """
        batch_size, seq_len, num_channels = history_values.shape

        medians = jnp.zeros(batch_size, 1, num_channels)
        iqrs = jnp.ones(batch_size, 1, num_channels)

        for b in range(batch_size):
            for c in range(num_channels):
                channel_data = history_values[b, :, c]

                if history_mask is not None:
                    mask = history_mask[b, :].bool()
                    valid_data = channel_data[mask]
                else:
                    valid_data = channel_data

                if len(valid_data) == 0:
                    continue

                valid_data = valid_data[jnp.isfinite(valid_data)]

                if len(valid_data) == 0:
                    continue

                median_val = jnp.median(valid_data)
                medians[b, 0, c] = median_val

                if len(valid_data) > 1:
                    try:
                        q75 = jnp.quantile(valid_data, 0.75)
                        q25 = jnp.quantile(valid_data, 0.25)
                        iqr_val = q75 - q25
                        iqr_val = jnp.max(iqr_val, jnp.array(self.min_scale))
                        iqrs[b, 0, c] = iqr_val
                    except Exception:
                        std_val = jnp.std(valid_data)
                        iqrs[b, 0, c] = jnp.max(std_val, jnp.array(self.min_scale))
                else:
                    iqrs[b, 0, c] = self.min_scale

        return {"median": medians, "iqr": iqrs}

    def scale(self, data: chex.Array, statistics: dict[str, chex.Array]) -> chex.Array:
        """
        Apply robust scaling: (data - median) / (iqr + epsilon).
        """
        median = statistics["median"]
        iqr = statistics["iqr"]

        denominator = jnp.max(iqr + self.epsilon, jnp.array(self.min_scale))
        scaled_data = (data - median) / denominator
        scaled_data = jnp.clamp(scaled_data, -50.0, 50.0)

        return scaled_data

    def inverse_scale(self, scaled_data: chex.Array, statistics: dict[str, chex.Array]) -> chex.Array:
        """
        Apply inverse robust scaling, now compatible with 3D or 4D tensors.
        """
        median = statistics["median"]
        iqr = statistics["iqr"]

        denominator = jnp.max(iqr + self.epsilon, jnp.array(self.min_scale))

        if scaled_data.ndim == 4:
            denominator = jnp.expand_dims(denominator, axis=-1)
            median = jnp.expand_dims(median, axis=-1)

        return scaled_data * denominator + median
