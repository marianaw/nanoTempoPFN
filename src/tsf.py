from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
import optax
import chex
import pickle
import numpy as np
from flax import linen as nn

from src.data.containers import NpBatchTSContainer
from src.data.time_features import compute_batch_time_features

from src.model.recurrent_lstm_layer import mLSTMWeavingLayerConfig, mLSTMWeavingLayer


@dataclass
class WeavingBlockLSTMConfig:
    n_layers: int = 4
    embedding_dim: int = 8
    num_heads: int = 2
    weaving_layer_config: mLSTMWeavingLayerConfig = field(default_factory=lambda: mLSTMWeavingLayerConfig())

    def __post_init__(self):
        self.weaving_layer_config.embedding_dim = self.embedding_dim
        self.weaving_layer_config.num_heads = self.num_heads


Params = chex.ArrayTree
PRNGKey = chex.PRNGKey
Array = chex.Array


class WeavingLSTM(nn.Module):

    config: WeavingBlockLSTMConfig

    @nn.compact
    def __call__(self, x, c_state, n_state, m_state, training=True):
        # x here should be the concatenation of the encoded history and the target pos embedding, shape (B*num_channels, len_history+len_target, embedding_dim)
        for i in range(self.config.n_layers):
            x, (c_state, n_state, m_state) = mLSTMWeavingLayer(config=self.config.weaving_layer_config)(x, c_state, n_state, m_state, training)

        return x, (c_state, n_state, m_state)


@dataclass
class ModelConfig:
    input_dim: int = 1
    embedding_dim: int = 24
    head_embedding_dim: int = 8
    num_heads: int = 2
    n_layers: int = 4
    output_dim: int = 9

    weaving_block_config: WeavingBlockLSTMConfig = field(default_factory=lambda: WeavingBlockLSTMConfig())

    def __post_init__(self):
        self.weaving_block_config.weaving_layer_config.embedding_dim = self.head_embedding_dim
        self.weaving_block_config.weaving_layer_config.num_heads = self.num_heads
        self.weaving_block_config.n_layers = self.n_layers


class xLSTMTSModel(nn.Module):
    config: ModelConfig

    def setup(self):
        self.pos_encoder = nn.Dense(self.config.embedding_dim)

        # channel-wise, for now 1 to embedding_dim
        self.hist_encoder = nn.Dense(self.config.embedding_dim)

        self.weaving_block = WeavingLSTM(config=self.config.weaving_block_config,
                                         name="weaving_lstm")
        self.output_proj = nn.Dense(self.config.output_dim)

    def __call__(self,
                 x_hist: jax.Array,  # Inputs, (B, h_len, n_channels, 1)
                 t_hist: jax.Array,  # History timestamps, (B, h_len, K)
                 t_future: jax.Array,  # Future timestamps, (B, f_len, K)
                 training: bool = False) -> jax.Array:

        B, h_len, n_channels, _ = x_hist.shape
        f_len = t_future.shape[1]

        # history pos embedding
        h_pos = self.pos_encoder(t_hist)  # (B, h_len, embedding_dim)
        # (B, h_len, 1, embedding_dim) -- we add a channel dimension
        h_pos = h_pos[:, :, None, :]
        h_pos = jnp.repeat(h_pos, repeats=n_channels, axis=2, total_repeat_length=n_channels)  # (B, h_len, n_channels, embedding_dim)

        # future pos embedding
        f_pos = self.pos_encoder(t_future)  # (B, f_len, embedding_dim)
        f_pos = f_pos[:, :, None, :]  # (B, f_len, 1, embedding_dim)
        f_pos = jnp.repeat(f_pos, repeats=n_channels, axis=2, total_repeat_length=n_channels)  # (B, f_len, n_channels, embedding_dim)

        # embedd history
        # x_hist is (B, h_len, n_channels, 1) -- n_channels = 1 for now
        # (B, h_len, n_channels, embedding_dim)  -- n_channels = 1 for now
        h_emb = self.hist_encoder(x_hist)
        h_emb = h_emb + h_pos  # (B, h_len, n_channels, embedding_dim)

        # generate predictions with input given by: embedded history + history pos encoding + future pos encoding
        # reshape to (B * n_channels, h_len, embedding_dim)
        h_emb = h_emb.reshape(B * n_channels, h_len, self.config.embedding_dim)
        f_pos = f_pos.reshape(B * n_channels, f_len, self.config.embedding_dim)
        inputs = jnp.concatenate([h_emb, f_pos], axis=1)

        NH = self.config.weaving_block_config.weaving_layer_config.num_heads
        DH = self.config.weaving_block_config.weaving_layer_config.embedding_dim

        c_state = jnp.zeros((B * n_channels, NH, DH, DH))
        n_state = jnp.zeros((B * n_channels, NH, DH, 1))
        m_state = jnp.zeros((B * n_channels, NH, 1, 1))


        preds, _ = self.weaving_block(inputs, c_state, n_state, m_state, training=training)
        preds = preds[:, -f_len:, :]
        preds = self.output_proj(preds)
        preds = preds.reshape(B, f_len, n_channels, self.config.output_dim)
        return preds


@chex.dataclass(frozen=True)
class ModelState:
    params: Params
    opt_state: optax.OptState
    step: int


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    dropout: float = 0.1
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def quantile_loss(preds, targets, quantiles):
    errors = targets - preds
    return jnp.mean(jnp.maximum(quantiles * errors, (quantiles - 1) * errors))


class TimeSeriesForecaster:
    """Time series forecaster using xLSTM model."""
    pass

    # def __init__(self, config: WeavingLSTMConfig = None, seed: int = 42):
    #     self.config = config or WeavingLSTMConfig()
    #     self.key = jax.random.PRNGKey(seed)
    #     self.quantiles = jnp.array(config.quantiles)

    #     # Initialize model with proper input dimension
    #     rng_init = self._next_rng()
    #     dummy_input = jnp.ones((1, self.config.context_length, self.config.input_dim))
    #     params = self.model.init({"params": rng_init, "dropout": rng_init}, dummy_input)

    #     # Optimizer
    #     self.optimizer = optax.adamw(
    #         learning_rate=self.config.learning_rate, 
    #         weight_decay=self.config.weight_decay
    #     )

    #     # Model state
    #     self.model_state = ModelState(
    #         params=params,
    #         opt_state=self.optimizer.init(params),
    #         step=0,
    #     )

    #     # Build JIT-compiled functions
    #     self._build_train_step()
        
    #     # Training history
    #     self.losses = []

    # def _build_train_step(self):
    #     """Build JIT-compiled training step."""
        
    #     def loss_fn(params, rng, x, y, mask):
    #         preds = self.model.apply(params, x, train=True, rngs={"dropout": rng})
    #         return quantile_loss(preds, y, self.quantiles)  # (B, T, D)

    #     def _train_step(state, x, y, mask, rng):
    #         loss, grads = jax.value_and_grad(loss_fn)(state.params, rng, x, y, mask)
    #         updates, opt_state = self.optimizer.update(grads, state.opt_state, state.params)
    #         params = optax.apply_updates(state.params, updates)
    #         new_state = state.replace(params=params, opt_state=opt_state, step=state.step + 1)
    #         return new_state, loss

    #     self._train_step = jax.jit(_train_step)

    # def train(self, batch: NpBatchTSContainer):
        
    #     for step in range(num_steps):
    #         x, y, mask = dataset.sample_batch(batch_size)
    #         x, y, mask = jnp.array(x), jnp.array(y), jnp.array(mask)
            
    #         self.model_state, loss = self._train_step(
    #             self.model_state, x, y, mask, self._next_rng()
    #         )
    #         self.losses.append(float(loss))
            
    #         if step % log_every == 0:
    #             print(f"Step {step:4d} | Loss: {loss:.6f}")
        
    #     print(f"Training complete. Final loss: {self.losses[-1]:.6f}")
    #     return self.losses

    # def predict(self, x: Array) -> Array:
    #     """One-step ahead prediction."""
    #     if x.ndim == 2:
    #         x = x[:, :, None]
    #     return self.model.apply(self.model_state.params, x, train=False)

    # def generate(self, context: Array, n_steps: int = None) -> Array:
    #     """Autoregressive generation from context.
        
    #     Args:
    #         context: Initial context of shape (B, S, D) or (B, S)
    #         n_steps: Number of steps to generate (default: config.max_new_steps)
        
    #     Returns:
    #         Full sequence including context and generated steps: (B, S + n_steps, D)
    #     """
    #     n_steps = n_steps or self.config.max_new_steps
        
    #     # Ensure 3D input
    #     if context.ndim == 2:
    #         context = context[:, :, None]
        
    #     B, S, D = context.shape
    #     buffer = jnp.concatenate([context, jnp.zeros((B, n_steps, D))], axis=1)

    #     for i in range(n_steps):
    #         ctx = buffer[:, i:i + S, :]
    #         preds = self.model.apply(self.model_state.params, ctx, train=False)
    #         next_val = preds[:, -1, :]  # (B, D)
    #         buffer = buffer.at[:, S + i, :].set(next_val)
        
    #     return np.array(buffer)

    # def _next_rng(self):
    #     self.key, rng = jax.random.split(self.key)
    #     return rng

    # def save(self, path: str):
    #     pickle.dump(self.model_state.params, open(path, "wb"))

    # def load(self, path: str):
    #     params = pickle.load(open(path, "rb"))
    #     self.model_state = self.model_state.replace(params=params)
