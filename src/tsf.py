from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
import optax
import chex
import pickle
import numpy as np
from flax import linen as nn
from tqdm import tqdm
from src.data.containers import DataLoader, NpBatchTSContainer
from src.data.time_features import compute_batch_time_features
from src.model.recurrent_lstm_layer import mLSTMWeavingLayerConfig, mLSTMWeavingLayer
from src.model.robust_scaler import RobustScaler


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
        # x: (B*num_channels, seq_len, d_model)
        d_model = x.shape[-1]

        for i in range(self.config.n_layers):
            # Pre-norm + mLSTM layer
            x_norm = nn.LayerNorm()(x)
            dx, (c_state, n_state, m_state) = mLSTMWeavingLayer(config=self.config.weaving_layer_config)(
                x_norm, c_state, n_state, m_state, training
            )
            # Project dx to match x dimension if needed
            if dx.shape[-1] != d_model:
                dx = nn.Dense(d_model, use_bias=False)(dx)
            x = x + dx

            # Pre-norm + MLP
            x_norm = nn.LayerNorm()(x)
            dx = nn.Dense(d_model * 4)(x_norm)
            dx = nn.gelu(dx)
            dx = nn.Dense(d_model)(dx)
            x = x + dx

        # Final norm
        x = nn.LayerNorm()(x)

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


def _handle_missing_data(x):
    """Replace NaN values with zeros."""
    return jnp.where(jnp.isnan(x), 0.0, x)


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
                 x_hist: jax.Array,  # Inputs, (B, h_len, n_channels, 1) -- input_dim is 1 by default.
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


# This is for training, maybe it's worth re-arranging this stuff somewhere else.
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
    model_config: ModelConfig = field(default_factory=lambda: ModelConfig())
    log_every: int = 10000
    num_epochs: int = 100
    batch_size: int = 32
    time_dim: int = 6  # what they call K_max in tempoPFN


def quantile_loss(preds, targets, quantiles):
    errors = targets - preds
    return jnp.mean(jnp.maximum(quantiles * errors, (quantiles - 1) * errors))


class TimeSeriesForecaster:
    """Time series forecaster using xLSTM model."""
    
    def __init__(self, config: TrainingConfig = None, seed: int = 42):
        self.config = config or TrainingConfig()
        self.key = jax.random.PRNGKey(seed)
        self.quantiles = jnp.array(self.config.quantiles)

        #scaler for data
        self.robust_scaler = RobustScaler(epsilon=1e-6, min_scale=1e-3)

        # define model:
        self.model = xLSTMTSModel(config=self.config.model_config)

        # Initialize model with proper input dimension
        x = jax.random.normal(self._next_rng(), (self.config.batch_size, 1, 1, self.config.model_config.input_dim))
        t_hist = jax.random.normal(self._next_rng(), (self.config.batch_size, 1, self.config.time_dim))
        t_future = jax.random.normal(self._next_rng(), (self.config.batch_size, 1, self.config.time_dim))

        params = self.model.init(self._next_rng(), x, t_hist, t_future)

        # Optimizer with gradient clipping
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        )

        # Model state
        self.model_state = ModelState(
            params=params,
            opt_state=self.optimizer.init(params),
            step=0,
        )

        # Build JIT-compiled functions
        self._build_train_step()

    def _build_train_step(self):
        """Build JIT-compiled training step."""
        
        def loss_fn(params, rng, x, t_hist, t_future, y, mask=None):
            x = _handle_missing_data(x)
            x_scaled, (m, iqr) = self.robust_scaler.scale(x)
            y_scaled, _ = self.robust_scaler.scale(y, stats=(m, iqr))
            preds = self.model.apply(params, x_scaled, t_hist, t_future, training=True, rngs={"dropout": rng})
            return quantile_loss(preds, y_scaled, self.quantiles)

        def _train_step(state, x, t_hist, t_future, y, mask, rng):
            loss, grads = jax.value_and_grad(loss_fn)(state.params, rng, x, t_hist, t_future, y, mask)
            updates, opt_state = self.optimizer.update(grads, state.opt_state, state.params)
            params = optax.apply_updates(state.params, updates)
            new_state = state.replace(params=params, opt_state=opt_state, step=state.step + 1)
            return new_state, loss

        self._train_step = jax.jit(_train_step)

    def train_step(self, batch: NpBatchTSContainer) -> float:
        # Use preloaded time features if available, otherwise compute
        if batch.history_time_features is not None and batch.future_time_features is not None:
            history_tf = batch.history_time_features
            future_tf = batch.future_time_features
        else:
            history_tf, future_tf = compute_batch_time_features(
                start=batch.start,
                history_length=batch.history_length,
                future_length=batch.future_length,
                batch_size=batch.batch_size,
                frequency=batch.frequency,
            )

        x, y = batch.history, batch.future

        # Check for NaN or inf in inputs
        if jnp.isnan(x).any():
            print(f"WARNING: NaN detected in input x at step {self.model_state.step}")
        if jnp.isinf(x).any():
            print(f"WARNING: Inf detected in input x at step {self.model_state.step}")
        if jnp.isnan(y).any():
            print(f"WARNING: NaN detected in target y at step {self.model_state.step}")
        if jnp.isinf(y).any():
            print(f"WARNING: Inf detected in target y at step {self.model_state.step}")

        self.model_state, loss = self._train_step(
            self.model_state, x, history_tf, future_tf, y, None, self._next_rng()
        )

        # Check for NaN in loss
        if jnp.isnan(loss):
            print(f"ERROR: NaN loss at step {self.model_state.step}")
            print(f"  Input stats: min={jnp.min(x):.4f}, max={jnp.max(x):.4f}, mean={jnp.mean(x):.4f}")
            print(f"  Target stats: min={jnp.min(y):.4f}, max={jnp.max(y):.4f}, mean={jnp.mean(y):.4f}")

        return loss.item()

    def train(self, loader: DataLoader) -> list[float]:
        losses = []
        for batch in tqdm(loader, total=len(loader)):
            loss = self.train_step(batch)
            if self.model_state.step % self.config.log_every == 0:
                print(f"Step {self.model_state.step:4d} | Loss: {loss:.6f}")
            losses.append(loss)
        print(f"Training complete. Final loss: {losses[-1]:.6f}")
        return losses

    def predict(self, x_hist: Array, t_hist: Array, t_future: Array) -> Array:
        if x_hist.ndim == 3:
            x_hist = x_hist[:, :, :, None]

        x_hist, (m, iqr) = self.robust_scaler.scale(x_hist)
        preds = self.model.apply(self.model_state.params, x_hist, t_hist, t_future, train=False)
        preds = self.robust_scaler.inverse_scale(preds, m, iqr)
        return preds

    def _next_rng(self):
        self.key, rng = jax.random.split(self.key)
        return rng

    def save(self, path: str):
        pickle.dump(self.model_state.params, open(path, "wb"))

    def load(self, path: str):
        params = pickle.load(open(path, "rb"))
        self.model_state = self.model_state.replace(params=params)
