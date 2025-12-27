from dataclasses import dataclass
import jax
import jax.numpy as jnp
import optax
import chex
import pickle
import numpy as np

import sys
sys.path.append('/Users/mariana/Documents/research/xlstm-jax')

from models import xLSTMTabModel, xLSTMTabModelConfig
from xlstm_jax.models.xlstm_clean.blocks.mlstm.layer import mLSTMLayerConfig
from xlstm_jax.models.xlstm_clean.components.feedforward import FeedForwardConfig
from xlstm_jax.models.xlstm_clean.blocks.mlstm.block import xLSTMBlockConfig


@dataclass
class ModelConfig:
    input_dim: int = 1
    embedding_dim: int = 32
    num_blocks: int = 2
    num_heads: int = 4
    context_length: int = 32
    max_new_steps: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    dropout: float = 0.1


Params = chex.ArrayTree
PRNGKey = chex.PRNGKey
Array = chex.Array


@chex.dataclass(frozen=True)
class ModelState:
    params: Params
    opt_state: optax.OptState
    step: int


class xLSTMTimeSeries:
    """xLSTM model for time series forecasting."""

    def __init__(self, config: ModelConfig = None, seed: int = 42):
        self.config = config or ModelConfig()
        self.key = jax.random.PRNGKey(seed)

        xlstm_config = xLSTMTabModelConfig(
            embedding_dim=self.config.embedding_dim,
            num_blocks=self.config.num_blocks,
            context_length=self.config.context_length,
            tie_weights=False,
            add_embedding_dropout=False,
            add_post_blocks_norm=True,
            dtype="bfloat16",
            mlstm_block=xLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,
                    qkv_proj_blocksize=4,
                    num_heads=self.config.num_heads,
                    proj_factor=2.0,
                    embedding_dim=self.config.embedding_dim,
                    bias=True,
                    dropout=self.config.dropout,
                    context_length=self.config.context_length,
                    dtype="bfloat16",
                ),
                _num_blocks=1,
                _block_idx=0,
                feedforward=FeedForwardConfig(
                    proj_factor=4.0,
                    embedding_dim=self.config.embedding_dim,
                    dropout=self.config.dropout,
                    dtype="bfloat16",
                ),
            ),
        )
        
        self.model = xLSTMTabModel(config=xlstm_config)

        # Initialize model with proper input dimension
        rng_init = self._next_rng()
        dummy_input = jnp.ones((1, self.config.context_length, self.config.input_dim))
        params = self.model.init({"params": rng_init, "dropout": rng_init}, dummy_input)

        # Optimizer
        self.optimizer = optax.adamw(
            learning_rate=self.config.learning_rate, 
            weight_decay=self.config.weight_decay
        )

        # Model state
        self.model_state = ModelState(
            params=params,
            opt_state=self.optimizer.init(params),
            step=0,
        )

        # Build JIT-compiled functions
        self._build_train_step()
        
        # Training history
        self.losses = []

    def _build_train_step(self):
        """Build JIT-compiled training step."""
        
        def loss_fn(params, rng, x, y, mask):
            preds = self.model.apply(params, x, train=True, rngs={"dropout": rng})
            sq_err = optax.losses.squared_error(preds, y)  # (B, T, D)
            # Expand mask for multivariate: (B, T) -> (B, T, 1)
            mask_expanded = mask[:, :, None]
            return jnp.sum(sq_err * mask_expanded) / (jnp.sum(mask_expanded) + 1e-8)

        def _train_step(state, x, y, mask, rng):
            loss, grads = jax.value_and_grad(loss_fn)(state.params, rng, x, y, mask)
            updates, opt_state = self.optimizer.update(grads, state.opt_state, state.params)
            params = optax.apply_updates(state.params, updates)
            new_state = state.replace(params=params, opt_state=opt_state, step=state.step + 1)
            return new_state, loss

        self._train_step = jax.jit(_train_step)

    def train(self, dataset, num_steps: int, batch_size: int = 32, log_every: int = 100):
        """Train the model on a ChunkedDataset."""
        self.losses = []
        
        for step in range(num_steps):
            x, y, mask = dataset.sample_batch(batch_size)
            x, y, mask = jnp.array(x), jnp.array(y), jnp.array(mask)
            
            self.model_state, loss = self._train_step(
                self.model_state, x, y, mask, self._next_rng()
            )
            self.losses.append(float(loss))
            
            if step % log_every == 0:
                print(f"Step {step:4d} | Loss: {loss:.6f}")
        
        print(f"Training complete. Final loss: {self.losses[-1]:.6f}")
        return self.losses

    def predict(self, x: Array) -> Array:
        """One-step ahead prediction."""
        if x.ndim == 2:
            x = x[:, :, None]
        return self.model.apply(self.model_state.params, x, train=False)

    def generate(self, context: Array, n_steps: int = None) -> Array:
        """Autoregressive generation from context.
        
        Args:
            context: Initial context of shape (B, S, D) or (B, S)
            n_steps: Number of steps to generate (default: config.max_new_steps)
        
        Returns:
            Full sequence including context and generated steps: (B, S + n_steps, D)
        """
        n_steps = n_steps or self.config.max_new_steps
        
        # Ensure 3D input
        if context.ndim == 2:
            context = context[:, :, None]
        
        B, S, D = context.shape
        buffer = jnp.concatenate([context, jnp.zeros((B, n_steps, D))], axis=1)

        for i in range(n_steps):
            ctx = buffer[:, i:i + S, :]
            preds = self.model.apply(self.model_state.params, ctx, train=False)
            next_val = preds[:, -1, :]  # (B, D)
            buffer = buffer.at[:, S + i, :].set(next_val)
        
        return np.array(buffer)

    def _next_rng(self):
        self.key, rng = jax.random.split(self.key)
        return rng

    def save(self, path: str):
        pickle.dump(self.model_state.params, open(path, "wb"))

    def load(self, path: str):
        params = pickle.load(open(path, "rb"))
        self.model_state = self.model_state.replace(params=params)
