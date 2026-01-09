import sys
sys.path.append('/Users/mariana/Documents/research/xlstm-jax')
import matplotlib.pyplot as plt
import numpy as np
from xlstm_jax.models.xlstm_clean.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig
from xlstm_jax.models.xlstm_clean.components.init import small_init
from flax import linen as nn
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
import optax



@dataclass
class xLSTMTabModelConfig(xLSTMBlockStackConfig):
    embedding_dim: int = 16
    tie_weights: bool = False
    weight_decay_on_embedding: bool = False
    add_embedding_dropout: bool = True
    output_dim: int = 1
    quantiles: list[float] = field(default_factory=lambda: [
                                   0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


class xLSTMWeavingBlock(nn.Module):
    pass


class xLSTMTabModel(nn.Module):
    config: xLSTMTabModelConfig

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
        x = nn.Dense(
            features=self.config.embedding_dim,
            kernel_init=small_init(self.config.embedding_dim),
            dtype=self.config._dtype,
            name="token_embedding",
        )(x)
        pos_emb = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (self.config.context_length, self.config.embedding_dim),
        )
        x = x + pos_emb[:x.shape[1]]
        if self.config.add_embedding_dropout:
            x = nn.Dropout(rate=self.config.dropout)(
                x, deterministic=not train)
        x = xLSTMBlockStack(config=self.config,
                            name="xlstm_block_stack")(x, train=train)
        logits = nn.Dense(
            features=self.config.output_dim,
            kernel_init=small_init(self.config.embedding_dim),
            use_bias=False,
            dtype=jnp.float32,
            name="pred_head",
        )(x)
        return logits


class xLSTMTabModelWithPos(nn.Module):
    config: xLSTMTabModelConfig

    def setup(self):
        self.pos_encoder = nn.Dense(self.config.embedding_dim)
        # channel-wise, for now 1 to embedding_dim
        self.hist_encoder = nn.Dense(self.config.embedding_dim)
        # self.weaving_block = xLSTMWeavingBlock()
        self.weaving_block = xLSTMBlockStack(config=self.config,
                                             name="xlstm_block_stack")
        self.output_proj = nn.Dense(len(self.config.quantiles))

    def __call__(self,
                 x_hist: jax.Array,
                 t_hist: jax.Array,
                 t_future: jax.Array,
                 train: bool = False) -> jax.Array:

        # history pos embedding
        h_pos = self.pos_encoder(t_hist)  # (B, h_len, embedding_dim)
        # (B, h_len, 1, embedding_dim) -- we add a channel dimension
        h_pos = h_pos[:, :, None, :]

        # future pos embedding
        f_pos = self.pos_encoder(t_future)  # (B, f_len, embedding_dim)
        f_pos = f_pos[:, :, None, :]  # (B, f_len, 1, embedding_dim)

        # embedd history
        # x_hist is (B, h_len, n_channels, 1) -- n_channels = 1 for now
        # (B, h_len, n_channels, embedding_dim)  -- n_channels = 1 for now
        h_emb = self.hist_encoder(x_hist)
        h_emb = h_emb + h_pos

        # generate predictions with input given by: embedded history + history pos encoding + future pos encoding
        # reshape to (B * n_channels, h_len, embedding_dim)
        B, h_len, n_channels, embedding_dim = h_emb.shape
        _, f_len, _, _ = f_pos.shape
        h_emb = h_emb.reshape(B * n_channels, h_len, embedding_dim)
        f_pos = f_pos.reshape(B * n_channels, f_len, embedding_dim)
        inputs = jnp.concatenate([h_emb, f_pos], axis=1)
        preds = self.weaving_block(inputs, train=train)
        preds = preds[:, -f_len:, :]
        preds = self.output_proj(preds)
        output_dim = len(self.config.quantiles)
        preds = preds.reshape(B, f_len, n_channels, output_dim)
        return preds
