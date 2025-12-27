import jax
import jax.numpy as jnp
import optax
import sys
sys.path.append('/Users/mariana/Documents/research/xlstm-jax')

from dataclasses import dataclass
from flax import linen as nn

from xlstm_jax.models.xlstm_clean.components.init import small_init
from xlstm_jax.models.xlstm_clean.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class xLSTMTabModelConfig(xLSTMBlockStackConfig):
    embedding_dim: int = 16
    tie_weights: bool = False
    weight_decay_on_embedding: bool = False
    add_embedding_dropout: bool = True
    output_dim: int = 1


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
            x = nn.Dropout(rate=self.config.dropout)(x, deterministic=not train)
        x = xLSTMBlockStack(config=self.config, name="xlstm_block_stack")(x, train=train)
        logits = nn.Dense(
            features=self.config.output_dim,
            kernel_init=small_init(self.config.embedding_dim),
            use_bias=False,
            dtype=jnp.float32,
            name="pred_head",
        )(x)
        return logits


