from flax import linen as nn
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import chex

import sys
sys.path.append('/Users/mariana/Documents/research/xlstm-jax')
from xlstm_jax.models.xlstm_clean.blocks.mlstm.layer import mLSTMLayerConfig
from xlstm_jax.models.xlstm_clean.components.init import small_init, wang_init
from xlstm_jax.models.xlstm_clean.components.conv import CausalConv1d, CausalConv1dConfig
from xlstm_jax.models.xlstm_clean.components.linear_headwise import (
    LinearHeadwiseExpand,
    LinearHeadwiseExpandConfig,
)
from recurrent_lstm_cell import mLSTMWeavingCell, mLSTMWeavingCellConfig


class mLSTMWeavingLayer(nn.Module):
    config: mLSTMLayerConfig

    @nn.compact
    def __call__(self, x: chex.Array, train: bool = False) -> chex.Array:
        
        # x is of shape (B, S, D)
        hh = nn.Dense(
            features=2 * self.config._inner_embedding_dim,
            dtype=self.config._dtype,
            kernel_init=small_init(x.shape[-1]),
            use_bias=self.config.bias,
            name="proj_up",
        )(x)
        x_lstm, z = jnp.split(hh, 2, axis=-1)  # (B, S, inner_dim) * 2

        # mlstm branch
        x_mlstm_conv = CausalConv1d(
            config=CausalConv1dConfig(
                feature_dim=self.config._inner_embedding_dim,
                kernel_size=self.config.conv1d_kernel_size,
                dtype=self.config.dtype,
            ),
            name="conv1d",
        )(x_lstm)
        x_mlstm_conv_act = nn.swish(x_mlstm_conv)

        # inside the layer we handle the heads, which have a per-head dimension.
        num_proj_heads = round(self.config._inner_embedding_dim // self.config.qkv_proj_blocksize)

        # We get q k and v
        q = LinearHeadwiseExpand(
                config=LinearHeadwiseExpandConfig(
                    in_features=self.config._inner_embedding_dim,
                    num_heads=num_proj_heads,
                    bias=self.config.bias,
                    dtype=self.config.dtype,
                ),
                kernel_init=small_init(self.config.embedding_dim),
                name="q_proj",
            )(x_mlstm_conv_act)
        k = LinearHeadwiseExpand(
                config=LinearHeadwiseExpandConfig(
                    in_features=self.config._inner_embedding_dim,
                    num_heads=num_proj_heads,
                    bias=self.config.bias,
                    dtype=self.config.dtype,
                ),
                kernel_init=small_init(self.config.embedding_dim),
                name="k_proj",
            )(x_mlstm_conv_act)
        v = LinearHeadwiseExpand(
                config=LinearHeadwiseExpandConfig(
                    in_features=self.config._inner_embedding_dim,
                    num_heads=num_proj_heads,
                    bias=self.config.bias,
                    dtype=self.config.dtype,
                ),
                kernel_init=small_init(self.config.embedding_dim),
                name="v_proj",
            )(x_lstm)

        h_tilde_state, (c_state, n_state, m_state) = mLSTMWeavingCell(config=self.config.mlstm_cell, name="mlstm_cell")(q=q, k=k, v=v)
        learnable_skip = self.param("learnable_skip", nn.initializers.ones, (x_mlstm_conv_act.shape[-1],))
        learnable_skip = jnp.broadcast_to(learnable_skip, x_mlstm_conv_act.shape)
        h_tilde_state_skip = h_tilde_state + (learnable_skip * x_mlstm_conv_act)

        # output / z branch
        h_state = h_tilde_state_skip * nn.swish(z)

        # down-projection
        y = nn.Dense(
            features=self.config.embedding_dim,
            dtype=self.config._dtype,
            kernel_init=wang_init(x.shape[-1], num_blocks=self.config._num_blocks),
            use_bias=self.config.bias,
            name="proj_down",
        )(h_state)
        y = nn.Dropout(rate=self.config.dropout, deterministic=not train)(y)
        return y, (c_state, n_state, m_state)



        
        
