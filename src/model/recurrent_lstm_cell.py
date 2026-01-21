import jax
import jax.numpy as jnp
import math
from flax import linen as nn
import jax
import jax.numpy as jnp
from dataclasses import dataclass

import sys
sys.path.append('/Users/mariana/Documents/research/xlstm-jax')
from xlstm_jax.models.xlstm_clean.components.init import bias_linspace_init
from xlstm_jax.models.xlstm_clean.components.ln import MultiHeadLayerNorm
from typing import Tuple


def weaving_recurrent_lstm(q: jax.Array, k: jax.Array, v: jax.Array, igate_preact: jax.Array, fgate_preact: jax.Array, 
                           c_state: jax.Array, n_state: jax.Array, m_state: jax.Array, eps: float = 1e-6):

    B, NH, S, DH = q.shape

    # Initialize the carry
    # c_state = jnp.zeros((B, NH, DH, DH))
    # n_state = jnp.zeros((B, NH, DH, 1))
    # m_state = jnp.zeros((B, NH, 1, 1))

    def recurrent_step(carry, xs):
        c_state, n_state, m_state = carry
        q, k, v, fgate_preact, igate_preact = xs

        # Add dimension and transpose accordingly
        q = jnp.expand_dims(q, axis=2)  # (B, NH, 1, DH)
        k = jnp.expand_dims(k, axis=-1)  # (B, NH, DH, 1)
        v = jnp.expand_dims(v, axis=2)  # (B, NH, 1, DH)

        # gates
        log_fg_act = jax.nn.log_sigmoid(fgate_preact)  # (B, NH, 1, 1)

        # update rule
        m_state_new = jnp.maximum(log_fg_act + m_state, igate_preact)  # (B, NH, 1, 1)

        fg_act = jnp.exp(log_fg_act + m_state - m_state_new)  # (B, NH, 1, 1)
        ig_act = jnp.exp(igate_preact - m_state_new)  # (B, NH, 1, 1)

        k_scaled = k / math.sqrt(DH)

        c_state_new = fg_act * c_state + ig_act * (k_scaled @ v)  # (B, NH, DH, DH)
        n_state_new = fg_act * n_state + ig_act * k_scaled  # (B, NH, DH, 1)
        
        h_num = q @ c_state_new  # (B, NH, 1, DH)

        qn_dotproduct = q @ n_state_new  # (B, NH, 1, 1)
        max_val = jnp.exp(-m_state_new)  # (B, NH, 1, 1)
        h_denom = jnp.maximum(jnp.abs(qn_dotproduct), max_val) + eps
        h = h_num / h_denom  # (B, NH, 1, DH) / (B, NH, 1, 1) = (B, NH, 1, DH)

        h = h.squeeze(axis=2)  # (B, NH, DH)
        carry = (c_state_new, n_state_new, m_state_new)

        return carry, h
    
    # q, k, v have to have shape (S, B, NH, DH)
    q = jnp.transpose(q, (2, 0, 1, 3))  # from (B, NH, S, DH) to (S, B, NH, DH)
    k = jnp.transpose(k, (2, 0, 1, 3))  # from (B, NH, S, DH) to (S, B, NH, DH)
    v = jnp.transpose(v, (2, 0, 1, 3))  # from (B, NH, S, DH) to (S, B, NH, DH)
    igate_preact = jnp.transpose(igate_preact, (2, 0, 1, 3))  # from (B, NH, S, 1) to (S, B, NH, 1)
    fgate_preact = jnp.transpose(fgate_preact, (2, 0, 1, 3))  # from (B, NH, S, 1) to (S, B, NH, 1)
    igate_preact = jnp.expand_dims(igate_preact, axis=-1)  # (S, B, NH, 1, 1)
    fgate_preact = jnp.expand_dims(fgate_preact, axis=-1)  # (S, B, NH, 1, 1)

    (c_state_new, n_state_new, m_state_new), out = jax.lax.scan(f=recurrent_step,
        init=(c_state, n_state, m_state),
        xs=(q, k, v, fgate_preact, igate_preact)
    )

    out = jnp.transpose(out, (1, 2, 0, 3))  # from (S, B, NH, DH) to (B, NH, S, DH)

    return out, (c_state_new, n_state_new, m_state_new)


class mLSTMWeavingBackend(nn.Module):

    @nn.compact
    def __call__(self, q: jax.Array, k: jax.Array, v: jax.Array, i:jax.Array, f:jax.Array,
                 c_state: jax.Array, n_state: jax.Array, m_state: jax.Array, eps:float=1e-6) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]:
        return weaving_recurrent_lstm(q, k, v, i, f, c_state, n_state, m_state, eps)

@dataclass
class mLSTMWeavingCellConfig:
    embedding_dim: int = -1
    num_heads: int = -1
    dtype: str = "bfloat16"

    @property
    def _dtype(self) -> jnp.dtype:
        """
        Returns the real dtype instead of the str from configs.

        Returns:
            The jnp dtype corresponding to the string value.
        """
        return getattr(jnp, self.dtype)


class mLSTMWeavingCell(nn.Module):
    config: mLSTMWeavingCellConfig

    @nn.compact
    def __call__(self, q: jax.Array, k: jax.Array, v: jax.Array, c_state: jax.Array, n_state: jax.Array, m_state: jax.Array, **kwargs):
        B, S, _ = q.shape
        qkv = jnp.concatenate([q, k, v], axis=-1)  # (B, NH, S, 3*DH)

        # compute input and forget gate pre-activations  - why taking all heads as input?
        igate_preact = nn.Dense(
            features=self.config.num_heads,
            dtype=self.config._dtype,
            bias_init=nn.initializers.normal(stddev=0.1),
            kernel_init=nn.initializers.zeros,
            name="igate",
        )(qkv)
        fgate_preact = nn.Dense(
            features=self.config.num_heads,
            dtype=self.config._dtype,
            bias_init=bias_linspace_init(3.0, 6.0),
            kernel_init=nn.initializers.zeros,
            name="fgate",
        )(qkv)

        q = q.reshape(B, self.config.num_heads, S, -1)  # (B, NH, S, DH)
        k = k.reshape(B, self.config.num_heads, S, -1)  # (B, NH, S, DH)
        v = v.reshape(B, self.config.num_heads, S, -1)  # (B, NH, S, DH)

        igate_preact = igate_preact.transpose(0, 2, 1)[..., None]  # (B, NH, S, 1)
        fgate_preact = fgate_preact.transpose(0, 2, 1)[..., None]  # (B, NH, S, 1)

        backend_fn = mLSTMWeavingBackend()
        h_state, (c_state, n_state, m_state) = backend_fn(q, k, v, igate_preact, fgate_preact, c_state, n_state, m_state)
        # h_state is of shape (B, NH, S, DH)
        
        h_state_norm = MultiHeadLayerNorm(weight=True, bias=False, dtype=self.config._dtype, name="outnorm")(h_state)
        h_state_norm = h_state_norm.transpose(0, 2, 1, 3).reshape(B, S, -1)  # (B, S, NH*DH)
        return h_state_norm, (c_state, n_state, m_state)