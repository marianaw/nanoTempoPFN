"""Time series data generators for model training and evaluation."""

import numpy as np
from typing import Callable


def make_sinusoidal_generator(
    freq_range: tuple[float, float] = (0.05, 0.2),
    amp_range: tuple[float, float] = (0.5, 2.0),
    phase_range: tuple[float, float] = (0, 2 * np.pi),
    noise_std: float = 0.0,
    rng: np.random.Generator = None,
) -> Callable:
    """Generate sinusoidal sequences with random frequency, amplitude, phase.
    
    x_t = A * sin(2π * f * t + φ) + noise
    """
    rng = rng or np.random.default_rng()
    
    def gen(n_samples: int, horizon: int):
        seqs = np.zeros((n_samples, horizon + 1, 2))  # [value, mask]
        ts = np.full(n_samples, horizon + 1, dtype=int)
        cs = np.ones(n_samples, dtype=bool)
        
        t = np.arange(horizon + 1)
        for i in range(n_samples):
            freq = rng.uniform(*freq_range)
            amp = rng.uniform(*amp_range)
            phase = rng.uniform(*phase_range)
            seqs[i, :, 0] = amp * np.sin(2 * np.pi * freq * t + phase)
            if noise_std > 0:
                seqs[i, :, 0] += noise_std * rng.normal(size=horizon + 1)
            seqs[i, :, 1] = 1  # mask
        
        return seqs.astype(np.float32), ts, cs
    return gen


def make_ar1_generator(
    phi: float = 0.9,
    x0_scale: float = 2.0,
    noise_std: float = 0.0,
    rng: np.random.Generator = None,
) -> Callable:
    """Generate AR(1) sequences: x_t = φ * x_{t-1} + noise.
    
    Args:
        phi: AR coefficient, |phi| < 1 for stationarity
        x0_scale: Scale for random initial values
        noise_std: Standard deviation of additive noise (0 = deterministic)
    """
    rng = rng or np.random.default_rng()
    
    def gen(n_samples: int, horizon: int):
        seqs = np.zeros((n_samples, horizon + 1, 2))
        ts = np.full(n_samples, horizon + 1, dtype=int)
        cs = np.ones(n_samples, dtype=bool)
        
        for i in range(n_samples):
            seqs[i, 0, 0] = x0_scale * rng.normal()
            seqs[i, 0, 1] = 1
            for j in range(1, horizon + 1):
                seqs[i, j, 0] = phi * seqs[i, j-1, 0]
                if noise_std > 0:
                    seqs[i, j, 0] += noise_std * rng.normal()
                seqs[i, j, 1] = 1
        
        return seqs.astype(np.float32), ts, cs
    return gen


def make_linear_trend_generator(
    slope_range: tuple[float, float] = (-0.1, 0.1),
    intercept_range: tuple[float, float] = (-2.0, 2.0),
    noise_std: float = 0.0,
    rng: np.random.Generator = None,
) -> Callable:
    """Generate linear trend sequences: x_t = a * t + b + noise."""
    rng = rng or np.random.default_rng()
    
    def gen(n_samples: int, horizon: int):
        seqs = np.zeros((n_samples, horizon + 1, 2))
        ts = np.full(n_samples, horizon + 1, dtype=int)
        cs = np.ones(n_samples, dtype=bool)
        
        t = np.arange(horizon + 1)
        for i in range(n_samples):
            slope = rng.uniform(*slope_range)
            intercept = rng.uniform(*intercept_range)
            seqs[i, :, 0] = slope * t + intercept
            if noise_std > 0:
                seqs[i, :, 0] += noise_std * rng.normal(size=horizon + 1)
            seqs[i, :, 1] = 1
        
        return seqs.astype(np.float32), ts, cs
    return gen


def make_random_walk_generator(
    step_std: float = 0.5,
    x0_scale: float = 1.0,
    rng: np.random.Generator = None,
) -> Callable:
    """Generate random walk: x_t = x_{t-1} + noise."""
    rng = rng or np.random.default_rng()
    
    def gen(n_samples: int, horizon: int):
        seqs = np.zeros((n_samples, horizon + 1, 2))
        ts = np.full(n_samples, horizon + 1, dtype=int)
        cs = np.ones(n_samples, dtype=bool)
        
        for i in range(n_samples):
            seqs[i, 0, 0] = x0_scale * rng.normal()
            seqs[i, 0, 1] = 1
            for j in range(1, horizon + 1):
                seqs[i, j, 0] = seqs[i, j-1, 0] + step_std * rng.normal()
                seqs[i, j, 1] = 1
        
        return seqs.astype(np.float32), ts, cs
    return gen


def make_mixed_generator(
    generators: list[Callable],
    weights: list[float] = None,
    rng: np.random.Generator = None,
) -> Callable:
    """Mix multiple generators randomly."""
    rng = rng or np.random.default_rng()
    weights = weights or [1.0 / len(generators)] * len(generators)
    weights = np.array(weights) / sum(weights)
    
    def gen(n_samples: int, horizon: int):
        seqs = np.zeros((n_samples, horizon + 1, 2))
        ts = np.full(n_samples, horizon + 1, dtype=int)
        cs = np.ones(n_samples, dtype=bool)
        
        choices = rng.choice(len(generators), size=n_samples, p=weights)
        for gen_idx in range(len(generators)):
            mask = choices == gen_idx
            n = mask.sum()
            if n > 0:
                gen_seqs, _, _ = generators[gen_idx](n, horizon)
                seqs[mask] = gen_seqs
        
        return seqs.astype(np.float32), ts, cs
    return gen


# Convenience dict for quick access
GENERATORS = {
    "sinusoidal": make_sinusoidal_generator,
    "ar1": make_ar1_generator,
    "ar1_noisy": lambda **kw: make_ar1_generator(noise_std=0.1, **kw),
    "linear": make_linear_trend_generator,
    "random_walk": make_random_walk_generator,
}
