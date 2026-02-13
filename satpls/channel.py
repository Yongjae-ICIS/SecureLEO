"""Shadowed-Rician channel model for satellite communications.

Reference:
    A. Abdi et al., "A new simple model for land mobile satellite channels,"
    IEEE Trans. Wireless Commun., 2003.
"""
from __future__ import annotations

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray


def generate_channel_matrix(
    num_rx: int,
    num_tx: int,
    rician_k: float = 3.0,
    nakagami_m: float = 5.0,
    rng: Generator | None = None,
) -> NDArray[np.complexfloating]:
    """Generate Shadowed-Rician fading channel matrix.

    h = xi * h_LOS + h_NLOS where xi ~ Nakagami-m shadowing.

    Returns:
        Complex channel matrix of shape (num_rx, num_tx)
    """
    if rng is None:
        rng = np.random.default_rng()

    shape = (num_rx, num_tx)

    phi = rng.uniform(0.0, 2 * np.pi, size=shape)
    k_ratio = np.sqrt(rician_k / (1 + rician_k))
    p = k_ratio * np.cos(phi)
    q = k_ratio * np.sin(phi)

    sigma = 1.0 / np.sqrt(2 * (1 + rician_k))
    xi = np.sqrt(rng.gamma(nakagami_m, 1.0 / nakagami_m, size=shape))

    x_real = rng.standard_normal(size=shape) * sigma + xi * p
    x_imag = rng.standard_normal(size=shape) * sigma + xi * q

    return x_real + 1j * x_imag


def create_eve_sampler(
    num_rx: int,
    num_tx: int,
    rician_k: float = 3.0,
    nakagami_m: float = 5.0,
    rng: Generator | None = None,
) -> callable:
    """Create a channel generator callable for Eve MC sampling."""
    if rng is None:
        rng = np.random.default_rng()

    def _generate():
        return generate_channel_matrix(num_rx, num_tx, rician_k, nakagami_m, rng)

    return _generate
