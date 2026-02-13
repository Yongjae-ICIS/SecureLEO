"""Signal processing: precoder, AN, SINR, and secrecy rate computation.

Combines precoder design, null-space AN generation, ZF/MMSE SINR,
and secrecy rate calculation into a single module.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Log-determinant
# ---------------------------------------------------------------------------

def compute_log_det(matrix: NDArray[np.complexfloating]) -> float:
    """Compute log2(det(I + A)) for capacity computation."""
    identity = np.eye(matrix.shape[0], dtype=matrix.dtype)
    sign, logdet = np.linalg.slogdet(identity + matrix)
    if sign <= 0:
        return 0.0
    return float(logdet / np.log(2.0))


# ---------------------------------------------------------------------------
# Data precoder (ZF-style per-satellite beamforming)
# ---------------------------------------------------------------------------

def compute_data_precoder(
    channel_gbs: NDArray,
    data_indices: list[int],
    num_sat_ant: int = 1,
    power_amplitude: float = 1.0,
) -> NDArray[np.complexfloating]:
    """Build data precoding matrix W for selected data satellites.

    Returns:
        Precoder matrix of shape (N*M_s, K_d)
    """
    num_total_ant = channel_gbs.shape[1]
    num_data = len(data_indices)
    if num_data == 0:
        return np.zeros((num_total_ant, 0), dtype=np.complex128)

    precoder = np.zeros((num_total_ant, num_data), dtype=np.complex128)

    for stream_idx, sat_idx in enumerate(data_indices):
        col_s = sat_idx * num_sat_ant
        col_e = col_s + num_sat_ant
        block = channel_gbs[:, col_s:col_e]

        if num_sat_ant == 1:
            local_beam = np.ones((1, 1), dtype=np.complex128)
        else:
            _, _, vh = np.linalg.svd(block, full_matrices=False)
            local_beam = vh.conj().T[:, [0]]

        norm = np.linalg.norm(local_beam)
        if norm > 1e-12:
            local_beam = local_beam / norm
        precoder[col_s:col_e, stream_idx : stream_idx + 1] = local_beam

    return precoder * (power_amplitude / np.sqrt(num_data))


# ---------------------------------------------------------------------------
# Artificial Noise (null-space beamforming)
# ---------------------------------------------------------------------------

def _design_an_beamformer(
    channel_an: NDArray,
    num_eve_ant: int,
    cap_by_eve: bool = True,
) -> tuple[NDArray, int]:
    """Design AN beams in Null(H_gbs) for AN satellites."""
    _, sv, vh = np.linalg.svd(channel_an, full_matrices=True)
    v = vh.conj().T

    rank = int(np.sum(sv > 1e-8))
    null_dim = max(0, channel_an.shape[1] - rank)

    if null_dim > 0:
        z_basis = v[:, -null_dim:].copy()
        num_streams = min(null_dim, int(num_eve_ant)) if cap_by_eve else null_dim
        z = z_basis[:, :num_streams]
        for col in range(z.shape[1]):
            n = np.linalg.norm(z[:, col])
            if n > 1e-12:
                z[:, col] /= n
    else:
        num_an = channel_an.shape[1]
        z = np.ones((num_an, 1), dtype=np.complex128) / np.sqrt(num_an)
        null_dim = 0

    return z, int(null_dim)


def compute_an_matrix(
    channel_gbs: NDArray,
    an_indices: list[int],
    num_sat_ant: int,
    num_eve_ant: int,
    power_amplitude: float = 1.0,
) -> tuple[NDArray, int]:
    """Build full AN transmit matrix for selected AN satellites.

    Returns:
        Tuple of (AN matrix shape (N*M_s, L_an), number of AN streams)
    """
    num_total_ant = channel_gbs.shape[1]
    num_total_sats = num_total_ant // num_sat_ant

    if len(an_indices) == 0:
        z = np.ones((num_total_ant, 1), dtype=np.complex128) / np.sqrt(num_total_ant)
        return z, 1

    an_cols = np.concatenate([
        np.arange(idx * num_sat_ant, (idx + 1) * num_sat_ant)
        for idx in an_indices
    ])
    channel_an = channel_gbs[:, an_cols]
    is_all = len(an_indices) == num_total_sats

    z_local, _ = _design_an_beamformer(channel_an, num_eve_ant, cap_by_eve=(not is_all))

    if z_local is not None and z_local.size > 0:
        num_streams = z_local.shape[1]
        an_full = np.zeros((num_total_ant, num_streams), dtype=np.complex128)
        an_full[an_cols, :] = z_local * power_amplitude
        return an_full, num_streams

    z = np.ones((num_total_ant, 1), dtype=np.complex128) / np.sqrt(num_total_ant)
    return z, 1


# ---------------------------------------------------------------------------
# SINR computation: ZF and MMSE receivers
# ---------------------------------------------------------------------------

def _zf_equalizer(h_eq: NDArray, eps: float = 1e-8) -> NDArray:
    """W_zf = (H^H H + eps I)^{-1} H^H"""
    n = h_eq.shape[1]
    if n == 0:
        return np.zeros((0, h_eq.shape[0]), dtype=np.complex128)
    gram = h_eq.conj().T @ h_eq
    try:
        gram_inv = np.linalg.inv(gram + eps * np.eye(n, dtype=gram.dtype))
    except np.linalg.LinAlgError:
        gram_inv = np.linalg.pinv(gram)
    return gram_inv @ h_eq.conj().T


def _mmse_equalizer(h_eq: NDArray, reg: float, eps: float = 1e-8) -> NDArray:
    """W_mmse = (H^H H + lambda I)^{-1} H^H"""
    n = h_eq.shape[1]
    if n == 0:
        return np.zeros((0, h_eq.shape[0]), dtype=np.complex128)
    gram = h_eq.conj().T @ h_eq
    try:
        gram_inv = np.linalg.inv(gram + reg * np.eye(n, dtype=gram.dtype))
    except np.linalg.LinAlgError:
        gram_inv = np.linalg.pinv(gram + reg * np.eye(n, dtype=gram.dtype))
    return gram_inv @ h_eq.conj().T


def compute_sinr_zf(
    channel: NDArray,
    precoder: NDArray,
    an_matrix: NDArray | None,
    noise_var: float,
) -> list[float]:
    """Per-stream SINR under Zero-Forcing receiver."""
    h_eq = channel @ precoder
    num_streams = h_eq.shape[1]
    if num_streams == 0:
        return []

    w_zf = _zf_equalizer(h_eq)
    sinrs = []

    if an_matrix is None or an_matrix.size == 0:
        for i in range(num_streams):
            w_i = w_zf[i:i+1, :]
            row_norm_sq = max(float(np.sum(np.abs(w_i) ** 2).real), 1e-12)
            sinrs.append(max(0.0, 1.0 / (noise_var * row_norm_sq)))
    else:
        h_an = channel @ an_matrix
        r_int = h_an @ h_an.conj().T
        for i in range(num_streams):
            w_i = w_zf[i:i+1, :]
            var_noise = noise_var * float(np.sum(np.abs(w_i) ** 2).real)
            var_an = float((w_i @ r_int @ w_i.conj().T).item().real)
            sinrs.append(max(0.0, 1.0 / max(1e-12, var_noise + var_an)))

    return sinrs


def compute_sinr_mmse(
    channel: NDArray,
    precoder: NDArray,
    an_matrix: NDArray | None,
    noise_var: float,
    reg: float | None = None,
) -> list[float]:
    """Per-stream SINR under MMSE receiver."""
    h_eq = channel @ precoder
    num_streams = h_eq.shape[1]
    if num_streams == 0:
        return []

    if reg is None:
        reg = noise_var
    w_mmse = _mmse_equalizer(h_eq, reg)

    if an_matrix is not None and an_matrix.size > 0:
        h_an = channel @ an_matrix
        r_int = h_an @ h_an.conj().T
    else:
        r_int = None

    sinrs = []
    for i in range(num_streams):
        w_i = w_mmse[i:i+1, :]
        h_i = h_eq[:, [i]]
        sig = float(np.abs((w_i @ h_i).item()) ** 2)
        noise = noise_var * float(np.sum(np.abs(w_i) ** 2).real)
        an_pw = float((w_i @ r_int @ w_i.conj().T).item().real) if r_int is not None else 0.0
        sinrs.append(max(0.0, sig / max(1e-12, noise + an_pw)))

    return sinrs


# ---------------------------------------------------------------------------
# Secrecy rate
# ---------------------------------------------------------------------------

def compute_secrecy_rate(
    channel_gbs: NDArray,
    eve_sampler: Callable[[], NDArray],
    data_indices: list[int],
    an_indices: list[int],
    an_power_ratio: float,
    snr_gbs_db: float,
    snr_eve_db: float,
    num_gbs_ant: int,
    num_eve_ant: int,
    num_sat_ant: int = 1,
    num_mc: int = 1,
    eve_receiver: str = "mmse",
) -> tuple[float, float, float]:
    """Compute secrecy rate = [R_gbs - R_eve]^+.

    Returns:
        (R_gbs, R_eve_ergodic, R_sec_ergodic)
    """
    phi = float(np.clip(an_power_ratio, 0.0, 1.0))
    data_amp = np.sqrt(max(1e-12, 1.0 - phi))
    an_amp = np.sqrt(max(1e-12, phi))

    precoder = compute_data_precoder(channel_gbs, data_indices, num_sat_ant, data_amp)
    an_matrix, num_an_streams = compute_an_matrix(
        channel_gbs, an_indices, num_sat_ant, num_eve_ant, an_amp,
    )

    num_data = precoder.shape[1]
    if num_data == 0:
        return 0.0, 0.0, 0.0

    snr_gbs = 10 ** (snr_gbs_db / 10.0)
    snr_eve = 10 ** (snr_eve_db / 10.0)
    noise_gbs = 1.0 / snr_gbs
    noise_eve = 1.0 / snr_eve

    # GBS: ZF receiver (AN in null space -> no interference)
    sinrs_gbs = compute_sinr_zf(channel_gbs, precoder, None, noise_gbs)
    rates_gbs = [float(np.log2(1.0 + max(0.0, s))) for s in sinrs_gbs]
    r_gbs = float(np.sum(rates_gbs))

    # Eve: MC sampling for ergodic rate
    r_sec_samples = []
    for _ in range(max(1, num_mc)):
        channel_eve = eve_sampler()

        if eve_receiver.lower() == "mmse":
            reg = (float(num_eve_ant) * noise_eve) / max(1e-12, 1.0 - phi)
            sinrs_eve = compute_sinr_mmse(channel_eve, precoder, an_matrix, noise_eve, reg)
        else:
            sinrs_eve = compute_sinr_zf(channel_eve, precoder, an_matrix, noise_eve)

        rates_eve = [float(np.log2(1.0 + max(0.0, s))) for s in sinrs_eve]
        r_sec = sum(max(0.0, rg - re) for rg, re in zip(rates_gbs, rates_eve))
        r_sec_samples.append(r_sec)

    r_eve_ergodic = r_gbs - float(np.mean(r_sec_samples)) if r_gbs > 0 else 0.0
    r_sec_ergodic = float(np.mean(r_sec_samples))

    return r_gbs, r_eve_ergodic, r_sec_ergodic
