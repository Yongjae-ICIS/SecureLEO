"""Satellite scheduling algorithms: brute-force optimal and random baseline."""
from __future__ import annotations

import itertools
from typing import Callable

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from satpls.signal import compute_secrecy_rate


def get_an_union_indices(
    data_indices: list[int],
    an_only_indices: list[int],
) -> list[int]:
    """Union of data + AN-only satellites for cooperative AN generation."""
    return sorted(set(data_indices) | set(an_only_indices))


def _get_an_satellites(
    data_indices: list[int],
    num_total_sats: int,
    num_sat_ant: int,
    num_gbs_ant: int,
    num_eve_ant: int,
    policy: str,
) -> list[int]:
    """Determine AN satellite set based on policy ('min_sat' or 'max_all')."""
    if policy.lower() in ("max", "max_all", "all"):
        return list(range(num_total_sats))

    an_only_needed = max(
        0,
        (num_gbs_ant + num_eve_ant - len(data_indices) * num_sat_ant) // num_sat_ant,
    )
    remaining = [i for i in range(num_total_sats) if i not in data_indices]
    an_only = remaining[:an_only_needed]
    return get_an_union_indices(data_indices, an_only)


def optimal_scheduling_statistical(
    channel_gbs: NDArray,
    num_data_sats: int,
    eve_sampler: Callable[[], NDArray],
    num_gbs_ant: int,
    num_eve_ant: int,
    num_sat_ant: int,
    an_power_ratio: float,
    snr_gbs_db: float,
    snr_eve_db: float,
    num_mc: int = 100,
    an_policy: str = "min_sat",
    eve_receiver: str = "mmse",
) -> tuple[NDArray[np.floating], float]:
    """Brute-force optimal scheduling using statistical (ergodic) Eve CSI.

    Returns:
        (binary selection mask shape (N,), best secrecy rate)
    """
    num_total = channel_gbs.shape[1] // num_sat_ant
    best_rate = -1.0
    best_mask = None

    for combo in itertools.combinations(range(num_total), num_data_sats):
        data_idx = list(combo)
        an_idx = _get_an_satellites(
            data_idx, num_total, num_sat_ant, num_gbs_ant, num_eve_ant, an_policy,
        )

        _, _, sec_rate = compute_secrecy_rate(
            channel_gbs, eve_sampler, data_idx, an_idx,
            an_power_ratio, snr_gbs_db, snr_eve_db,
            num_gbs_ant, num_eve_ant, num_sat_ant,
            num_mc, eve_receiver,
        )

        if sec_rate > best_rate:
            best_rate = sec_rate
            best_mask = np.zeros(num_total, dtype=np.float32)
            best_mask[data_idx] = 1.0

    return best_mask, float(best_rate)


def optimal_scheduling_oracle(
    channel_gbs: NDArray,
    channel_eve: NDArray,
    num_data_sats: int,
    num_gbs_ant: int,
    num_eve_ant: int,
    num_sat_ant: int,
    an_power_ratio: float,
    snr_gbs_db: float,
    snr_eve_db: float,
    an_policy: str = "min_sat",
    eve_receiver: str = "mmse",
) -> tuple[NDArray[np.floating], float]:
    """Brute-force optimal scheduling with instantaneous Eve CSI (oracle upper bound)."""
    return optimal_scheduling_statistical(
        channel_gbs, num_data_sats,
        lambda: channel_eve,
        num_gbs_ant, num_eve_ant, num_sat_ant,
        an_power_ratio, snr_gbs_db, snr_eve_db,
        num_mc=1, an_policy=an_policy, eve_receiver=eve_receiver,
    )


def random_scheduling(
    num_total_sats: int,
    num_data_sats: int,
    rng: Generator | None = None,
) -> NDArray[np.floating]:
    """Random satellite selection baseline.

    Returns:
        Binary selection mask shape (num_total_sats,)
    """
    if rng is None:
        rng = np.random.default_rng()
    selected = rng.choice(num_total_sats, size=num_data_sats, replace=False)
    mask = np.zeros(num_total_sats, dtype=np.float32)
    mask[selected] = 1.0
    return mask
