"""Benchmarking and visualization for scheduling algorithms."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from secureleo.config import SystemConfig
from secureleo.dataset import SatelliteEnvironment
from secureleo.scheduling import (
    get_an_union_indices,
    optimal_scheduling_oracle,
    optimal_scheduling_statistical,
    random_scheduling,
)
from secureleo.signal import compute_secrecy_rate

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    snr_values: list[float]
    rates_oracle: list[float]
    rates_statistical: list[float]
    rates_model: list[float]
    rates_random: list[float]


class Benchmarker:
    """Benchmark scheduler performance across SNR values."""

    def __init__(
        self,
        config: SystemConfig,
        model: torch.nn.Module | None = None,
        device: torch.device | None = None,
    ):
        self.config = config
        self.model = model
        self.device = device or torch.device("cpu")
        if model is not None:
            model.to(self.device)
            model.eval()

    def run(
        self,
        snr_values: list[float],
        num_trials: int = 1000,
        num_mc: int = 100,
        num_data_sats: int = 2,
        an_policy: str = "min_sat",
        eve_receiver: str = "mmse",
    ) -> BenchmarkResult:
        rates_o, rates_s, rates_m, rates_r = [], [], [], []

        for snr_db in tqdm(snr_values, desc="SNR sweep"):
            ro, rs, rm, rr = self._eval_snr(
                snr_db, num_trials, num_mc, num_data_sats, an_policy, eve_receiver,
            )
            rates_o.append(ro)
            rates_s.append(rs)
            rates_m.append(rm)
            rates_r.append(rr)

        return BenchmarkResult(snr_values, rates_o, rates_s, rates_m, rates_r)

    def _eval_snr(
        self, snr_db: float, num_trials: int, num_mc: int,
        num_data_sats: int, an_policy: str, eve_receiver: str,
    ) -> tuple[float, float, float, float]:
        c = self.config
        env = SatelliteEnvironment(c)
        sums = [0.0, 0.0, 0.0, 0.0]  # oracle, stat, model, random

        for trial in range(num_trials):
            env.reset(seed=trial)

            # Oracle
            _, r_oracle = optimal_scheduling_oracle(
                env.channel_gbs, env.channel_eve, num_data_sats,
                c.num_gbs_antennas, c.num_eve_antennas, c.num_sat_antennas,
                c.an_power_ratio, snr_db, c.snr_eve_db, an_policy, eve_receiver,
            )
            sums[0] += r_oracle

            # Statistical
            _, r_stat = optimal_scheduling_statistical(
                env.channel_gbs, num_data_sats, env.eve_sampler(),
                c.num_gbs_antennas, c.num_eve_antennas, c.num_sat_antennas,
                c.an_power_ratio, snr_db, c.snr_eve_db, num_mc, an_policy, eve_receiver,
            )
            sums[1] += r_stat

            # Model
            if self.model is not None:
                r_model = self._eval_model(env, snr_db, num_data_sats, num_mc, an_policy, eve_receiver)
            else:
                r_model = r_stat
            sums[2] += r_model

            # Random
            mask_rand = random_scheduling(c.num_scheduled_sats, num_data_sats)
            data_idx = np.where(mask_rand > 0.5)[0].tolist()
            an_idx = self._an_indices(data_idx, an_policy)
            _, _, r_rand = compute_secrecy_rate(
                env.channel_gbs, env.eve_sampler(), data_idx, an_idx,
                c.an_power_ratio, snr_db, c.snr_eve_db,
                c.num_gbs_antennas, c.num_eve_antennas, c.num_sat_antennas,
                num_mc, eve_receiver,
            )
            sums[3] += r_rand

        return tuple(s / num_trials for s in sums)

    def _eval_model(
        self, env: SatelliteEnvironment, snr_db: float,
        num_data_sats: int, num_mc: int, an_policy: str, eve_receiver: str,
    ) -> float:
        with torch.no_grad():
            loc = torch.from_numpy(env.local_features()).unsqueeze(0).to(self.device)
            glb = torch.from_numpy(env.global_features()).unsqueeze(0).to(self.device)
            mask = self.model.predict_topk(loc, glb, num_data_sats).cpu().numpy()[0]

        data_idx = np.where(mask > 0.5)[0].tolist()
        an_idx = self._an_indices(data_idx, an_policy)
        _, _, r_sec = compute_secrecy_rate(
            env.channel_gbs, env.eve_sampler(), data_idx, an_idx,
            self.config.an_power_ratio, snr_db, self.config.snr_eve_db,
            self.config.num_gbs_antennas, self.config.num_eve_antennas,
            self.config.num_sat_antennas, num_mc, eve_receiver,
        )
        return r_sec

    def _an_indices(self, data_idx: list[int], policy: str) -> list[int]:
        if policy.lower() in ("max", "max_all", "all"):
            return list(range(self.config.num_scheduled_sats))
        return get_an_union_indices(data_idx, [])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_secrecy_rate(
    result: BenchmarkResult,
    output_path: str | Path | None = None,
    title: str = "Secrecy Rate vs SNR",
    show: bool = True,
) -> plt.Figure:
    """Plot secrecy rate comparison across methods."""
    fig, ax = plt.subplots(figsize=(8, 6))
    snr = result.snr_values
    ax.plot(snr, result.rates_oracle, "k-o", label="Oracle (Inst. CSI)", linewidth=2)
    ax.plot(snr, result.rates_statistical, "b-s", label="Optimal (Stat. CSI)", linewidth=2)
    ax.plot(snr, result.rates_model, "r-^", label="Set Transformer", linewidth=2)
    ax.plot(snr, result.rates_random, "g--d", label="Random", linewidth=2)
    ax.set_xlabel("GBS SNR (dB)", fontsize=12)
    ax.set_ylabel("Ergodic Secrecy Rate (bits/s/Hz)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    val_accuracies: list[float] | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot training/validation loss and accuracy curves."""
    ncols = 2 if val_accuracies else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    if ncols == 1:
        axes = [axes]
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, "b-", label="Train", linewidth=2)
    axes[0].plot(epochs, val_losses, "r-", label="Val", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    if val_accuracies:
        axes[1].plot(epochs, val_accuracies, "g-", linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig
