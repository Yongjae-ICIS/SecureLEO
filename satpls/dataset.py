"""Dataset, environment, and data generation for supervised learning."""
from __future__ import annotations

import multiprocessing as mp
import os
from functools import partial
from pathlib import Path

import numpy as np
import torch
from numpy.random import Generator
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from satpls.channel import create_eve_sampler, generate_channel_matrix
from satpls.config import SystemConfig
from satpls.scheduling import optimal_scheduling_statistical


# ---------------------------------------------------------------------------
# Environment: generates channel realizations and feature vectors
# ---------------------------------------------------------------------------

class SatelliteEnvironment:
    """Generates channel realizations and feature vectors for training/eval."""

    def __init__(self, config: SystemConfig | None = None, seed: int = 42):
        self.config = config or SystemConfig()
        self.rng = np.random.default_rng(seed)

        self.channel_gbs = None
        self.channel_eve = None
        self.snr_gbs_db: float = 0.0

    @property
    def num_sats(self) -> int:
        return self.config.num_scheduled_sats

    @property
    def local_dim(self) -> int:
        return 2 * self.config.num_gbs_antennas * self.config.num_sat_antennas

    @property
    def global_dim(self) -> int:
        return 2 * self.config.num_gbs_antennas * self.num_sats * self.config.num_sat_antennas

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        c = self.config
        self.snr_gbs_db = self.rng.uniform(c.snr_gbs_min_db, c.snr_gbs_max_db)

        tx = self.num_sats * c.num_sat_antennas
        self.channel_gbs = generate_channel_matrix(
            c.num_gbs_antennas, tx, c.rician_k_factor, c.nakagami_m_param, self.rng,
        )
        self.channel_eve = generate_channel_matrix(
            c.num_eve_antennas, tx, c.rician_k_factor, c.nakagami_m_param, self.rng,
        )
        return self

    def local_features(self) -> np.ndarray:
        """Per-satellite features, shape (K, local_dim)."""
        features = []
        for i in range(self.num_sats):
            s = i * self.config.num_sat_antennas
            e = s + self.config.num_sat_antennas
            block = self.channel_gbs[:, s:e]
            features.append(np.concatenate([block.real.flatten(), block.imag.flatten()]).astype(np.float32))
        return np.stack(features)

    def global_features(self) -> np.ndarray:
        """Global features, shape (global_dim,)."""
        return np.concatenate([
            self.channel_gbs.real.flatten(),
            self.channel_gbs.imag.flatten(),
        ]).astype(np.float32)

    def eve_sampler(self) -> callable:
        c = self.config
        return create_eve_sampler(
            c.num_eve_antennas,
            self.num_sats * c.num_sat_antennas,
            c.rician_k_factor,
            c.nakagami_m_param,
            self.rng,
        )


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class SatelliteDataset(Dataset):
    """Loads pre-generated training data from NPZ files."""

    def __init__(self, npz_path: str | Path):
        data = np.load(npz_path)
        self.local = data["local"]
        self.global_ = data["global_"]
        self.labels = data["label"]
        self.num_data_sats = int(data["k_data"]) if "k_data" in data.files else int(np.sum(self.labels[0]))
        self.num_sats = int(data["K"]) if "K" in data.files else self.local.shape[1]
        self.local_dim = self.local.shape[2]
        self.global_dim = self.global_.shape[1]

    def __len__(self) -> int:
        return self.local.shape[0]

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.local[idx].astype(np.float32)),
            torch.from_numpy(self.global_[idx].astype(np.float32)),
            torch.from_numpy(self.labels[idx].astype(np.float32)),
        )


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _generate_single_sample(
    seed: int,
    system_config: SystemConfig,
    num_data_sats: int,
    num_mc: int,
    eve_receiver: str,
    an_policy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one labeled sample (for multiprocessing)."""
    env = SatelliteEnvironment(system_config, seed).reset(seed)
    label, _ = optimal_scheduling_statistical(
        env.channel_gbs, num_data_sats, env.eve_sampler(),
        system_config.num_gbs_antennas, system_config.num_eve_antennas,
        system_config.num_sat_antennas, system_config.an_power_ratio,
        env.snr_gbs_db, system_config.snr_eve_db,
        num_mc, an_policy, eve_receiver,
    )
    return env.local_features(), env.global_features(), label.astype(np.float32)


def generate_training_dataset(
    output_path: str | Path,
    system_config: SystemConfig | None = None,
    num_samples: int = 40000,
    num_data_sats: int = 2,
    num_mc: int = 100,
    seed: int = 42,
    use_multiprocessing: bool = False,
    num_workers: int | None = None,
    eve_receiver: str = "mmse",
    an_policy: str = "min_sat",
) -> Path:
    """Generate labeled training dataset with brute-force oracle labels."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    system_config = system_config or SystemConfig()

    locals_, globals_, labels_ = [], [], []

    if use_multiprocessing:
        workers = num_workers or (os.cpu_count() or 1)
        fn = partial(
            _generate_single_sample,
            system_config=system_config, num_data_sats=num_data_sats,
            num_mc=num_mc, eve_receiver=eve_receiver, an_policy=an_policy,
        )
        with mp.Pool(processes=workers) as pool:
            for loc, glb, lbl in tqdm(
                pool.imap_unordered(fn, [seed + n for n in range(num_samples)]),
                total=num_samples, desc="Generating (parallel)",
            ):
                locals_.append(loc)
                globals_.append(glb)
                labels_.append(lbl)
    else:
        env = SatelliteEnvironment(system_config, seed)
        for n in tqdm(range(num_samples), desc="Generating dataset"):
            env.reset(seed=seed + n)
            label, _ = optimal_scheduling_statistical(
                env.channel_gbs, num_data_sats, env.eve_sampler(),
                system_config.num_gbs_antennas, system_config.num_eve_antennas,
                system_config.num_sat_antennas, system_config.an_power_ratio,
                env.snr_gbs_db, system_config.snr_eve_db,
                num_mc, an_policy, eve_receiver,
            )
            locals_.append(env.local_features())
            globals_.append(env.global_features())
            labels_.append(label)

    np.savez(
        output_path,
        local=np.stack(locals_),
        global_=np.stack(globals_),
        label=np.stack(labels_),
        k_data=num_data_sats,
        K=system_config.num_scheduled_sats,
        L=locals_[0].shape[-1],
        G=globals_[0].shape[-1],
    )
    return output_path


def create_data_loaders(
    train_path: str | Path,
    batch_size: int = 256,
    num_workers: int = 4,
    val_split: float = 0.1,
) -> tuple[DataLoader, DataLoader]:
    """Create train/val DataLoaders from NPZ file."""
    full_dataset = SatelliteDataset(train_path)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
