"""Configuration dataclasses for SatPLS framework."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass(frozen=True)
class SystemConfig:
    """System-level configuration for satellite communication."""

    num_visible_sats: int = 15
    num_scheduled_sats: int = 10
    num_data_sats: int = 2
    num_sat_antennas: int = 1
    num_gbs_antennas: int = 2
    num_eve_antennas: int = 2
    rician_k_factor: float = 3.0
    nakagami_m_param: float = 5.0
    snr_gbs_min_db: float = 0.0
    snr_gbs_max_db: float = 20.0
    snr_eve_db: float = 10.0
    an_power_ratio: float = 0.30

    def __post_init__(self) -> None:
        if self.num_scheduled_sats > self.num_visible_sats:
            raise ValueError(
                f"num_scheduled_sats ({self.num_scheduled_sats}) > "
                f"num_visible_sats ({self.num_visible_sats})"
            )
        if self.num_data_sats > self.num_scheduled_sats:
            raise ValueError(
                f"num_data_sats ({self.num_data_sats}) > "
                f"num_scheduled_sats ({self.num_scheduled_sats})"
            )
        if not 0.0 <= self.an_power_ratio <= 1.0:
            raise ValueError(f"an_power_ratio ({self.an_power_ratio}) must be in [0, 1]")

    @property
    def num_an_sats(self) -> int:
        return self.num_scheduled_sats - self.num_data_sats

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_visible_sats": self.num_visible_sats,
            "num_scheduled_sats": self.num_scheduled_sats,
            "num_data_sats": self.num_data_sats,
            "num_sat_antennas": self.num_sat_antennas,
            "num_gbs_antennas": self.num_gbs_antennas,
            "num_eve_antennas": self.num_eve_antennas,
            "rician_k_factor": self.rician_k_factor,
            "nakagami_m_param": self.nakagami_m_param,
            "snr_gbs_min_db": self.snr_gbs_min_db,
            "snr_gbs_max_db": self.snr_gbs_max_db,
            "snr_eve_db": self.snr_eve_db,
            "an_power_ratio": self.an_power_ratio,
        }


@dataclass
class ModelConfig:
    """Neural network model configuration."""

    architecture: str = "set_transformer"
    embed_dim: int = 128
    num_heads: int = 4
    num_encoder_layers: int = 2
    ff_hidden_dim: int = 256
    dropout: float = 0.1
    num_pma_seeds: int = 1

    def __post_init__(self) -> None:
        if self.architecture == "set_transformer":
            if self.embed_dim % self.num_heads != 0:
                raise ValueError(
                    f"embed_dim ({self.embed_dim}) must be divisible by "
                    f"num_heads ({self.num_heads})"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_encoder_layers": self.num_encoder_layers,
            "ff_hidden_dim": self.ff_hidden_dim,
            "dropout": self.dropout,
            "num_pma_seeds": self.num_pma_seeds,
        }


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    num_samples: int = 100_000
    batch_size: int = 256
    num_epochs: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    ergodic_mc_samples: int = 100
    device: str = "auto"
    seed: int = 42
    num_workers: int = 4
    output_dir: Path = field(default_factory=lambda: Path("results"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))

    def __post_init__(self) -> None:
        if isinstance(self.output_dir, str):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if isinstance(self.checkpoint_dir, str):
            object.__setattr__(self, "checkpoint_dir", Path(self.checkpoint_dir))

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "ergodic_mc_samples": self.ergodic_mc_samples,
            "device": self.device,
            "seed": self.seed,
            "num_workers": self.num_workers,
            "output_dir": str(self.output_dir),
            "checkpoint_dir": str(self.checkpoint_dir),
        }


@dataclass
class ExperimentConfig:
    """Complete experiment configuration with YAML support."""

    name: str = "default"
    description: str = ""
    system: SystemConfig = field(default_factory=SystemConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> ExperimentConfig:
        if not HAS_YAML:
            raise ImportError("PyYAML required: pip install pyyaml")
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(
            name=data.get("name", path.stem),
            description=data.get("description", ""),
            system=SystemConfig(**data.get("system", {})),
            model=ModelConfig(**data.get("model", {})),
            training=TrainingConfig(**{
                k: v for k, v in data.get("training", {}).items()
                if k not in ("evaluation",)
            }),
        )

    def to_yaml(self, path: Path | str) -> None:
        if not HAS_YAML:
            raise ImportError("PyYAML required: pip install pyyaml")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name,
            "description": self.description,
            "system": self.system.to_dict(),
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def __str__(self) -> str:
        return (
            f"ExperimentConfig: {self.name}\n"
            f"  System: N={self.system.num_visible_sats}, "
            f"K={self.system.num_scheduled_sats}, K_d={self.system.num_data_sats}\n"
            f"  Antennas: M_s={self.system.num_sat_antennas}, "
            f"M_b={self.system.num_gbs_antennas}, M_e={self.system.num_eve_antennas}\n"
            f"  Model: {self.model.architecture} "
            f"(dim={self.model.embed_dim}, heads={self.model.num_heads}, "
            f"layers={self.model.num_encoder_layers})\n"
            f"  Training: {self.training.num_samples} samples, "
            f"{self.training.num_epochs} epochs"
        )
