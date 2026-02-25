#!/usr/bin/env python3
"""CLI entry point for SecureLEO framework.

Usage:
    python main.py train --config experiments/default.yaml
    python main.py evaluate checkpoints/model.pt
    python main.py version
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import typer

from secureleo import __version__
from secureleo.config import ExperimentConfig, ModelConfig, SystemConfig, TrainingConfig
from secureleo.dataset import SatelliteDataset, create_data_loaders, generate_training_dataset
from secureleo.models import DeepSetsScheduler, SetTransformerScheduler
from secureleo.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="secureleo",
    help="Satellite Physical Layer Security - Deep Learning Scheduler",
    add_completion=False,
)


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


@app.command()
def train(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config file"),
    num_samples: int = typer.Option(40000, help="Training dataset size"),
    num_epochs: int = typer.Option(12, help="Number of training epochs"),
    batch_size: int = typer.Option(256, help="Batch size"),
    num_data_sats: int = typer.Option(2, "--num-data-sats", "-k", help="Data satellites (K_d)"),
    architecture: str = typer.Option("set_transformer", "--arch", "-a", help="Model architecture"),
    embed_dim: int = typer.Option(128, "--embed-dim", "-d", help="Embedding dimension"),
    num_heads: int = typer.Option(4, "--num-heads", help="Attention heads"),
    num_layers: int = typer.Option(2, "--num-layers", help="SAB layers"),
    learning_rate: float = typer.Option(1e-3, "--lr", help="Learning rate"),
    device: str = typer.Option("auto", help="Device: auto, cpu, cuda, mps"),
    output_dir: Path = typer.Option(Path("results"), help="Output directory"),
    seed: int = typer.Option(42, help="Random seed"),
    num_mc_samples: int = typer.Option(100, "--mc-samples", help="MC samples for labels"),
    eve_receiver: str = typer.Option("mmse", "--eve-rx", help="Eve receiver: zf or mmse"),
    regenerate_data: bool = typer.Option(False, "--regenerate", help="Force data regeneration"),
) -> None:
    """Train a satellite scheduling model."""
    logger.info(f"SecureLEO v{__version__}")

    if config_path and config_path.exists():
        config = ExperimentConfig.from_yaml(config_path)
        logger.info(f"Loaded config from {config_path}")
    else:
        config = ExperimentConfig(
            system=SystemConfig(),
            model=ModelConfig(
                architecture=architecture, embed_dim=embed_dim,
                num_heads=num_heads, num_encoder_layers=num_layers,
            ),
            training=TrainingConfig(
                num_samples=num_samples, batch_size=batch_size,
                num_epochs=num_epochs, learning_rate=learning_rate,
                device=device, seed=seed, output_dir=output_dir,
            ),
        )

    logger.info(f"\n{config}")
    device_obj = _resolve_device(config.training.device)
    logger.info(f"Using device: {device_obj}")

    # Generate or load dataset
    dataset_path = config.training.output_dir / f"dataset_k{num_data_sats}_{eve_receiver}.npz"
    if not dataset_path.exists() or regenerate_data:
        logger.info("Generating training dataset...")
        generate_training_dataset(
            output_path=dataset_path,
            system_config=config.system,
            num_samples=config.training.num_samples,
            num_data_sats=num_data_sats,
            num_mc=num_mc_samples,
            seed=config.training.seed,
            eve_receiver=eve_receiver,
        )
        logger.info(f"Dataset saved to {dataset_path}")
    else:
        logger.info(f"Using existing dataset: {dataset_path}")

    train_loader, val_loader = create_data_loaders(
        dataset_path, batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )

    ds = SatelliteDataset(dataset_path)
    logger.info(f"Feature dims: local={ds.local_dim}, global={ds.global_dim}")

    # Create model
    if config.model.architecture == "set_transformer":
        model = SetTransformerScheduler(
            local_dim=ds.local_dim, global_dim=ds.global_dim,
            embed_dim=config.model.embed_dim, num_heads=config.model.num_heads,
            num_layers=config.model.num_encoder_layers,
            ff_dim=config.model.ff_hidden_dim, dropout=config.model.dropout,
        )
        model_name = f"settx_{eve_receiver}"
    else:
        model = DeepSetsScheduler(
            local_dim=ds.local_dim, global_dim=ds.global_dim,
            embed_dim=config.model.embed_dim,
            hidden_dim=config.model.ff_hidden_dim, dropout=config.model.dropout,
        )
        model_name = f"deepsets_{eve_receiver}"

    logger.info(f"Model: {config.model.architecture} ({sum(p.numel() for p in model.parameters()):,} params)")

    trainer = Trainer(model, train_loader, val_loader, config.training, num_data_sats, device_obj)
    result = trainer.train(model_name=model_name)

    logger.info("Training complete!")
    logger.info(f"Best val loss: {result.best_val_loss:.4f} (epoch {result.best_epoch + 1})")
    logger.info(f"Best val accuracy: {result.best_val_accuracy:.4f}")
    logger.info(f"Model saved to: {result.model_path}")


@app.command()
def evaluate(
    model_path: Path = typer.Argument(..., help="Path to model checkpoint"),
    config_path: Optional[Path] = typer.Option(None, "--config", help="Config file"),
    num_trials: int = typer.Option(1000, help="Trials per SNR"),
    snr_start: int = typer.Option(0, help="SNR start (dB)"),
    snr_end: int = typer.Option(20, help="SNR end (dB)"),
    snr_step: int = typer.Option(2, help="SNR step (dB)"),
    num_data_sats: int = typer.Option(2, "--num-data-sats", help="Data satellites"),
    device: str = typer.Option("auto", help="Device"),
    output_dir: Path = typer.Option(Path("results"), help="Output directory"),
) -> None:
    """Evaluate a trained model."""
    from secureleo.benchmark import Benchmarker, plot_secrecy_rate

    logger.info(f"Evaluating model: {model_path}")

    config = ExperimentConfig.from_yaml(config_path) if config_path and config_path.exists() else ExperimentConfig()
    device_obj = _resolve_device(device)

    checkpoint = torch.load(model_path, map_location=device_obj)
    mc = checkpoint.get("config", {})
    model = SetTransformerScheduler(
        local_dim=mc.get("local_dim", 4), global_dim=mc.get("global_dim", 40),
        embed_dim=config.model.embed_dim, num_heads=config.model.num_heads,
        num_layers=config.model.num_encoder_layers,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device_obj)
    model.eval()

    snr_values = list(range(snr_start, snr_end + 1, snr_step))
    benchmarker = Benchmarker(config.system, model, device_obj)
    result = benchmarker.run(snr_values=snr_values, num_trials=num_trials, num_data_sats=num_data_sats)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_secrecy_rate(result, output_path=output_dir / "secrecy_rate_comparison.pdf", show=False)

    logger.info("Results:")
    for i, snr in enumerate(snr_values):
        logger.info(
            f"SNR={snr:2d}dB: Oracle={result.rates_oracle[i]:.3f}, "
            f"Model={result.rates_model[i]:.3f}, Random={result.rates_random[i]:.3f}"
        )


@app.command()
def version() -> None:
    """Show version and device information."""
    print(f"SecureLEO version {__version__}")
    print(f"PyTorch version {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("MPS (Apple Silicon) available")
    else:
        print("Running on CPU")


if __name__ == "__main__":
    app()
