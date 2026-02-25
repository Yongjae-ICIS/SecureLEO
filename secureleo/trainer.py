"""Training pipeline: loss functions and Trainer class."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from secureleo.config import TrainingConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss & metrics
# ---------------------------------------------------------------------------

def bce_topk_loss(logits: Tensor, labels: Tensor, num_select: int | None = None) -> Tensor:
    """Weighted BCE loss for top-k satellite selection."""
    if num_select is not None:
        num_sats = logits.shape[1]
        pw = (num_sats - num_select) / num_select
        weight = (1 - labels) + labels * pw
        return F.binary_cross_entropy_with_logits(logits, labels, weight=weight, reduction="mean")
    return F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")


def topk_accuracy(logits: Tensor, labels: Tensor, k: int) -> float:
    """Fraction of correctly selected satellites in top-k."""
    _, pred_idx = torch.topk(logits, k, dim=1)
    _, true_idx = torch.topk(labels, k, dim=1)
    correct = 0
    for i in range(logits.shape[0]):
        pred = set(pred_idx[i].cpu().numpy().tolist())
        true = set(true_idx[i].cpu().numpy().tolist())
        correct += len(pred & true)
    return correct / (logits.shape[0] * k)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    best_val_loss: float
    best_val_accuracy: float
    best_epoch: int
    train_losses: list[float]
    val_losses: list[float]
    val_accuracies: list[float]
    model_path: Path | None


class Trainer:
    """Training loop with validation, checkpointing, and LR scheduling."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        num_data_sats: int,
        device: torch.device | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.num_data_sats = num_data_sats

        self.device = device or _resolve_device(config.device)
        self.model.to(self.device)

        self.optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.num_epochs, eta_min=config.learning_rate * 0.01)

        config.output_dir.mkdir(parents=True, exist_ok=True)
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_accuracy = 0.0
        self.best_epoch = 0

    def train(self, model_name: str = "model") -> TrainingResult:
        train_losses, val_losses, val_accs = [], [], []

        logger.info(f"Training {self.config.num_epochs} epochs on {self.device}")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            t_loss = self._train_epoch()
            v_loss, v_acc = self._validate()
            self.scheduler.step()

            train_losses.append(t_loss)
            val_losses.append(v_loss)
            val_accs.append(v_acc)

            lr = self.scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} | "
                f"Train: {t_loss:.4f} | Val: {v_loss:.4f} | Acc: {v_acc:.4f} | LR: {lr:.2e}"
            )

            if v_loss < self.best_val_loss:
                self.best_val_loss = v_loss
                self.best_val_accuracy = v_acc
                self.best_epoch = epoch
                self._save(self.config.checkpoint_dir / f"{model_name}_best.pt")

        final_path = self.config.checkpoint_dir / f"{model_name}_final.pt"
        self._save(final_path)

        return TrainingResult(
            best_val_loss=self.best_val_loss,
            best_val_accuracy=self.best_val_accuracy,
            best_epoch=self.best_epoch,
            train_losses=train_losses,
            val_losses=val_losses,
            val_accuracies=val_accs,
            model_path=self.config.checkpoint_dir / f"{model_name}_best.pt",
        )

    def _train_epoch(self) -> float:
        self.model.train()
        total, n = 0.0, 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}", leave=False)
        for local, global_, labels in pbar:
            local = local.to(self.device)
            global_ = global_.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            loss = bce_topk_loss(self.model(local, global_), labels, self.num_data_sats)
            loss.backward()
            self.optimizer.step()

            total += loss.item()
            n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return total / max(n, 1)

    def _validate(self) -> tuple[float, float]:
        self.model.eval()
        total_loss, total_acc, n = 0.0, 0.0, 0
        with torch.no_grad():
            for local, global_, labels in self.val_loader:
                local = local.to(self.device)
                global_ = global_.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(local, global_)
                total_loss += bce_topk_loss(logits, labels, self.num_data_sats).item()
                total_acc += topk_accuracy(logits, labels, self.num_data_sats)
                n += 1
        return total_loss / max(n, 1), total_acc / max(n, 1)

    def _save(self, path: Path) -> None:
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "config": {"local_dim": self.model.local_dim, "global_dim": self.model.global_dim},
        }, path)
        logger.info(f"Saved checkpoint: {path}")


def _resolve_device(device_str: str) -> torch.device:
    if device_str is None or device_str.lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)
