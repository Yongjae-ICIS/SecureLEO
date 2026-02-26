# Supplementary Materials

**Paper:** Deep Learning-Based Secure Scheduling and Cooperative Artificial Noise Generation in LEO Satellite Networks

**Manuscript ID:** WCL2025-2914

**Journal:** IEEE Wireless Communications Letters

---

Due to the page limitations of IEEE WCL (4 pages), we provide additional experimental results and technical details in this repository as supplementary materials. These materials correspond to the revision responses referenced in our reply letter.

## Contents

| File | Reviewer Comment | Description |
|------|-----------------|-------------|
| [parameter_sensitivity.md](parameter_sensitivity.md) | R1.3 | System parameter sensitivity analysis across 12 configurations + Deep Sets baseline comparison (1M-trial GPU simulation) |
| [performance_gap_analysis.md](performance_gap_analysis.md) | R1.4, R2.3 | SNR-wise performance gap decomposition (CSI gap vs. algorithm gap) |
| [ablation_study.md](ablation_study.md) | R1.5 | Set Transformer hyperparameter ablation study (9 configurations) |
| [an_satellite_analysis.md](an_satellite_analysis.md) | R1.2 | AN satellite count analysis ($K_{\text{AN}}$ variation) |
| [convergence_analysis.md](convergence_analysis.md) | R2.5 | Training convergence analysis (50 epochs, 5 random seeds) |
| [an_protocol.md](an_protocol.md) | R2.6 | Cooperative AN generation protocol and null-space computation |

## Default System Configuration

Unless otherwise stated, experiments use the following default parameters:

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Visible satellites | $N$ | $15$ | LEO satellites in GBS field of view |
| Scheduled satellites | $K$ | $10$ | Satellites selected for transmission |
| Data satellites | $K_d$ | $2$ | Satellites transmitting data |
| AN satellites | $K_{\text{AN}}$ | $K - K_d = 8$ | Satellites transmitting artificial noise |
| GBS antennas | $M_b$ | $2$ | Zero-forcing (ZF) receiver |
| Eve antennas | $M_e$ | $2$ | MMSE receiver |
| Shadowed-Rician $K$ | $K_{\text{SR}}$ | $3$ | Rician K-factor |
| Shadowed-Rician $m$ | $m$ | $5$ | Nakagami-$m$ parameter |
| Training samples | - | 40,000 | Number of training channel realizations |
| MC samples | - | 100 | Monte Carlo samples for ergodic rate estimation |

## Reproducibility

All experiments can be reproduced using the source code in this repository. See the main [README.md](../README.md) for installation and usage instructions.

```bash
# Example: Train with a specific system configuration
python main.py train --num-samples 40000 --num-epochs 12 --num-data-sats 2

# Example: Evaluate a trained model
python main.py evaluate checkpoints/model.pt --num-trials 1000
```
