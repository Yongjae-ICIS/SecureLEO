# SatPLS: Secure Scheduling for LEO Satellite Networks

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Wireless%20Communications%20Letters-blue)](https://ieeexplore.ieee.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **"Deep Learning-Based Secure Scheduling and Cooperative Artificial Noise Generation in LEO Satellite Networks"**

> **Authors:** Yongjae Lee (Student Member, IEEE), Taehoon Kim (Member, IEEE), Inkyu Bang (Member, IEEE), Erdal Panayirci (Life Fellow, IEEE), and H. Vincent Poor (Life Fellow, IEEE)
>
> **Submitted to:** IEEE Wireless Communications Letters (under review)

---

## Overview

SatPLS is a deep learning-based **physical-layer security (PLS)** framework for LEO satellite communication networks. It jointly optimizes satellite scheduling and cooperative artificial noise (AN) generation using **Set Transformer** architecture, operating with only **statistical eavesdropper CSI**.

### Key Features

- **Set Transformer-Based Scheduling**: Permutation-invariant architecture that captures inter-satellite interactions for optimal subset selection
- **Cooperative AN Generation**: Non-data satellites transmit AN in the null space of legitimate channels to degrade eavesdropper reception
- **Statistical CSI Only**: No instantaneous eavesdropper CSI required — uses ergodic secrecy rate maximization via Monte Carlo sampling
- **Multi-Architecture Support**:
  - Set Transformer (default): Attention-based, permutation-invariant
  - Deep Sets: Lightweight baseline

### Performance Highlights

- **83–93% of oracle-optimal** secrecy rate across diverse configurations
- **2.6–3.7x improvement** over random scheduling baseline
- **Sub-millisecond inference** vs. seconds for brute-force search
- Robust across varying system parameters (N, K_d, M_e, K_AN)

<p align="center">
  <img src="figures/fig1_system_model.png" alt="System Model" width="500">
</p>

---

## Project Structure

```
SatPLS/
├── main.py                 # CLI entry point (train / evaluate / version)
├── pyproject.toml
├── requirements.txt
├── satpls/                 # Main package (flat, ~1500 lines)
│   ├── __init__.py         # Version & public API
│   ├── config.py           # YAML-based experiment configuration
│   ├── channel.py          # Shadowed-Rician fading channel generation
│   ├── signal.py           # Precoder + AN + SINR (ZF/MMSE) + Secrecy Rate
│   ├── scheduling.py       # Brute-force optimal + Random baseline
│   ├── models.py           # Set Transformer + Deep Sets
│   ├── dataset.py          # Environment + Dataset + Data generation
│   ├── trainer.py          # Training loop + Loss functions
│   └── benchmark.py        # Multi-baseline benchmarking + Plotting
├── experiments/
│   └── default.yaml        # Default experiment configuration
├── figures/                # Paper figures
└── supplementary/          # Additional experimental results & technical details
```

---

## Installation

### Requirements

```bash
# Clone the repository
git clone https://github.com/Yongjae-ICIS/SatPLS.git
cd SatPLS

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Dependencies

- Python >= 3.9
- PyTorch >= 1.12.0 (CUDA / MPS / CPU)
- NumPy, SciPy, Matplotlib
- typer, tqdm, PyYAML

---

## Usage

### Quick Start

```bash
# Train with default parameters
python main.py train --num-samples 40000 --num-epochs 12

# Train with YAML config
python main.py train --config experiments/default.yaml

# Evaluate a trained model
python main.py evaluate checkpoints/model.pt --num-trials 1000

# Check version & device
python main.py version
```

### Training

```bash
python main.py train \
    --num-samples 100000 \
    --num-epochs 12 \
    --batch-size 256 \
    --arch set_transformer \
    --embed-dim 128 \
    --num-heads 4 \
    --num-layers 2 \
    --lr 1e-3 \
    --mc-samples 100 \
    --device auto
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to YAML config file | None |
| `--num-samples` | Training dataset size | 40000 |
| `--num-epochs` | Number of training epochs | 12 |
| `--batch-size` | Mini-batch size | 256 |
| `--arch` | Model: set_transformer or deep_sets | set_transformer |
| `--embed-dim` | Embedding dimension | 128 |
| `--num-heads` | Attention heads | 4 |
| `--num-layers` | Number of SAB layers | 2 |
| `--lr` | Learning rate | 1e-3 |
| `--mc-samples` | MC samples for ergodic rate | 100 |
| `--num-data-sats` | Data satellites (K_d) | 2 |
| `--device` | Device: auto, cpu, cuda, mps | auto |

### System Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Visible satellites | N | 15 | LEO satellites in view |
| Scheduled satellites | K | 10 | Selected for transmission |
| Data satellites | K_d | 2 | Transmitting data |
| AN satellites | K_AN | K - K_d | Transmitting artificial noise |
| GBS antennas | M_b | 2 | Zero-forcing receiver |
| Eve antennas | M_e | 2 | MMSE receiver |
| Rician K-factor | K_SR | 3 | Shadowed-Rician parameter |
| Nakagami m | m | 5 | Shadowed-Rician parameter |

---

## Results

### Secrecy Rate Performance (Model / Oracle)

| Configuration | N | K_d | M_e | Model/Oracle (%) |
|---------------|---|-----|-----|------------------|
| N12_Kd2_Me2 | 12 | 2 | 2 | 81.8 |
| N12_Kd2_Me4 | 12 | 2 | 4 | 75.5 |
| N15_Kd2_Me2 | 15 | 2 | 2 | 82.0 |
| N15_Kd2_Me4 | 15 | 2 | 4 | 75.3 |
| N15_Kd3_Me2 | 15 | 3 | 2 | 84.4 |
| N18_Kd2_Me2 | 18 | 2 | 2 | 82.3 |
| N18_Kd3_Me2 | 18 | 3 | 2 | 84.2 |

### CSI Gap Analysis

| SNR (dB) | Oracle | Statistical | Model | Random | Statistical/Oracle (%) |
|----------|--------|-------------|-------|--------|------------------------|
| 0 | 0.63 | 0.44 | 0.10 | 0.01 | 69.8 |
| 10 | 5.07 | 4.80 | 4.00 | 1.52 | 94.7 |
| 14 | 7.53 | 7.27 | 6.46 | 3.32 | 96.5 |
| 20 | 11.41 | 11.16 | 10.34 | 6.57 | 97.8 |

### Set Transformer Ablation Study

| Parameter | Value | Model/Oracle (%) |
|-----------|-------|------------------|
| dim=64 | h=2, d=64, L=2 | 81.7 |
| dim=128 (default) | h=4, d=128, L=2 | 82.1 |
| dim=256 | h=4, d=256, L=2 | 82.1 |
| layers=1 | h=4, d=128, L=1 | 81.0 |
| layers=2 (default) | h=4, d=128, L=2 | 81.9 |
| layers=3 | h=4, d=128, L=3 | 82.2 |

---

## Supplementary Materials

Due to the page limitations of IEEE WCL, additional experimental results and technical details are provided in the [`supplementary/`](supplementary/) directory:

| Document | Description |
|----------|-------------|
| [Parameter Sensitivity Analysis](supplementary/parameter_sensitivity.md) | System parameter robustness across 12 configurations (N, K_d, M_e) |
| [Performance Gap Analysis](supplementary/performance_gap_analysis.md) | SNR-wise gap decomposition: CSI gap vs. algorithm gap |
| [Hyperparameter Ablation Study](supplementary/ablation_study.md) | Set Transformer architecture ablation (dim, heads, layers) |
| [AN Satellite Configuration](supplementary/an_satellite_analysis.md) | Effect of AN satellite count (K_AN) on secrecy rate |
| [Convergence Analysis](supplementary/convergence_analysis.md) | Training convergence over 50 epochs with 5 random seeds |
| [AN Generation Protocol](supplementary/an_protocol.md) | Null-space computation and distributed AN protocol details |

---

## Citation

*Citation information will be added upon publication.*

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- **Set Transformer**: [Lee et al., ICML 2019](https://arxiv.org/abs/1810.00825) — Permutation-invariant architecture
- **Physical-Layer Security**: Wyner's wiretap channel model and extensions

---

## Contact

For questions or issues, please open an issue or contact:
- Yongjae Lee: yjlee@edu.hanbat.ac.kr
- Inkyu Bang: ikbang@hanbat.ac.kr
