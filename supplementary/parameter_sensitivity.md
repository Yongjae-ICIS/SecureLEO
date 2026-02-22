# Parameter Sensitivity Analysis (R1.3)

**Reviewer Comment (R1.3):** "The authors should demonstrate the robustness of the proposed framework under diverse system parameter settings."

## Experiment Setup

We evaluate the proposed Set Transformer-based scheduling framework across **12 system configurations** by varying three key parameters:

- **N** (number of visible satellites): {12, 15, 18}
- **K_d** (number of data satellites): {2, 3}
- **M_e** (number of Eve antennas): {2, 4}

For each configuration, we:
1. Generate 40,000 training samples with the corresponding system parameters
2. Train a Set Transformer model (d=128, h=4, L=2) for 12 epochs
3. Evaluate using the secrecy rate benchmark with Oracle (brute-force with instantaneous Eve CSI), Statistical (brute-force with MC-sampled statistical Eve CSI), Model (Set Transformer), and Random baselines

The **Model/Oracle ratio** measures how closely the learned model approximates the oracle-optimal scheduling that has access to instantaneous eavesdropper CSI.

## Results

### Full Results Table

| Config | N | K_d | M_e | Scheduling Accuracy (%) | Model/Oracle Secrecy Rate (%) |
|--------|---|-----|-----|------------------------|-------------------------------|
| N12_Kd2_Me2 | 12 | 2 | 2 | 88.5 | 81.8 |
| N12_Kd2_Me4 | 12 | 2 | 4 | 88.7 | 75.5 |
| N12_Kd3_Me2 | 12 | 3 | 2 | 79.7 | 84.4 |
| N12_Kd3_Me4 | 12 | 3 | 4 | 80.9 | 74.7 |
| N15_Kd2_Me2 | 15 | 2 | 2 | 88.1 | 82.0 |
| N15_Kd2_Me4 | 15 | 2 | 4 | 90.2 | 75.3 |
| N15_Kd3_Me2 | 15 | 3 | 2 | 79.6 | 84.4 |
| N15_Kd3_Me4 | 15 | 3 | 4 | 80.7 | 74.8 |
| N18_Kd2_Me2 | 18 | 2 | 2 | 89.0 | 82.3 |
| N18_Kd2_Me4 | 18 | 2 | 4 | 88.8 | 75.5 |
| N18_Kd3_Me2 | 18 | 3 | 2 | 80.0 | 84.2 |
| N18_Kd3_Me4 | 18 | 3 | 4 | 80.8 | 74.2 |

### Analysis by Parameter

#### Effect of N (Number of Visible Satellites)

| N | Avg. Model/Oracle (%) | Range (%) |
|---|----------------------|-----------|
| 12 | 79.1 | 74.7 -- 84.4 |
| 15 | 79.1 | 74.8 -- 84.4 |
| 18 | 79.1 | 74.2 -- 84.2 |

The framework maintains consistent performance as N increases from 12 to 18, demonstrating **scalability** to different constellation sizes.

#### Effect of K_d (Number of Data Satellites)

| K_d | Avg. Model/Oracle (%) | Range (%) |
|-----|----------------------|-----------|
| 2 | 78.7 | 75.3 -- 82.3 |
| 3 | 79.6 | 74.2 -- 84.4 |

Performance remains stable across different K_d values, indicating the model generalizes well to different data/AN satellite ratios.

#### Effect of M_e (Number of Eve Antennas)

| M_e | Avg. Model/Oracle (%) | Range (%) |
|-----|----------------------|-----------|
| 2 | 83.2 | 81.8 -- 84.4 |
| 4 | 75.0 | 74.2 -- 75.5 |

When M_e increases from 2 to 4, the eavesdropper becomes stronger, making the scheduling problem harder. The Model/Oracle ratio decreases by approximately 8 percentage points, but the framework still achieves over 74% of the oracle performance even against a stronger eavesdropper.

## Key Findings

1. **Robust across N**: Performance variation is less than 1% across N in {12, 15, 18}, confirming that the Set Transformer scales well with constellation size.
2. **Stable across K_d**: The framework adapts to different data/AN satellite ratios without significant performance degradation.
3. **Graceful degradation with M_e**: Against a stronger eavesdropper (M_e=4), performance decreases but remains practically useful (74--76% of oracle).
4. **Consistent overall range**: 74--84% Model/Oracle across all 12 configurations, demonstrating the framework's generalizability.
