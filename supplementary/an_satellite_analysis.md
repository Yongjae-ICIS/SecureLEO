# AN Satellite Configuration Analysis (R1.2)

**Reviewer Comment (R1.2):** "The mechanism for selecting AN satellites should be further analyzed. How does the number of AN satellites affect performance?"

## Background

In our framework, K scheduled satellites are divided into:
- **K_d data satellites**: Transmit useful data to the GBS
- **K_AN = K - K_d AN satellites**: Transmit cooperative artificial noise to degrade the eavesdropper's reception

The AN satellites transmit noise in the **null space of the legitimate GBS channel**, ensuring zero interference at the GBS while maximizing disruption to the eavesdropper.

## Experiment Setup

We vary K_AN in {2, 4, 6, 8} while keeping K=10 and K_d=2 fixed (except K_AN=2, where K_d is adjusted accordingly to maintain K=10):

| K_AN | K_d | K | Description |
|------|-----|---|-------------|
| 2 | 8 | 10 | Minimal AN, maximum data streams |
| 4 | 6 | 10 | Moderate AN |
| 6 | 4 | 10 | Balanced AN/data |
| 8 | 2 | 10 | Maximum AN (default configuration) |

For each configuration, we train a dedicated Set Transformer model and evaluate it against the oracle-optimal baseline.

## Results

### Model/Oracle Secrecy Rate Ratio

| K_AN | Model/Oracle (%) |
|------|-----------------|
| 2 | 93.4 |
| 4 | 86.9 |
| 6 | 80.0 |
| 8 | 88.3 |

### Analysis

1. **All configurations achieve 80--93% of oracle**: The framework consistently performs well across different AN satellite counts, demonstrating robustness to the AN/data satellite ratio.

2. **K_AN = 2 achieves highest ratio (93.4%)**: With fewer AN satellites, the scheduling problem has fewer degrees of freedom, making it easier for the model to approximate the oracle. However, the absolute secrecy rate may be lower due to weaker AN protection.

3. **K_AN = 8 (default) achieves 88.3%**: The default configuration provides a good balance between AN effectiveness and scheduling approximation quality.

4. **Non-monotonic trend**: The ratio dips at K_AN = 6 (80.0%) and recovers at K_AN = 8 (88.3%), suggesting an interplay between the AN's ability to degrade the eavesdropper and the combinatorial complexity of the scheduling problem.

## Discussion

### Why Random AN Selection is Justified

In our framework, the Set Transformer selects **which K satellites to schedule** (the scheduling decision), and the K_d data satellites are selected from the scheduled set. The remaining K - K_d satellites automatically become AN satellites.

The AN satellite selection is implicitly optimized through the scheduling decision: by choosing the right subset of K satellites, the framework jointly optimizes both data and AN satellite placement.

Furthermore, since we assume **zero correlation** between the GBS-satellite channel (H) and the Eve-satellite channel (G), there is no additional information to exploit for AN satellite selection beyond the legitimate channel quality, which is already captured by the scheduling decision.

### Null-Space AN Design

Regardless of which satellites are designated as AN transmitters, each AN satellite independently computes the null space of its channel to the GBS (see [AN Protocol](an_protocol.md) for details). This ensures:
- **Zero interference to GBS**: AN lies in the null space of the legitimate channel
- **Maximum disruption to Eve**: AN appears as random interference at the eavesdropper
