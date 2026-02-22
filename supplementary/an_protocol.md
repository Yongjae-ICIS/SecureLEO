# Cooperative AN Generation Protocol (R2.6)

**Reviewer Comment (R2.6):** "The null space constraint for AN generation needs further explanation, particularly regarding how AN satellites compute the null space in a distributed manner."

## Overview

In our framework, non-data satellites transmit **artificial noise (AN)** in the null space of the legitimate GBS channel to degrade the eavesdropper's reception without affecting the intended communication. This document provides a detailed description of the AN generation protocol.

## System Model Recap

- **K_d data satellites**: Transmit data streams to the GBS using the legitimate channel
- **K_AN = K - K_d AN satellites**: Transmit cooperative artificial noise
- **GBS**: Equipped with M_b antennas, uses a zero-forcing (ZF) receiver
- **Eve**: Equipped with M_e antennas, uses an MMSE receiver

## AN Generation Protocol

### Step 1: Channel Estimation (GBS Pilot Phase)

The GBS broadcasts pilot signals to all K scheduled satellites. Each satellite i estimates its channel to the GBS:

```
h_i in C^{M_b x 1}    (channel from satellite i to GBS)
```

This is the **legitimate channel** and is estimated via standard pilot-based channel estimation. The eavesdropper's channel is NOT required.

### Step 2: Null Space Computation (Per-Satellite)

Each AN satellite j (j not in the data satellite set D) independently computes the null space of its own channel to the GBS.

**Procedure:**

1. Compute the Singular Value Decomposition (SVD) of h_j^H:

```
h_j^H = U * Sigma * V^H
```

where V = [v_1 | v_2 | ... | v_{M_b}].

2. The null space of h_j^H is spanned by the right singular vectors corresponding to zero singular values:

```
Null(h_j^H) = span{v_2, v_3, ..., v_{M_b}}
```

(Since h_j is M_b x 1, h_j^H has rank 1, so the null space has dimension M_b - 1.)

3. Form the null-space projection matrix:

```
P_j = V_null * V_null^H
```

where V_null = [v_2 | ... | v_{M_b}].

### Step 3: AN Signal Generation

Each AN satellite j generates a random noise vector z_j and projects it onto the null space:

```
w_j = P_j * z_j = V_null * V_null^H * z_j
```

The transmitted AN signal from satellite j is:

```
x_j = sqrt(p_AN) * w_j / ||w_j||
```

where p_AN is the AN transmit power.

### Step 4: Signal Reception

**At the GBS:**

The received signal from AN satellite j at the GBS is:

```
h_j^H * x_j = h_j^H * P_j * z_j = 0
```

This is **exactly zero** because w_j lies in the null space of h_j^H. Therefore, the AN causes **no interference** to the legitimate communication.

**At the eavesdropper (Eve):**

The received AN from satellite j at Eve is:

```
g_j^H * x_j =/= 0    (in general)
```

Since the eavesdropper's channel g_j is **independent** of the legitimate channel h_j (zero correlation assumption), the null space of h_j^H is NOT the null space of g_j^H. Therefore, the AN appears as **random interference** at the eavesdropper, degrading its SINR.

## Protocol Diagram

```
Phase 1: Pilot Broadcast (GBS -> Satellites)
==============================================
  GBS ---[pilot]---> Satellite 1 (data)     -> estimates h_1
  GBS ---[pilot]---> Satellite 2 (data)     -> estimates h_2
  GBS ---[pilot]---> Satellite 3 (AN)       -> estimates h_3
  GBS ---[pilot]---> Satellite 4 (AN)       -> estimates h_4
  ...
  GBS ---[pilot]---> Satellite K (AN)       -> estimates h_K

Phase 2: Data + AN Transmission (Satellites -> GBS, Eve)
==============================================
  Satellite 1 ---[data s_1]-----------> GBS (desired signal)
  Satellite 2 ---[data s_2]-----------> GBS (desired signal)
  Satellite 3 ---[AN w_3 in Null(h_3)]-> GBS (zero interference)
  Satellite 4 ---[AN w_4 in Null(h_4)]-> GBS (zero interference)
  ...                                    |
  Satellite K ---[AN w_K in Null(h_K)]-> GBS (zero interference)
                                         |
                        Eve (receives data + AN as interference)

Phase 3: Signal Processing
==============================================
  GBS: ZF receiver -> decode s_1, s_2 (AN-free)
  Eve: MMSE receiver -> decode s_1, s_2 (AN-corrupted, degraded SINR)
```

## Distributed Computation

A key advantage of our AN protocol is that it operates in a **fully distributed** manner:

1. **No inter-satellite coordination required**: Each AN satellite independently computes its own null space using only its own channel estimate h_j. There is no need for satellites to share channel information with each other.

2. **No centralized controller**: The GBS only broadcasts pilots; it does not need to compute or distribute AN beamforming vectors.

3. **Scalable**: Adding more AN satellites requires no changes to the protocol. Each new AN satellite simply estimates its channel and computes its own null space projection.

4. **Low overhead**: The only additional overhead compared to standard satellite communication is the null-space computation (one SVD per AN satellite), which is computationally inexpensive for small M_b (typically 2--4 antennas).

## Mathematical Justification

### Why Null-Space AN is Optimal

For a single AN satellite j with channel h_j to GBS and g_j to Eve:

- **Constraint**: The AN must cause zero interference at the GBS, i.e., h_j^H * w_j = 0
- **Objective**: Maximize interference at Eve, i.e., maximize E[|g_j^H * w_j|^2]

Since g_j is unknown (only statistical CSI available), and assuming g_j is isotropically distributed in the null space of h_j^H (due to the zero-correlation assumption), the optimal strategy is to spread the AN power uniformly across the null space, which is exactly what our protocol achieves.

### Zero-Correlation Assumption

We assume the GBS-satellite channel H and the Eve-satellite channel G are **uncorrelated** (rho = 0). This is justified because:

1. **Spatial separation**: The GBS and Eve are at different physical locations, experiencing different propagation paths
2. **Independent fading**: Under Shadowed-Rician fading, the channel realizations are independent for different receivers
3. **Conservative assumption**: Zero correlation represents the hardest case for AN design (no information about Eve's channel can be inferred from the legitimate channel)

Under this assumption, random AN satellite selection (among the scheduled K satellites) is justified, as there is no channel-based criterion to prefer one AN satellite over another.
