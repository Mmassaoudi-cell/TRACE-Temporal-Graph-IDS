# TRACE — Temporal-Relational Attack-path Cognition Engine

**A Novel Temporal GNN-Based Intrusion Detection with Attention-Guided Attack-Path Analysis**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch%20Geometric-2.7-3c9.svg)](https://pytorch-geometric.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)
[![Conference](https://img.shields.io/badge/IECON-2026%20(submitted)-orange.svg)](https://www.ieee-ies.org/)

TRACE is a temporal graph neural network framework for **Industrial IoT (IIoT) intrusion
detection**. Instead of classifying network flows in isolation, TRACE builds *protocol-role
communication graphs* from flow records, learns over real protocol channels with
edge-conditioned message passing, models the temporal evolution of multi-stage attack
campaigns, and produces analyst-interpretable attack-propagation paths via graph attention.

> Official implementation of the IECON 2026 paper by Massaoudi, Ez Eddin, Refaat, and Davis.
> Repository: ![The TRACE Architecture for Industrial IoT Intrusion Detection](assets/trace_architecture.png)

---

## Architecture

![The TRACE Architecture for Industrial IoT Intrusion Detection](assets/trace_architecture.png)

TRACE converts raw IIoT telemetry (network traffic, sensor readings, system logs) into a
**dynamic protocol-role graph** and processes it through five stages:

| Stage | Component | Role |
|-------|-----------|------|
| 1 | **TANE** — Temporal-Aware Node Embedding | Combines initial node features with temporal positional / sinusoidal encodings to produce time-stamped node states `H⁽⁰⁾(t)`. |
| 2 | **TA-GNN** — Temporal-Aware Graph Neural Network | Edge-conditioned message passing with dynamic edge masking for temporal causality and a temporal-aware attention mechanism, yielding refined embeddings `H⁽ᴸ⁾(t)`. |
| 3 | **GMHA** — Graph Multi-Head Attention | Aggregates global graph context across heads into comprehensive, context-aware sequence embeddings `Z(t)`. |
| 4 | **CE** — Contrastive Enhancement | Self-supervised auxiliary task that augments graph views (node/edge masking) and maximizes agreement between views while minimizing it against negatives (`L_con`). |
| 5 | **Classification Head** | An MLP maps `Z(t)` to intrusion labels (e.g., DoS, MITM, Normal). |

In the paper's full IIoT configuration, the spatial encoder uses an edge-conditioned
continuous-kernel convolution (**NNConv**) followed by a **GAT** layer, attention pooling,
a **bidirectional LSTM**, and **multi-scale multi-head attention** operating at short-,
medium-, and long-term timescales. GNN depth is selected automatically via a
**Dirichlet-energy collapse** test to avoid over-smoothing on topology-poor benchmarks.

### Key contributions

1. **Protocol-role graph construction** — destination ports map to semantic service-role
   nodes (MQTT broker, SSH server, DNS, NTP, …), source-port ranges map to client-role
   nodes, and measured flow statistics become edge attributes on real protocol channels.
2. **Edge-conditioned spatial learning (NNConv)** — per-edge weight matrices conditioned on
   actual traffic behavior, instead of uniform or synthetic edges.
3. **Attention-guided attack-path analysis (GAT)** — extractable per-edge attention
   coefficients rank the highest-risk device-service communication paths.
4. **Multi-scale temporal modeling (BiLSTM + multi-scale attention)** — captures the
   heterogeneous dynamics of escalating multi-stage campaigns under class imbalance.

---

## Repository layout

```text
TRACE-Temporal-Graph-IDS/
├── assets/
│   └── trace_architecture.png      # architecture diagram (above)
├── github/
│   └── trace_model.py              # leakage-safe packaged NSL-KDD implementation
├── Data/
│   └── KDD/
│       ├── kdd_train.csv
│       └── kdd_test.csv
└── README.md
```

---

## The reproducible NSL-KDD variant in this folder

NSL-KDD (like UNSW-NB15) does **not** publish explicit source/destination endpoint
topology, so the full protocol-role graph degenerates to a percentile-binned *fallback*
topology that carries limited structural signal. For these topology-poor benchmarks the
repository ships an efficient, **leakage-safe** TRACE variant that captures the same
surrogate-graph + temporal-modeling ideas in a lightweight, fully reproducible pipeline:

1. Preprocess numerical and categorical flow features using transformations fitted on
   **training data only**.
2. Derive surrogate graph statistics from train-fitted percentile node bins.
3. Concatenate flow features and surrogate graph features.
4. Split training data into model-training and validation folds.
5. Apply Borderline-SMOTE rare-class augmentation to the **model-training fold only**.
6. Train a class-balanced, regularized XGBoost multiclass classifier.
7. Calibrate class-specific thresholds on the held-out validation fold.

> **Leakage-safety note:** every transformation (scaling, binning, SMOTE, threshold
> calibration) is fit strictly inside the training partition; the test set never
> influences any fitted statistic.

### Default configuration

The efficient TRACE variant selected by the targeted NSL-KDD sweep:

```text
surrogate nodes       16
rare-class target     12000 samples/class
XGBoost estimators    1100
maximum tree depth    5
learning rate         0.03
subsample             0.90
column subsample      0.86
validation split      0.20
random seed           42
```

---

## Requirements

```bash
pip install numpy pandas scikit-learn imbalanced-learn xgboost
```

Optional, for the full temporal-GNN configuration described in the paper:

```bash
pip install torch torch-geometric
```

---

## Reproduce the NSL-KDD evaluation

From the repository root:

```bash
python github/trace_model.py \
  --train Data/KDD/kdd_train.csv \
  --test  Data/KDD/kdd_test.csv
```

On Windows PowerShell:

```powershell
python github/trace_model.py `
  --train Data/KDD/kdd_train.csv `
  --test  Data/KDD/kdd_test.csv
```




---

## Datasets

TRACE is evaluated on four public benchmarks spanning different IIoT / network intrusion
scenarios. Only NSL-KDD is packaged here for direct reproduction; the others follow the
preprocessing described in the paper.

| Dataset | Samples | Features | Classes | Graph construction |
|---------|--------:|---------:|--------:|--------------------|
| **NSL-KDD** | 125,973 train / 22,544 test | 41 | 5 | Flow-feature fallback graph (W=10, 14 nodes) |
| **RT-IoT2022** | 123,117 | 83 | 12 | Protocol-role graph, 50-flow windows, 14 nodes (1,598 snapshots) |
| **UNSW-NB15** | 82,332 train / 175,341 test | 45 | 10 | Flow-feature fallback graph (W=10, 14 nodes) |
| **APA-DDoS** | 151,200 | 22 | 3 | Fallback / cluster topology |

All datasets use temporal sequences of length `T = 8` in the full configuration.


## Citation

If you use this code or build on TRACE, please cite:

```bibtex
@inproceedings{massaoudi2026trace,
  title     = {A Novel Temporal {GNN}-Based Intrusion Detection with
               Attention-Guided Attack-Path Analysis},
  author    = {Massaoudi, Mohamed and Ez Eddin, Maymouna and
               Refaat, Shady S. and Davis, Katherine R.},
  booktitle = {Proceedings of the IEEE Industrial Electronics Society
               Annual Conference (IECON)},
  year      = {2026},
  note      = {Temporal-Relational Attack-path Cognition Engine (TRACE)}
}
```

---

## Acknowledgments

This work is supported by the U.S. Department of Energy under award **DE-CR0000018**.

---

## License

Released under the MIT License. See `LICENSE` for details.
