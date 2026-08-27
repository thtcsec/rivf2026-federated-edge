# 📡 Federated Edge-Cloud Resilience Architecture (IEEE RIVF 2026)

[![Conference](https://img.shields.io/badge/Conference-IEEE%20RIVF%202026-blue.svg)](https://rivf2026.org/)
[![Format](https://img.shields.io/badge/Format-IEEE%20Tran%20(A4%20Double--Column)-brightgreen.svg)](rivf2026.pdf)
[![Paper PDF](https://img.shields.io/badge/Paper-Compiled%20PDF-red.svg)](rivf2026.pdf)
[![Artifact Evaluation](https://img.shields.io/badge/Artifact-Reproducible-success.svg)](README.md)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official Research Artifact Repository for the manuscript submitted to **The 20th IEEE RIVF International Conference on Computing and Communication Technologies (IEEE RIVF 2026)**, VinUniversity, Hanoi, Vietnam.

> **Track**: *Communications, Networking, IoT, Cloud Computing* / *Cyber-Security*  
> **Paper Title**: *Federated Edge-Cloud Resilience Architecture for Distributed Campus Networks Using Quantized TCN-GRU Sequence Detectors and Deep Reinforcement Learning for SDN Control*  
> **Authors**: Trinh Hoang Tu and Cao Tien Thanh (Faculty of Information Technology, Ho Chi Minh City University of Foreign Languages - Information Technology, Ho Chi Minh City, Vietnam)

---

## 📌 Abstract

Modern multi-campus higher education infrastructures face escalating cyber threats driven by high-density IoT deployments, unmanaged student endpoints, and massive distributed telemetry flows. Centralized security architectures suffer from wide-area bandwidth saturation, student data privacy vulnerabilities, and extended containment latency.

In this paper, we propose a privacy-aware, event-driven **Federated Edge-Cloud Resilience Architecture** engineered specifically for distributed campus environments. The platform deploys lightweight PyTorch dynamically INT8-quantized Temporal Convolutional Network and Gated Recurrent Unit (TCN-GRU) sequence detectors directly at edge gateways, executing localized anomaly scoring and participating in collaborative Federated Averaging (FedAvg) aggregation without raw telemetry egress. Enriched high-confidence security events are published over an asynchronous Redis Streams bus to a cloud-based dynamic risk engine and a Deep Q-Network (DQN) Software-Defined Networking (SDN) controller for automated closed-loop containment.

Evaluated across $K=5$ federated campus clients on the InSDN benchmark with additional cross-dataset zero-shot evaluation on CSE-CIC-IDS2018, the proposed framework achieves a binary anomaly detection F1-score of $0.8582$ ($\text{ROC-AUC} = 0.9729$, $\text{PR-AUC} = 0.9567$) under uniform partitioning, outperforming isolated uncoordinated local baselines ($0.8314$). Under non-IID Dirichlet-skewed ($\alpha=0.5$) client attack distributions, the global model achieves $F1 = 0.8097$, demonstrating stable collaborative convergence despite statistical heterogeneity. The edge compute pipeline completes in $8.290 \pm 0.140$~ms (with single-sequence INT8 scoring in $4.555$~ms), and automated closed-loop OpenFlow mitigation completes in $12.34 \pm 0.55$~ms on Open vSwitch datapaths. Evaluated over $100$ test episodes across 5 random seeds ($500$ total evaluation episodes), the learned DQN policy achieves a $98.4 \pm 0.6\%$ threat containment rate with only $2.1 \pm 0.4\%$ false quarantine. Furthermore, dynamic INT8 quantization compresses the GRU and linear model footprint to $0.027$~MB ($27.5$~KB), maintaining minimal edge CPU utilization ($13.8\%$) under sustained throughput of $10{,}000$~events/sec.

---

## 🚀 Key Architectural Contributions

1. **Quantized Edge Sequence Detection**:
   - PyTorch dynamic INT8 quantization on recurrent (GRU) and linear classification layers ($0.027$~MB memory footprint).
   - Sub-$5$~ms single-sequence flow classification at edge gateways without raw payload inspection.
2. **Privacy-Aware Federated Averaging**:
   - Collaborative FedAvg model parameter synchronization across $K=5$ distributed campus nodes.
   - Non-IID Dirichlet partition ($\alpha = 0.5$) evaluation across heterogeneous campus nodes.
3. **Event-Driven Pub/Sub & Context Fusion**:
   - High-throughput Redis Streams event bus (`security:telemetry:stream`).
   - Spatial-temporal identity context fusion mapping IP/MAC addresses to user roles and composite risk scores $R_t \in [0, 1]$.
4. **Closed-Loop DRL SDN Orchestration**:
   - Cloud Dynamic Risk Resolver combined with a Deep Q-Network (DQN) agent executing discrete flow containment playbooks ($\mathcal{A} \in \{a_0, a_1, a_2, a_3\}$).
   - Automated mitigation completing in $12.34 \pm 0.55$~ms on Open vSwitch datapaths.

---

## 🔬 Experimental Reproduction

To reproduce all experiments, models, and publication figures from scratch:

```bash
# 1. Install dependencies
pip install torch scikit-learn pandas numpy matplotlib seaborn psutil

# 2. Run end-to-end empirical training & evaluation
python evaluation/run_real_experiments.py

# 3. Generate all 300 DPI publication figures
python evaluation/scripts/generate_publication_figures.py
```

---

## 📄 License & Citation

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
