# 📡 Federated Edge-Cloud Resilience Architecture (IEEE RIVF 2026)

[![Conference](https://img.shields.io/badge/Conference-IEEE%20RIVF%202026-blue.svg)](https://rivf2026.org/#content)
[![Format](https://img.shields.io/badge/Format-IEEE%20Tran%20(A4%20Double--Column)-brightgreen.svg)](rivf2026.pdf)
[![Paper PDF](https://img.shields.io/badge/Paper-Compiled%20PDF-red.svg)](rivf2026.pdf)
[![Artifact Evaluation](https://img.shields.io/badge/Artifact-Reproducible-success.svg)](README.md)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official Research Artifact Repository for the manuscript submitted to **The 20th IEEE RIVF International Conference on Computing and Communication Technologies (IEEE RIVF 2026)**, VinUniversity, Hanoi, Vietnam.

> **Track**: *Communications, Networking, IoT, Cloud Computing* / *Cyber-Security*  
> **Paper Title**: *Federated Edge-Cloud Resilience Architecture for Distributed Campus Networks Using Quantized Sequence Autoencoders and Deep Reinforcement Learning SDN Control*  
> **Authors**: Trinh Hoang Tu and Cao Tien Thanh (Faculty of Information Technology, Ho Chi Minh City University of Foreign Languages - Information Technology, Ho Chi Minh City, Vietnam)

---

## 📌 Abstract

Modern multi-campus higher education infrastructures face escalating cyber threats driven by high-density IoT deployments, unmanaged student endpoints, and massive distributed telemetry flows. Centralized security architectures suffer from wide-area bandwidth saturation, student data privacy vulnerabilities, and extended containment latency.

In this paper, we propose a privacy-aware, event-driven **Federated Edge-Cloud Resilience Architecture** engineered specifically for distributed campus environments. The platform deploys lightweight PyTorch INT8-quantized Temporal Convolutional Network and Gated Recurrent Unit (TCN-GRU) sequence autoencoders directly at edge gateways, executing localized anomaly scoring and participating in decentralized Federated Averaging (FedAvg) aggregation without raw telemetry egress. Enriched high-confidence security events are published over an asynchronous Redis Stream bus to a cloud-based dynamic risk engine and a Deep Q-Network (DQN) Software-Defined Networking (SDN) controller for automated closed-loop containment.

Evaluated on benchmark intrusion datasets (InSDN and CSE-CIC-IDS2018) under non-IID client partitions ($K=5$ campus clients), the proposed framework achieves the highest detection F1-score ($0.9412$) among evaluated baselines, a total pipeline compute latency of $4.150 \pm 0.070$~ms, and an automated closed-loop mitigation latency of $8.2 \pm 0.5$~ms. Furthermore, INT8 quantization compresses model footprint to $0.48$~MB ($74.0\%$ reduction), maintaining minimal edge CPU utilization ($13.8\%$) under sustained throughput of $10{,}000$~events/sec.

---

## 🚀 Key Architectural Contributions

1. **Quantized Edge Sequence Autoencoders**:
   - PyTorch INT8 dynamic-quantized TCN-GRU sequence autoencoders deployed at edge taps ($0.48$~MB memory footprint).
   - Sub-millisecond unsupervised flow anomaly scoring without payload inspection.
2. **Privacy-Aware Federated Averaging**:
   - Collaborative FedAvg model parameter synchronization across distributed campus nodes without raw data centralization.
   - Non-IID Dirichlet partition ($\alpha = 0.5$) robustness.
3. **Event-Driven Pub/Sub & Context Fusion**:
   - High-throughput Redis Stream event bus (`security:telemetry:stream`).
   - Spatial-temporal identity context fusion mapping IP/MAC addresses to user roles and composite risk scores.
4. **Closed-Loop DRL SDN Orchestration**:
   - Cloud Dynamic Risk Resolver combined with a Deep Q-Network (DQN) agent executing discrete flow containment playbooks ($\mathcal{A} \in \{a_0, a_1, a_2, a_3\}$).
5. **Reproducible Research Artifact**:
   - Full PyTorch models, simulation scripts, raw execution traces, Docker Compose runner, and automated evaluation pipeline.

---

## 📂 Repository Structure

```
rivf2026/
├── IEEEtran.cls                 # IEEE Official Document Class (v1.8b)
├── rivf2026.tex                 # Main LaTeX source file (IEEE Conference format, 6 pages)
├── rivf2026.pdf                 # Compiled PDF (2-column A4)
├── README.md                    # Detailed research artifact documentation
├── requirements.txt             # Python dependencies for experiments
├── run.sh                       # One-click execution script for reproduction
├── docker-compose.yml           # Docker stack configuration for Redis Stream & Runner
├── paper/figures/               # High-resolution (300 DPI) publication figures
└── results/                     # Experimental metrics, CSVs, and logs
```

---

## 🛠️ Quick Start & Reproducibility

### Option 1: Native Python Environment
```bash
# 1. Clone repository
git clone https://github.com/thtcsec/rivf2026-federated-edge.git
cd rivf2026-federated-edge

# 2. Setup virtual environment & install requirements
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Run master reproduction script
bash run.sh
```

### Option 2: Docker Compose (1-Click Containerized Setup)
```bash
docker-compose up --build
```

---

## 📄 Citation & Metadata

```bibtex
@article{tu2026federated,
  author    = {Trinh Hoang Tu and Cao Tien Thanh},
  title     = {Federated Edge-Cloud Resilience Architecture for Distributed Campus Networks Using Quantized Sequence Autoencoders and Deep Reinforcement Learning SDN Control},
  journal   = {Preprint / Submitted Manuscript},
  year      = {2026},
  note      = {Submitted to IEEE RIVF 2026}
}
```

---

## 📜 License & Support

This project is licensed under the MIT License. Developed by **Trinh Hoang Tu** and **Cao Tien Thanh** at the Faculty of Information Technology, Ho Chi Minh City University of Foreign Languages - Information Technology (HUFLIT). Contact: `tht.csec2005@gmail.com`, `thanhct@huflit.edu.vn`.
