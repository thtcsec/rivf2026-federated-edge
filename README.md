# 📡 Federated Edge-Cloud Resilience Architecture (IEEE RIVF 2026)

[![Conference](https://img.shields.io/badge/Conference-IEEE%20RIVF%202026-blue.svg)](https://rivf2026.org/#content)
[![Format](https://img.shields.io/badge/Format-IEEE%20Tran%20(A4%20Double--Column)-brightgreen.svg)](rivf2026.pdf)
[![Paper PDF](https://img.shields.io/badge/Paper-Compiled%20PDF-red.svg)](rivf2026.pdf)
[![Artifact Evaluation](https://img.shields.io/badge/Artifact-Reproducible-success.svg)](README.md)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official Research Artifact Repository for the paper submitted to **The 20th IEEE RIVF International Conference on Computing and Communication Technologies (IEEE RIVF 2026)**, VinUniversity, Hanoi, Vietnam.

> **Track**: *Communications, Networking, IoT, Cloud Computing*  
> **Paper Title**: *Federated Edge-Cloud Resilience Architecture for Distributed Campus Networks Using Quantized Sequence Autoencoders and Multi-Agent DRL Control*  
> **Authors**: Trinh Hoang Tu and Cao Tien Thanh (Department of Cybersecurity, Faculty of IT, HUFLIT)

---

## 📌 Abstract

Modern multi-campus university infrastructures face increasing cybersecurity risks due to high-density IoT deployments, unmanaged student endpoints, and massive distributed telemetry flows. Traditional centralized threat detection systems suffer from bandwidth saturation, privacy degradation, and prolonged Mean Time to Respond (MTTR > 30 minutes). 

In this paper, we propose a novel **Federated Edge-Cloud Resilience Architecture** engineered specifically for distributed campus networks. By deploying lightweight, PyTorch INT8-quantized TCN-GRU sequence autoencoders at edge gateways and executing decentralized Federated Averaging (FedAvg) aggregation, our framework achieves privacy-preserving real-time anomaly detection without streaming raw packet traces to central servers. Enriched high-confidence security events are published over a distributed Redis Stream pub/sub bus to a cloud-based dynamic risk engine and multi-agent Deep Reinforcement Learning (DRL) SDN controller for automated Zero-Trust containment.

Experimental evaluations on benchmark intrusion datasets demonstrate sub-10ms mitigation trueness ($4.15 \pm 0.07$ ms, Mean ± SD), a >99.99% reduction in MTTR compared to manual SOC workflows ($8.2 \pm 0.5$ ms vs $1510 \pm 340$ s), an optimal F1-score of 0.9412, and minimal edge gateway resource overhead (13.8% CPU, 48.5 MB RAM under 10,000 events/sec).

---

## 🚀 Key Architectural Contributions

1. **Decentralized Edge Federated Sequence Autoencoders**:
   - PyTorch INT8-quantized TCN-GRU sequence autoencoders deployed at edge taps.
   - Localized anomaly detection + FedAvg model parameter aggregation across multi-campus nodes without raw data centralization.
2. **Event-Driven Pub/Sub & Context Fusion**:
   - High-throughput Redis Stream event bus (`security:telemetry:stream`).
   - Spatial-temporal identity context fusion mapping IP/MAC addresses to user roles and trust scores.
3. **Closed-Loop DRL SDN Orchestration**:
   - Cloud Dynamic Risk Resolver combined with a Deep Q-Network (DQN) agent executing discrete flow isolation actions ($\mathcal{A} \in \{a_0, a_1, a_2, a_3\}$).
4. **Reproducible Research Artifact**:
   - Full PyTorch models, simulation scripts, raw execution traces, Docker Compose runner, and automated evaluation pipeline.

---

## 📂 Repository Structure

```
rivf2026/
├── IEEEtran.cls                 # IEEE Official Document Class (v1.8b)
├── rivf2026.tex                 # Main LaTeX source file (IEEE Conference format)
├── rivf2026.pdf                 # Compiled Camera-Ready PDF (2-column A4)
├── rivf2026.aux / .log          # LaTeX build artifacts
├── README.md                    # Detailed research artifact documentation
├── requirements.txt             # Python dependencies for experiments
├── run.sh                       # One-click execution script for reproduction
├── docker-compose.yml           # Docker stack configuration for Redis Stream & Runner
└── results/                     # Experimental metrics, CSVs, and figures
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

## 📄 IEEE RIVF 2026 Citation & Metadata

```bibtex
@inproceedings{tu2026federated,
  author    = {Trinh Hoang Tu and Cao Tien Thanh},
  title     = {Federated Edge-Cloud Resilience Architecture for Distributed Campus Networks Using Quantized Sequence Autoencoders and Multi-Agent DRL Control},
  booktitle = {Proc. 20th IEEE International Conference on Computing and Communication Technologies (RIVF 2026)},
  address   = {Hanoi, Vietnam},
  year      = {2026},
  publisher = {IEEE Xplore},
  note      = {Track: Communications, Networking, IoT, Cloud Computing}
}
```

---

## 📜 License & Support

This project is licensed under the MIT License. Developed by **Trinh Hoang Tu** and **Cao Tien Thanh** at the Department of Cybersecurity, Faculty of IT, HUFLIT. Contact: `tht.csec2005@gmail.com`, `thanhct@huflit.edu.vn`.
