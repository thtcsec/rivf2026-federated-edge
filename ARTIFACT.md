# Artifact Evaluation Guide -- IEEE RIVF 2026

**Paper Title**: *Federated Edge-Cloud Resilience Architecture for Distributed Campus Networks Using Quantized Sequence Autoencoders and Multi-Agent DRL Control*  
**Authors**: Trinh Hoang Tu, Cao Tien Thanh (HUFLIT)  
**Target Track**: *Communications, Networking, IoT, Cloud Computing*  

---

## 🎯 Reproducibility Badge Checklist
This artifact supports **100% full reproducibility** for all experimental findings reported in Section IV of the paper:
- [x] **Artifact Available**: Complete source code, Docker configs, and datasets available on GitHub.
- [x] **Artifact Evaluated -- Functional**: All Python prototype scripts and FedAvg parameter aggregation models execute without errors.
- [x] **Results Reproduced**: Benchmark outputs match Table II (Latency Breakdown), Table III (MTTR), Table IV (FedAvg Convergence), Table V (Resource Scaling), and Table VI (SOTA Comparison).

---

## 🚀 1-Click Reproducibility Guide

### Option A: Standard Shell Setup
```bash
# 1. Clone repository
git clone https://github.com/thtcsec/rivf2026-federated-edge.git
cd rivf2026-federated-edge

# 2. Install dependencies
pip install -r requirements.txt

# 3. Execute master benchmark suite
bash run.sh
```

### Option B: Docker Containerized Sandbox
```bash
docker-compose up --build
```

---

## 📊 Benchmark Output Artifacts
Executing `run.sh` produces:
1. `results/table2_latency.csv`: Raw execution latency per pipeline stage across 1,000 trials.
2. `results/table6_sota_comparison.csv`: Baseline comparison against iForest, FP32 Federated LSTM, INT8 Federated Transformer.
3. `paper/figures/`: High-resolution (300 DPI) publication-ready plots (`fig1_latency_breakdown.png`, `fig2_mttr_comparison.png`, `fig3_fedavg_convergence.png`, `fig4_resource_scaling.png`).
4. `results/raw_execution_trace.log`: Microsecond execution trace log.
