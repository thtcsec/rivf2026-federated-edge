"""
run_real_experiments.py - Complete Empirical Execution & True Scientific Figure Generation for RIVF 2026
"""

import os
import sys
import time
import copy
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score, roc_curve, precision_recall_curve, auc
)
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "paper", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 1. DATASET PROCESSING
# ----------------------------------------------------------------------
class FlowSeqDataset(Dataset):
    def __init__(self, X, y, seq_len=10):
        N, D = X.shape
        if N >= seq_len:
            self.data = np.lib.stride_tricks.sliding_window_view(X, (seq_len, D)).squeeze(1)
            self.labels = y[seq_len - 1:]
        else:
            pad = np.repeat(X, seq_len // N + 1, axis=0)[:seq_len]
            self.data = pad[np.newaxis, ...]
            self.labels = y[-1:]
            
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def load_dataset(dataset_name="insdn", sample_size=25000, seed=42):
    if dataset_name == "insdn":
        csv_path = r"d:\tu_projects\sdn-anomaly-detection-ml\dataset\public_benchmark\insdn_binary\flow_stats.csv"
    else:
        csv_path = r"d:\tu_projects\sdn-anomaly-detection-ml\dataset\public_benchmark\cicids2017_3class\test.csv"
        
    df = pd.read_csv(csv_path)
    label_col = "label" if "label" in df.columns else "Label"
    
    if dataset_name == "insdn":
        raw_labels = df[label_col].astype(str).str.lower().str.strip()
        df["target"] = (~raw_labels.isin(["normal", "benign", "0", "0.0"])).astype(int)
    else:
        df["target"] = (df[label_col].astype(str) != "0").astype(int)
        
    feature_cols = [
        "packet_count", "byte_count", "duration_sec", "packet_count_per_sec",
        "byte_count_per_sec", "packet_size_avg", "flow_duration", "ip_proto",
        "tp_src", "tp_dst"
    ]
    avail_cols = [c for c in feature_cols if c in df.columns]
    if len(avail_cols) < 10:
        numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c not in [label_col, "target", "is_synthetic"]]
        avail_cols = numeric[:10]
        
    # Balanced stratified sample
    df_norm = df[df["target"] == 0]
    df_att = df[df["target"] == 1]
    
    n_att = min(len(df_att), int(sample_size * 0.35))
    n_norm = min(len(df_norm), sample_size - n_att)
    
    sampled_norm = df_norm.sample(n=n_norm, random_state=seed)
    sampled_att = df_att.sample(n=n_att, random_state=seed)
    df_sample = pd.concat([sampled_norm, sampled_att]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    X = df_sample[avail_cols].replace([np.inf, -np.inf], 0).fillna(0).values
    y = df_sample["target"].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 80/20 train/test split
    split = int(len(X) * 0.8)
    return (X[:split], y[:split]), (X[split:], y[split:])

# ----------------------------------------------------------------------
# 2. MODEL ARCHITECTURE
# ----------------------------------------------------------------------
class TCNGRUAnomalyDetector(nn.Module):
    def __init__(self, num_features=10, hidden_dim=32):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=hidden_dim, kernel_size=3, padding=1, dilation=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [B, L, D] -> [B, D, L]
        x_trans = x.transpose(1, 2)
        f = self.relu(self.conv1(x_trans))
        f = self.relu(self.conv2(f)).transpose(1, 2)
        _, h = self.gru(f)
        out = self.classifier(self.dropout(h.squeeze(0)))
        return out

def federated_averaging(global_model, client_models):
    global_w = copy.deepcopy(global_model.state_dict())
    for key in global_w.keys():
        global_w[key] = torch.stack([client_models[i].state_dict()[key].float() for i in range(len(client_models))], dim=0).mean(dim=0)
    global_model.load_state_dict(global_w)
    return global_model

def train_local(model, dataloader, epochs=1, lr=0.005):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        for bx, by in dataloader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
    return model

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for bx, by in dataloader:
            logits = model(bx)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_preds.extend(probs)
            all_targets.extend(by.numpy().flatten())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    bin_preds = (all_preds > 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, bin_preds, average='binary', zero_division=0)
    auc_score = roc_auc_score(all_targets, all_preds)
    
    fpr, tpr, _ = roc_curve(all_targets, all_preds)
    prec_curve, rec_curve, _ = precision_recall_curve(all_targets, all_preds)
    pr_auc = auc(rec_curve, prec_curve)
    
    return {
        "precision": p, "recall": r, "f1": f1, "roc_auc": auc_score, "pr_auc": pr_auc,
        "fpr": fpr, "tpr": tpr, "prec_curve": prec_curve, "rec_curve": rec_curve,
        "probs": all_preds, "targets": all_targets
    }

# ----------------------------------------------------------------------
# 3. RUN FEDERATED EXPERIMENTS
# ----------------------------------------------------------------------
def run_all():
    print("=" * 70)
    print("🔥 EXECUTING REAL FEDERATED TRAINING AND EMPIRICAL EVALUATION")
    print("=" * 70)
    
    # 1. Load InSDN Data
    print("\n[1/4] Processing InSDN & CIC-IDS Benchmarks...")
    (X_tr_in, y_tr_in), (X_te_in, y_te_in) = load_dataset("insdn", sample_size=25000, seed=42)
    (X_tr_cic, y_tr_cic), (X_te_cic, y_te_cic) = load_dataset("cicids", sample_size=25000, seed=42)
    
    test_in_loader = DataLoader(FlowSeqDataset(X_te_in, y_te_in, 10), batch_size=128, shuffle=False)
    test_cic_loader = DataLoader(FlowSeqDataset(X_te_cic, y_te_cic, 10), batch_size=128, shuffle=False)
    
    K = 5
    # Partition Uniform & Dirichlet for InSDN
    split_size = len(X_tr_in) // K
    uniform_loaders = [DataLoader(FlowSeqDataset(X_tr_in[i*split_size:(i+1)*split_size], y_tr_in[i*split_size:(i+1)*split_size], 10), batch_size=64, shuffle=True) for i in range(K)]
    
    # Dirichlet Non-IID
    np.random.seed(42)
    norm_idx = np.where(y_tr_in == 0)[0]
    att_idx = np.where(y_tr_in == 1)[0]
    props = np.random.dirichlet(np.repeat(0.5, K))
    counts = (props / props.sum() * len(att_idx)).astype(int)
    counts[-1] = len(att_idx) - counts[:-1].sum()
    
    dirichlet_loaders = []
    norm_splits = np.array_split(norm_idx, K)
    cur = 0
    for i in range(K):
        c_att = att_idx[cur:cur+counts[i]]
        cur += counts[i]
        c_idx = np.concatenate([norm_splits[i], c_att])
        np.random.shuffle(c_idx)
        dirichlet_loaders.append(DataLoader(FlowSeqDataset(X_tr_in[c_idx], y_tr_in[c_idx], 10), batch_size=64, shuffle=True))
        
    isolated_loader = uniform_loaders[0]

    # --- Run 1: Uniform FedAvg ---
    print("\n[2/4] Training Condition 1: Uniform FedAvg (K=5)...")
    global_uniform = TCNGRUAnomalyDetector(10, 32)
    client_models = [TCNGRUAnomalyDetector(10, 32) for _ in range(K)]
    f1_uniform_rounds = []
    
    for r in range(1, 21):
        for cm in client_models:
            cm.load_state_dict(global_uniform.state_dict())
        for k in range(K):
            client_models[k] = train_local(client_models[k], uniform_loaders[k], epochs=1, lr=0.005)
        global_uniform = federated_averaging(global_uniform, client_models)
        res = evaluate_model(global_uniform, test_in_loader)
        f1_uniform_rounds.append(res["f1"])
        if r % 5 == 0 or r == 1 or r == 12:
            print(f"   Round {r:02d}/20 | F1: {res['f1']:.4f} | ROC-AUC: {res['roc_auc']:.4f}")
            
    # --- Run 2: Dirichlet Non-IID FedAvg ---
    print("\n[2/4] Training Condition 2: Dirichlet Non-IID FedAvg (alpha=0.5)...")
    global_dirichlet = TCNGRUAnomalyDetector(10, 32)
    client_models = [TCNGRUAnomalyDetector(10, 32) for _ in range(K)]
    f1_dirichlet_rounds = []
    
    for r in range(1, 21):
        for cm in client_models:
            cm.load_state_dict(global_dirichlet.state_dict())
        for k in range(K):
            client_models[k] = train_local(client_models[k], dirichlet_loaders[k], epochs=1, lr=0.005)
        global_dirichlet = federated_averaging(global_dirichlet, client_models)
        res = evaluate_model(global_dirichlet, test_in_loader)
        f1_dirichlet_rounds.append(res["f1"])
        if r % 5 == 0 or r == 1 or r == 16:
            print(f"   Round {r:02d}/20 | F1: {res['f1']:.4f} | ROC-AUC: {res['roc_auc']:.4f}")
            
    # --- Run 3: Isolated Local Model ---
    print("\n[2/4] Training Condition 3: Isolated Local Model (No FL)...")
    local_model = TCNGRUAnomalyDetector(10, 32)
    f1_isolated_rounds = []
    for r in range(1, 21):
        local_model = train_local(local_model, isolated_loader, epochs=1, lr=0.005)
        res = evaluate_model(local_model, test_in_loader)
        f1_isolated_rounds.append(res["f1"])
        
    # Quantization Benchmark
    print("\n[3/4] Quantizing Model to INT8 & Benchmarking Latency...")
    fp32_path = os.path.join(RESULTS_DIR, "model_fp32.pt")
    int8_path = os.path.join(RESULTS_DIR, "model_int8.pt")
    torch.save(global_uniform.state_dict(), fp32_path)
    
    int8_model = torch.quantization.quantize_dynamic(global_uniform, {nn.GRU, nn.Linear}, dtype=torch.qint8)
    torch.save(int8_model.state_dict(), int8_path)
    
    fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
    int8_size = os.path.getsize(int8_path) / (1024 * 1024)
    
    dummy = torch.randn(1, 10, 10)
    for _ in range(50): _ = int8_model(dummy)
    t0 = time.perf_counter()
    for _ in range(5000): _ = int8_model(dummy)
    t1 = time.perf_counter()
    measured_lat_ms = ((t1 - t0) / 5000.0) * 1000.0
    
    print(f"   -> INT8 Model Size: {int8_size:.4f} MB (from FP32: {fp32_size:.4f} MB)")
    print(f"   -> Measured INT8 Single-Sequence Latency: {measured_lat_ms:.4f} ms")
    
    # Final Evaluations on both Benchmarks
    eval_insdn = evaluate_model(global_uniform, test_in_loader)
    eval_cicids = evaluate_model(global_uniform, test_cic_loader)
    
    print(f"\n✅ InSDN Test Set: Precision={eval_insdn['precision']:.4f}, Recall={eval_insdn['recall']:.4f}, F1={eval_insdn['f1']:.4f}, ROC-AUC={eval_insdn['roc_auc']:.4f}, PR-AUC={eval_insdn['pr_auc']:.4f}")
    print(f"✅ CIC-IDS Test Set: Precision={eval_cicids['precision']:.4f}, Recall={eval_cicids['recall']:.4f}, F1={eval_cicids['f1']:.4f}, ROC-AUC={eval_cicids['roc_auc']:.4f}, PR-AUC={eval_cicids['pr_auc']:.4f}")
    
    # ------------------------------------------------------------------
    # 4. GENERATE TRUE EMPIRICAL FIGURES (FIG 4 & FIG 6)
    # ------------------------------------------------------------------
    print("\n[4/4] Plotting 100% Empirically Validated Figures for RIVF 2026...")
    
    # Styling
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8.5, 'grid.alpha': 0.35, 'grid.linestyle': '--'
    })
    
    # FIG 4: Convergence
    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=300)
    rounds = np.arange(1, 21)
    ax.plot(rounds, f1_uniform_rounds, marker='o', markersize=4.5, color='#0284c7', lw=2.2, label=f'Uniform FedAvg (Final F1={f1_uniform_rounds[-1]:.4f})')
    ax.plot(rounds, f1_dirichlet_rounds, marker='s', markersize=4.5, color='#10b981', lw=2.0, label=rf'Non-IID Dirichlet $\alpha=0.5$ (Final F1={f1_dirichlet_rounds[-1]:.4f})')
    ax.plot(rounds, f1_isolated_rounds, marker='^', markersize=4.5, color='#f59e0b', lw=1.8, linestyle='--', label=f'Isolated Local Model (Final F1={f1_isolated_rounds[-1]:.4f})')
    
    ax.set_xlabel("Federated Communication Round ($T$)", fontweight='bold')
    ax.set_ylabel("Test F1-Score", fontweight='bold')
    ax.set_xticks(np.arange(1, 21, 2))
    ax.set_ylim(0.40, 1.02)
    ax.grid(True)
    ax.legend(loc='lower right', framealpha=0.95)
    ax.set_title("Federated Convergence Across Communication Rounds", fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig4_fedavg_convergence.png"), dpi=300)
    plt.close()
    
    # FIG 6: ROC & PR Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.4), dpi=300)
    
    # (a) ROC
    ax1.plot(eval_insdn["fpr"], eval_insdn["tpr"], color='#0284c7', lw=2.2, label=f'InSDN (AUC = {eval_insdn["roc_auc"]:.3f})')
    ax1.plot(eval_cicids["fpr"], eval_cicids["tpr"], color='#10b981', lw=2.0, linestyle='-.', label=f'CSE-CIC-IDS (AUC = {eval_cicids["roc_auc"]:.3f})')
    ax1.plot([0, 1], [0, 1], color='#94a3b8', linestyle=':', label='Random (AUC = 0.500)')
    ax1.set_xlabel("False Positive Rate (FPR)", fontweight='bold')
    ax1.set_ylabel("True Positive Rate (TPR)", fontweight='bold')
    ax1.set_title("(a) Receiver Operating Characteristic", fontweight='bold', fontsize=9.5)
    ax1.grid(True)
    ax1.legend(loc='lower right', framealpha=0.95)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    
    # (b) PR
    ax2.plot(eval_insdn["rec_curve"], eval_insdn["prec_curve"], color='#0284c7', lw=2.2, label=f'InSDN (PR-AUC = {eval_insdn["pr_auc"]:.3f})')
    ax2.plot(eval_cicids["rec_curve"], eval_cicids["prec_curve"], color='#10b981', lw=2.0, linestyle='-.', label=f'CSE-CIC-IDS (PR-AUC = {eval_cicids["pr_auc"]:.3f})')
    ax2.set_xlabel("Recall", fontweight='bold')
    ax2.set_ylabel("Precision", fontweight='bold')
    ax2.set_title("(b) Precision-Recall Curves", fontweight='bold', fontsize=9.5)
    ax2.grid(True)
    ax2.legend(loc='lower left', framealpha=0.95)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(0.40, 1.02)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig6_ablation_pr_roc.png"), dpi=300)
    plt.close()
    
    # Export summary JSON
    summary = {
        "uniform_final_f1": float(f1_uniform_rounds[-1]),
        "dirichlet_final_f1": float(f1_dirichlet_rounds[-1]),
        "isolated_final_f1": float(f1_isolated_rounds[-1]),
        "insdn_precision": float(eval_insdn["precision"]),
        "insdn_recall": float(eval_insdn["recall"]),
        "insdn_f1": float(eval_insdn["f1"]),
        "insdn_roc_auc": float(eval_insdn["roc_auc"]),
        "insdn_pr_auc": float(eval_insdn["pr_auc"]),
        "cicids_precision": float(eval_cicids["precision"]),
        "cicids_recall": float(eval_cicids["recall"]),
        "cicids_f1": float(eval_cicids["f1"]),
        "cicids_roc_auc": float(eval_cicids["roc_auc"]),
        "cicids_pr_auc": float(eval_cicids["pr_auc"]),
        "int8_model_size_mb": float(int8_size),
        "int8_latency_ms": float(measured_lat_ms)
    }
    with open(os.path.join(RESULTS_DIR, "empirical_metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "=" * 70)
    print("🎯 EXECUTION COMPLETE! METRICS EXPORTED TO results/empirical_metrics_summary.json")
    print("=" * 70)
    return summary

if __name__ == "__main__":
    run_all()
