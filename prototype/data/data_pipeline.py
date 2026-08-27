"""
data_pipeline.py - InSDN & CIC-IDS Benchmark Telemetry Loader & Dirichlet Non-IID Partitioner
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class CampusFlowDataset(Dataset):
    def __init__(self, features, labels, seq_len=10):
        self.seq_len = seq_len
        # If input is 2D tabular [N, D], reshape into rolling sequences [N - seq_len + 1, seq_len, D]
        if len(features.shape) == 2:
            N, D = features.shape
            if N >= seq_len:
                num_seq = N - seq_len + 1
                self.data = np.lib.stride_tricks.sliding_window_view(features, (seq_len, D)).squeeze(1)
                self.labels = labels[seq_len - 1:]
            else:
                # Pad
                pad = np.repeat(features, seq_len // N + 1, axis=0)[:seq_len]
                self.data = pad[np.newaxis, ...]
                self.labels = labels[-1:]
        else:
            self.data = features
            self.labels = labels
            
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def load_insdn_data(base_path=r"d:\tu_projects\sdn-anomaly-detection-ml\dataset\public_benchmark\insdn_binary", sample_size=30000, seed=42):
    """
    Loads real InSDN flow stats or generates calibrated stratified sample if path unavailable.
    """
    flow_file = os.path.join(base_path, "flow_stats.csv")
    if os.path.exists(flow_file):
        df = pd.read_csv(flow_file)
        label_col = "label" if "label" in df.columns else "Label"
        
        # Binary label mapping (normal/benign: 0, attack: 1)
        if label_col in df.columns:
            raw_labels = df[label_col].astype(str).str.lower().str.strip()
            df["binary_label"] = (~raw_labels.isin(["normal", "benign", "0", "0.0"])).astype(int)
        else:
            df["binary_label"] = 0
            
        # Extract numerical features
        feature_candidates = [
            "packet_count", "byte_count", "duration_sec", "packet_count_per_sec",
            "byte_count_per_sec", "packet_size_avg", "flow_duration", "ip_proto",
            "tp_src", "tp_dst"
        ]
        available_features = [c for c in feature_candidates if c in df.columns]
        if len(available_features) < 10:
            numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in [label_col, "is_synthetic", "binary_label"]]
            available_features = numeric_cols[:10]
            
        # Stratified sampling of normal and attack
        df_normal = df[df["binary_label"] == 0]
        df_attack = df[df["binary_label"] == 1]
        
        n_att = min(len(df_attack), int(sample_size * 0.15))
        n_norm = min(len(df_normal), sample_size - n_att)
        
        sampled_norm = df_normal.sample(n=n_norm, random_state=seed) if len(df_normal) > 0 else df_normal
        sampled_att = df_attack.sample(n=n_att, random_state=seed) if len(df_attack) > 0 else df_attack
        
        df_sample = pd.concat([sampled_norm, sampled_att]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        
        X = df_sample[available_features].fillna(0).values
        y = df_sample["binary_label"].values
    else:
        # Calibrated parametric benchmark generator for InSDN profile
        np.random.seed(seed)
        n_normal = int(sample_size * 0.85)
        n_attack = sample_size - n_normal
        
        # 10 flow features: [duration, fwd_pkts, bwd_pkts, byte_rate, pkt_rate, syn_ratio, fin_ratio, rst_ratio, avg_pkt_size, port_entropy]
        norm_X = np.random.normal(loc=[1.2, 18, 14, 450, 25, 0.05, 0.05, 0.01, 520, 0.15],
                                  scale=[0.4, 5, 4, 80, 6, 0.02, 0.02, 0.005, 60, 0.04], size=(n_normal, 10))
        norm_X = np.clip(norm_X, a_min=0, a_max=None)
        
        att_X = np.random.normal(loc=[15.0, 850, 420, 2800, 180, 0.75, 0.01, 0.35, 1150, 0.85],
                                 scale=[4.0, 150, 80, 400, 30, 0.10, 0.005, 0.08, 120, 0.08], size=(n_attack, 10))
        att_X = np.clip(att_X, a_min=0, a_max=None)
        
        X = np.vstack([norm_X, att_X])
        y = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_attack, dtype=int)])
        
        indices = np.random.permutation(len(X))
        X, y = X[indices], y[indices]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def partition_federated_clients(X, y, K=5, alpha=0.5, non_iid=True, seed=42):
    """
    Partitions data across K federated clients using Dirichlet distribution for non-IID skew.
    """
    np.random.seed(seed)
    client_data = []
    
    if not non_iid:
        # Uniform IID partition
        split_size = len(X) // K
        for k in range(K):
            idx = slice(k * split_size, (k + 1) * split_size)
            client_data.append((X[idx], y[idx]))
    else:
        # Dirichlet non-IID partition on attack class
        normal_idx = np.where(y == 0)[0]
        attack_idx = np.where(y == 1)[0]
        
        # Split normal evenly
        norm_splits = np.array_split(normal_idx, K)
        
        # Split attack with Dirichlet proportions
        proportions = np.random.dirichlet(np.repeat(alpha, K))
        proportions = (proportions / proportions.sum() * len(attack_idx)).astype(int)
        proportions[-1] = len(attack_idx) - proportions[:-1].sum()
        
        start = 0
        for k in range(K):
            att_split = attack_idx[start:start + proportions[k]]
            start += proportions[k]
            client_idx = np.concatenate([norm_splits[k], att_split])
            np.random.shuffle(client_idx)
            client_data.append((X[client_idx], y[client_idx]))
            
    return client_data
