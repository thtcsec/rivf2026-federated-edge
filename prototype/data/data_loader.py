"""
data_loader.py - Synthetic InSDN Telemetry Dataset Generator for RIVF 2026
"""

import os
import csv
import numpy as np
import torch

def generate_insdn_telemetry_csv(output_path, num_samples=2000, seed=42):
    np.random.seed(seed)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    headers = [
        "flow_duration", "pkt_rate", "byte_rate", "syn_ratio", "flow_entropy",
        "pkt_in_rate", "interarrival_time", "bwd_pkt_rate", "bwd_byte_rate", "len_variance", "label"
    ]
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for i in range(num_samples):
            is_anomaly = 1 if np.random.rand() < 0.25 else 0
            if is_anomaly:
                row = [
                    np.random.uniform(5.0, 50.0),
                    np.random.uniform(1000, 50000),
                    np.random.uniform(1e6, 5e7),
                    np.random.uniform(0.7, 1.0),
                    np.random.uniform(0.1, 1.2),
                    np.random.uniform(500, 10000),
                    np.random.uniform(0.001, 0.01),
                    np.random.uniform(500, 20000),
                    np.random.uniform(5e5, 2e7),
                    np.random.uniform(100, 5000),
                    1
                ]
            else:
                row = [
                    np.random.uniform(0.1, 5.0),
                    np.random.uniform(10, 500),
                    np.random.uniform(1000, 50000),
                    np.random.uniform(0.01, 0.1),
                    np.random.uniform(3.5, 4.8),
                    np.random.uniform(1, 50),
                    np.random.uniform(0.05, 0.5),
                    np.random.uniform(10, 200),
                    np.random.uniform(500, 20000),
                    np.random.uniform(5, 50),
                    0
                ]
            writer.writerow(row)

def load_telemetry_dataset(csv_path, seq_len=10):
    data = []
    labels = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data.append([float(x) for x in row[:-1]])
            labels.append(int(row[-1]))
            
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    
    # Normalize
    mean = data.mean(axis=0)
    std = data.std(axis=0) + 1e-8
    data = (data - mean) / std
    
    num_seqs = len(data) // seq_len
    X_seq = data[:num_seqs * seq_len].reshape(num_seqs, seq_len, -1)
    y_seq = labels[:num_seqs * seq_len].reshape(num_seqs, seq_len).max(axis=1)
    
    return torch.tensor(X_seq), torch.tensor(y_seq)
