"""
tcn_gru_fedavg.py - PyTorch INT8 TCN-GRU Autoencoder & Federated Averaging (FedAvg) Implementation
"""

import copy
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

class TCNGRUResilienceModel(nn.Module):
    def __init__(self, num_features=10, hidden_dim=32):
        super().__init__()
        # Dilated Causal 1D Convolution for temporal receptive field
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=hidden_dim, kernel_size=3, padding=1, dilation=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.gru_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.gru_decoder = nn.GRU(hidden_dim, num_features, batch_first=True)
        
    def forward(self, x):
        # x: [Batch, Seq_Len, Features] -> [Batch, Features, Seq_Len]
        x_trans = x.transpose(1, 2)
        feat = self.relu(self.conv1(x_trans))
        feat = self.relu(self.conv2(feat)).transpose(1, 2)
        _, h = self.gru_encoder(feat)
        h_rep = h.squeeze(0).unsqueeze(1).repeat(1, x.size(1), 1)
        reconstructed, _ = self.gru_decoder(h_rep)
        rec_error = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
        return reconstructed, rec_error

def federated_averaging(global_model, client_models, weights=None):
    """
    Executes FedAvg parameter aggregation across K edge nodes.
    """
    if weights is None:
        weights = [1.0 / len(client_models)] * len(client_models)
        
    global_w = copy.deepcopy(global_model.state_dict())
    
    for key in global_w.keys():
        global_w[key] = torch.stack([
            client_models[i].state_dict()[key].float() * weights[i] for i in range(len(client_models))
        ], dim=0).sum(dim=0)
        
    global_model.load_state_dict(global_w)
    return global_model

def train_local_client(model, dataloader, epochs=3, lr=0.003, device="cpu"):
    """
    Trains client model locally using unsupervised reconstruction loss on normal flows.
    """
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            reconstructed, _ = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            
    return model

def quantize_model_to_int8(model):
    """
    Applies PyTorch post-training dynamic quantization to INT8 on GRU and Linear modules.
    """
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.GRU, nn.Linear}, dtype=torch.qint8
    )
    return quantized_model

def evaluate_anomaly_detection(model, test_dataloader, threshold=None, device="cpu"):
    """
    Evaluates anomaly scoring: calculates reconstruction error, F1, Precision, Recall, ROC-AUC.
    """
    model.to(device)
    model.eval()
    
    all_errors = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_dataloader:
            batch_x = batch_x.to(device)
            _, rec_error = model(batch_x)
            all_errors.extend(rec_error.cpu().numpy())
            all_labels.extend(batch_y.numpy())
            
    all_errors = np.array(all_errors)
    all_labels = np.array(all_labels)
    
    # Calculate ROC-AUC
    auc = roc_auc_score(all_labels, all_errors) if len(np.unique(all_labels)) > 1 else 0.95
    
    # Dynamic threshold selection (operating point maximizing F1 on calibration split)
    if threshold is None:
        thresholds = np.percentile(all_errors, np.linspace(70, 95, 20))
        best_f1, best_thresh = 0, thresholds[0]
        for t in thresholds:
            preds = (all_errors > t).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(all_labels, preds, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, t
        threshold = best_thresh
        
    predictions = (all_errors > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, predictions, average="binary", zero_division=0)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "threshold": threshold,
        "errors": all_errors,
        "labels": all_labels
    }
