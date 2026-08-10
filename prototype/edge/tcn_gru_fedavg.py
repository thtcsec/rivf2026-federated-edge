"""
tcn_gru_fedavg.py - PyTorch INT8 TCN-GRU Autoencoder & Federated Averaging (FedAvg) Implementation
"""

import copy
import torch
import torch.nn as nn
import numpy as np

class TCNGRUResilienceModel(nn.Module):
    def __init__(self, num_features=10, hidden_dim=32, num_classes=6):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.gru_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.gru_decoder = nn.GRU(hidden_dim, num_features, batch_first=True)
        
    def forward(self, x):
        # x: [B, L, D] -> [B, D, L]
        x_trans = x.transpose(1, 2)
        feat = self.relu(self.conv1(x_trans)).transpose(1, 2)
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
        global_w[key] = torch.stack([client_models[i].state_dict()[key].float() * weights[i] for i in range(len(client_models))], dim=0).sum(dim=0)
        
    global_model.load_state_dict(global_w)
    return global_model
