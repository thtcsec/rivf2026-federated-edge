"""
dqn_sdn_controller.py - Deep Q-Network (DQN) for Closed-Loop SDN Threat Containment
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_dim=5, action_dim=5, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x):
        return self.fc(x)

class SDNMitigationEnvironment:
    """
    Simulates SDN datapath dynamics, flow telemetry anomalies, and OpenFlow rule execution.
    Actions: 0: No Action, 1: Rate Limit 10M, 2: Rate Limit 1M, 3: Quarantine VLAN, 4: Drop Flow
    """
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.action_space_n = 5
        self.reset()

    def reset(self):
        self.is_attack = np.random.rand() < 0.45
        self.attack_severity = np.random.uniform(0.7, 1.0) if self.is_attack else np.random.uniform(0.05, 0.25)
        self.flow_rate = np.random.uniform(500, 2000) if self.is_attack else np.random.uniform(20, 100)
        self.alert_confidence = self.attack_severity + np.random.normal(0, 0.05)
        self.alert_confidence = np.clip(self.alert_confidence, 0.0, 1.0)
        self.consecutive_alerts = 1 if self.is_attack else 0
        self.client_id = np.random.randint(0, 5) / 5.0
        
        self.state = np.array([
            self.attack_severity,
            self.flow_rate / 2000.0,
            self.alert_confidence,
            self.consecutive_alerts / 5.0,
            self.client_id
        ], dtype=np.float32)
        
        self.step_count = 0
        return self.state

    def step(self, action):
        self.step_count += 1
        reward = 0.0
        contained = False
        false_quarantine = False

        if self.is_attack:
            if action == 4:  # Drop Flow
                reward = +10.0
                contained = True
            elif action == 3:  # Quarantine VLAN
                reward = +8.0
                contained = True
            elif action == 2:  # Rate Limit 1M
                reward = +3.0
            elif action == 1:  # Rate Limit 10M
                reward = +1.0
            else:  # No Action
                reward = -10.0
        else:  # Benign User
            if action == 0:  # No Action (Correct)
                reward = +5.0
            elif action in [1, 2]:  # Mild throttling
                reward = -2.0
            else:  # False quarantine / drop
                reward = -15.0
                false_quarantine = True

        done = True
        return self.state, reward, done, {"contained": contained, "false_quarantine": false_quarantine, "is_attack": self.is_attack}

class DQNSDNAgent:
    def __init__(self, state_dim=5, action_dim=5, lr=0.001, gamma=0.95):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = 0.1
        self.memory = deque(maxlen=2000)
        
        self.model = QNetwork(state_dim, action_dim)
        self.target_model = QNetwork(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def select_action(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values).item()

    def train_step(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(np.array(next_states))
        dones_t = torch.FloatTensor(dones).unsqueeze(1)
        
        q_eval = self.model(states_t).gather(1, actions_t)
        with torch.no_grad():
            q_next = self.target_model(next_states_t).max(1)[0].unsqueeze(1)
            q_target = rewards_t + (1 - dones_t) * self.gamma * q_next
            
        loss = self.criterion(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def evaluate_dqn_across_seeds(seeds=[21, 42, 84, 123, 777], num_episodes_per_seed=100):
    results = []
    
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        env = SDNMitigationEnvironment(seed=seed)
        agent = DQNSDNAgent(lr=0.002)
        
        # Pre-train Agent with Epsilon Decay
        for ep in range(500):
            agent.epsilon = max(0.02, 1.0 - (ep / 350.0))
            s = env.reset()
            a = agent.select_action(s)
            s_next, r, done, _ = env.step(a)
            agent.memory.append((s, a, r, s_next, done))
            agent.train_step(batch_size=32)
            if ep % 20 == 0:
                agent.target_model.load_state_dict(agent.model.state_dict())
            
        agent.target_model.load_state_dict(agent.model.state_dict())
        
        # Evaluation Phase
        attack_count = 0
        contained_count = 0
        benign_count = 0
        false_quarantine_count = 0
        
        for ep in range(num_episodes_per_seed):
            s = env.reset()
            a = agent.select_action(s, eval_mode=True)
            _, _, _, info = env.step(a)
            
            if info["is_attack"]:
                attack_count += 1
                if info["contained"]:
                    contained_count += 1
            else:
                benign_count += 1
                if info["false_quarantine"]:
                    false_quarantine_count += 1
                    
        containment_rate = (contained_count / attack_count) * 100 if attack_count > 0 else 100.0
        false_quarantine_rate = (false_quarantine_count / benign_count) * 100 if benign_count > 0 else 0.0
        
        results.append({
            "seed": seed,
            "containment_rate": containment_rate,
            "false_quarantine_rate": false_quarantine_rate
        })
        
    containment_mean = np.mean([r["containment_rate"] for r in results])
    containment_std = np.std([r["containment_rate"] for r in results])
    false_q_mean = np.mean([r["false_quarantine_rate"] for r in results])
    false_q_std = np.std([r["false_quarantine_rate"] for r in results])
    
    return {
        "containment_mean": containment_mean,
        "containment_std": containment_std,
        "false_q_mean": false_q_mean,
        "false_q_std": false_q_std,
        "raw_results": results
    }
