"""
generate_architecture_diagram.py
---------------------------------
Generates a crisp, publication-grade system architecture diagram for IEEE RIVF 2026:
"Federated Edge-Cloud Resilience Architecture for Distributed Campus Networks"
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

OUTPUT_DIR = Path("paper/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_arch_diagram():
    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")

    # Color palette (Modern academic theme)
    c_edge_bg    = "#EBF5FB"  # Light soft blue
    c_edge_border= "#2980B9"
    c_cloud_bg   = "#FEF9E7"  # Light soft gold/amber
    c_cloud_border="#D4AC0D"
    c_bus_bg     = "#FDEDEC"  # Light soft coral/red
    c_bus_border = "#C0392B"
    c_sdn_bg     = "#EAFAF1"  # Light soft green
    c_sdn_border = "#27AE60"
    c_box        = "#FFFFFF"
    c_box_b      = "#34495E"
    
    # ── Tier 1: Distributed Campus Edge Gateways ─────────────────────────────
    rect_tier1 = patches.FancyBboxPatch((2, 2), 27, 47, boxstyle="round,pad=0.8",
                                        facecolor=c_edge_bg, edgecolor=c_edge_border, linewidth=1.5)
    ax.add_patch(rect_tier1)
    ax.text(15.5, 46.5, "Tier 1: Distributed Edge Taps\n(Satellite Campuses A, B, C)", 
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="#1B4F72")

    # Campus A box
    box_cA = patches.FancyBboxPatch((4, 25), 23, 18, boxstyle="round,pad=0.5",
                                   facecolor="#FFFFFF", edgecolor="#2980B9", linewidth=1.2)
    ax.add_patch(box_cA)
    ax.text(15.5, 40.5, "Campus A Edge Gateway", ha="center", fontsize=8, fontweight="bold", color="#2471A3")
    ax.text(15.5, 37.0, "• 10-Dim Flow Feature Extractor", ha="center", fontsize=7, color="#2C3E50")
    ax.text(15.5, 33.5, "• PyTorch INT8 TCN-GRU AE", ha="center", fontsize=7.5, fontweight="bold", color="#884EA0")
    ax.text(15.5, 30.0, "• Local Unsupervised Scoring (RE)", ha="center", fontsize=7, color="#2C3E50")
    ax.text(15.5, 27.0, "• Local Gradient Step (Adam)", ha="center", fontsize=7, color="#2C3E50")

    # Campus B & C (Compact boxes)
    box_cB = patches.FancyBboxPatch((4, 14), 23, 9, boxstyle="round,pad=0.4",
                                   facecolor="#FFFFFF", edgecolor="#5DADE2", linewidth=1.0)
    ax.add_patch(box_cB)
    ax.text(15.5, 20.0, "Campus B Edge Tap (INT8 AE)", ha="center", fontsize=7.5, fontweight="bold", color="#2980B9")
    ax.text(15.5, 16.5, "Localized Flow Anomaly Scoring", ha="center", fontsize=7, color="#566573")

    box_cC = patches.FancyBboxPatch((4, 4), 23, 8.5, boxstyle="round,pad=0.4",
                                   facecolor="#FFFFFF", edgecolor="#5DADE2", linewidth=1.0)
    ax.add_patch(box_cC)
    ax.text(15.5, 9.5, "Campus C Edge Tap (INT8 AE)", ha="center", fontsize=7.5, fontweight="bold", color="#2980B9")
    ax.text(15.5, 6.0, "Non-DPI Telemetry Vectorization", ha="center", fontsize=7, color="#566573")

    # ── Tier 2: Event Bus & FedAvg Coordinator ──────────────────────────────
    rect_tier2 = patches.FancyBboxPatch((32, 2), 22, 47, boxstyle="round,pad=0.8",
                                        facecolor=c_bus_bg, edgecolor=c_bus_border, linewidth=1.5)
    ax.add_patch(rect_tier2)
    ax.text(43.0, 46.5, "Tier 2: Event Bus & FL\n(Privacy-Aware Sync)", 
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="#78281F")

    # FedAvg Server Box
    box_fl = patches.FancyBboxPatch((34, 25), 18, 18, boxstyle="round,pad=0.5",
                                   facecolor="#FFFFFF", edgecolor="#C0392B", linewidth=1.2)
    ax.add_patch(box_fl)
    ax.text(43.0, 40.5, "Cloud FedAvg Server", ha="center", fontsize=8, fontweight="bold", color="#922B21")
    ax.text(43.0, 36.5, "Decentralized Weight Sync", ha="center", fontsize=7, color="#2C3E50")
    ax.text(43.0, 33.0, r"$\mathbf{W}^{t+1} = \sum \frac{N_k}{N}\mathbf{W}_k^{t+1}$", ha="center", fontsize=7.5, color="#154360")
    ax.text(43.0, 29.0, "No Raw Payload Egress", ha="center", fontsize=7, fontweight="bold", color="#27AE60")
    ax.text(43.0, 26.5, "Bandwidth Savings: 98.6%", ha="center", fontsize=6.8, color="#566573")

    # Redis Stream Box
    box_redis = patches.FancyBboxPatch((34, 4), 18, 18.5, boxstyle="round,pad=0.5",
                                      facecolor="#FFFFFF", edgecolor="#C0392B", linewidth=1.2)
    ax.add_patch(box_redis)
    ax.text(43.0, 19.5, "Redis Streams Pub/Sub", ha="center", fontsize=8, fontweight="bold", color="#A93226")
    ax.text(43.0, 16.0, "security:telemetry:stream", ha="center", fontsize=6.5, family="monospace", color="#7B241C")
    ax.text(43.0, 13.0, "• High Throughput (10k ev/s)", ha="center", fontsize=7, color="#2C3E50")
    ax.text(43.0, 9.8, "• Sub-millisecond Egress", ha="center", fontsize=7, color="#2C3E50")
    ax.text(43.0, 6.5, "• Asynchronous Decoupling", ha="center", fontsize=7, color="#2C3E50")

    # ── Tier 3: Cloud Risk Engine & DRL Control ─────────────────────────────
    rect_tier3 = patches.FancyBboxPatch((57, 2), 20, 47, boxstyle="round,pad=0.8",
                                        facecolor=c_cloud_bg, edgecolor=c_cloud_border, linewidth=1.5)
    ax.add_patch(rect_tier3)
    ax.text(67.0, 46.5, "Tier 3: Risk Engine & DRL\n(Autonomous Decision)", 
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="#7D6608")

    # Context Fusion & Risk Engine
    box_risk = patches.FancyBboxPatch((58.5, 25), 17, 18, boxstyle="round,pad=0.5",
                                     facecolor="#FFFFFF", edgecolor="#F39C12", linewidth=1.2)
    ax.add_patch(box_risk)
    ax.text(67.0, 40.5, "Identity Risk Resolver", ha="center", fontsize=8, fontweight="bold", color="#B7950B")
    ax.text(67.0, 36.5, "IP/MAC-to-Role Mapping", ha="center", fontsize=7, color="#2C3E50")
    ax.text(67.0, 33.0, r"$R = \sum w_i \cdot \text{Score}_i$", ha="center", fontsize=7.5, color="#154360")
    ax.text(67.0, 29.5, "Context-Aware Scoring", ha="center", fontsize=7, color="#2C3E50")
    ax.text(67.0, 26.5, "Threshold Trigger: R ≥ 0.70", ha="center", fontsize=6.8, fontweight="bold", color="#C0392B")

    # DRL (DQN) Controller
    box_dqn = patches.FancyBboxPatch((58.5, 4), 17, 18.5, boxstyle="round,pad=0.5",
                                    facecolor="#FFFFFF", edgecolor="#F39C12", linewidth=1.2)
    ax.add_patch(box_dqn)
    ax.text(67.0, 19.5, "Deep Q-Network (DQN)", ha="center", fontsize=8, fontweight="bold", color="#B7950B")
    ax.text(67.0, 16.0, r"State $s_t \in \mathbb{R}^6$ (CPU, Link, RE)", ha="center", fontsize=6.8, color="#2C3E50")
    ax.text(67.0, 13.0, r"Action Selection: $\arg\max_a Q$", ha="center", fontsize=7, color="#154360")
    ax.text(67.0, 9.8, "Experience Replay Memory", ha="center", fontsize=7, color="#2C3E50")
    ax.text(67.0, 6.5, "Reward: Utility - Cost", ha="center", fontsize=7, fontweight="bold", color="#27AE60")

    # ── Tier 4: Closed-Loop SDN Containment ──────────────────────────────────
    rect_tier4 = patches.FancyBboxPatch((80, 2), 18, 47, boxstyle="round,pad=0.8",
                                        facecolor=c_sdn_bg, edgecolor=c_sdn_border, linewidth=1.5)
    ax.add_patch(rect_tier4)
    ax.text(89.0, 46.5, "Tier 4: SDN Actuation\n(Zero-Trust Enforce)", 
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="#196F3D")

    # SOAR Playbooks Box
    box_sdn = patches.FancyBboxPatch((81.5, 14), 15, 29, boxstyle="round,pad=0.5",
                                    facecolor="#FFFFFF", edgecolor="#27AE60", linewidth=1.2)
    ax.add_patch(box_sdn)
    ax.text(89.0, 40.0, "SDN Controller", ha="center", fontsize=8, fontweight="bold", color="#1E8449")
    ax.text(89.0, 36.5, "OpenDaylight / Ryu", ha="center", fontsize=7, color="#566573")
    ax.text(89.0, 32.5, "Closed-Loop Actions:", ha="center", fontsize=7.2, fontweight="bold", color="#2C3E50")
    ax.text(89.0, 28.5, "• a₀: Monitor / Log", ha="center", fontsize=6.8, color="#2C3E50")
    ax.text(89.0, 25.0, "• a₁: Rate-Limit (Meter)", ha="center", fontsize=6.8, color="#2C3E50")
    ax.text(89.0, 21.5, "• a₂: Quarantine (VLAN)", ha="center", fontsize=6.8, color="#D35400")
    ax.text(89.0, 18.0, "• a₃: Drop (Flowspec)", ha="center", fontsize=6.8, fontweight="bold", color="#C0392B")
    ax.text(89.0, 15.0, "Actuation: ~4.15 ms", ha="center", fontsize=6.8, fontweight="bold", color="#196F3D")

    # OpenFlow Switches Box
    box_sw = patches.FancyBboxPatch((81.5, 4), 15, 8.5, boxstyle="round,pad=0.4",
                                   facecolor="#FFFFFF", edgecolor="#52BE80", linewidth=1.0)
    ax.add_patch(box_sw)
    ax.text(89.0, 9.5, "Campus Data Plane", ha="center", fontsize=7.5, fontweight="bold", color="#27AE60")
    ax.text(89.0, 6.0, "OpenFlow 1.3 Switches", ha="center", fontsize=7, color="#566573")

    # ── Connecting Arrows ───────────────────────────────────────────────────
    # Edge -> FedAvg (weights)
    ax.annotate("", xy=(34, 34), xytext=(27, 34),
                arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.6))
    ax.text(30.5, 36.0, "Weights", fontsize=6.5, ha="center", color="#C0392B", fontweight="bold")

    # Edge -> Redis (Alerts)
    ax.annotate("", xy=(34, 13), xytext=(27, 13),
                arrowprops=dict(arrowstyle="->", color="#2980B9", lw=1.6))
    ax.text(30.5, 10.5, "RE > τ", fontsize=6.5, ha="center", color="#2980B9", fontweight="bold")

    # Redis -> Risk Engine
    ax.annotate("", xy=(58.5, 34), xytext=(52, 16),
                arrowprops=dict(arrowstyle="->", color="#E67E22", lw=1.6))
    ax.text(54.0, 26.5, "Events", fontsize=6.5, ha="center", color="#E67E22", fontweight="bold")

    # Risk Engine -> DQN
    ax.annotate("", xy=(67, 22.5), xytext=(67, 25),
                arrowprops=dict(arrowstyle="->", color="#D4AC0D", lw=1.6))
    ax.text(70.5, 23.5, "R ≥ 0.70", fontsize=6.5, ha="left", color="#D4AC0D", fontweight="bold")

    # DQN -> SDN Controller
    ax.annotate("", xy=(81.5, 28), xytext=(75.5, 13),
                arrowprops=dict(arrowstyle="->", color="#27AE60", lw=1.6))
    ax.text(77.5, 22.0, "Action a*", fontsize=6.5, ha="center", color="#27AE60", fontweight="bold")

    # SDN -> Data Plane
    ax.annotate("", xy=(89, 12.5), xytext=(89, 14),
                arrowprops=dict(arrowstyle="->", color="#1E8449", lw=1.4))

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_architecture.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Cleanly generated Fig 1 Architecture Diagram at {OUTPUT_DIR / 'fig1_architecture.png'}")

if __name__ == "__main__":
    generate_arch_diagram()
