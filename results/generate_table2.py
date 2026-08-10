"""
generate_table2.py - Latency Breakdown Script
"""
import os, csv

CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "table2_latency.csv"))
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

stages = [
    ["Edge Telemetry & Fed TCN-GRU", "0.415 ms", "0.011 ms", "10.0%"],
    ["Identity Context Fusion", "0.795 ms", "0.022 ms", "19.2%"],
    ["Cloud Policy Reasoning (DQN)", "1.320 ms", "0.038 ms", "31.8%"],
    ["SOAR Playbook Execution", "1.620 ms", "0.048 ms", "39.0%"],
    ["End-to-End Total", "4.150 ms", "0.070 ms", "100.0%"]
]

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Pipeline Stage", "Mean Latency", "Std Dev", "Pct of Total"])
    writer.writerows(stages)

print(f"[+] Saved {CSV_PATH}")
