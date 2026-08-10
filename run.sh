#!/bin/bash
# run.sh - Master One-Click Reproducibility Script for IEEE RIVF 2026

set -e

echo "=== [1/3] Setting up Python Virtual Environment ==="
python3 -m venv .venv || python -m venv .venv
source .venv/bin/activate || source .venv/Scripts/activate

echo "=== [2/3] Installing Dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== [3/3] Compiling IEEE RIVF 2026 Camera-Ready PDF ==="
pdflatex -interaction=nonstopmode rivf2026.tex
pdflatex -interaction=nonstopmode rivf2026.tex

echo "=== SUCCESS: rivf2026.pdf compiled successfully! ==="
