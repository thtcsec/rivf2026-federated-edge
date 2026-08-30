#!/usr/bin/env bash
set -euo pipefail

python -m pip install -r requirements.txt
python evaluation/run_verified_federated_baseline.py
pdflatex -halt-on-error -interaction=nonstopmode rivf2026.tex
pdflatex -halt-on-error -interaction=nonstopmode rivf2026.tex
mkdir -p output/pdf
cp rivf2026.pdf output/pdf/rivf2026_submission.pdf
echo "Reproduced metrics and output/pdf/rivf2026_submission.pdf"
