$ErrorActionPreference = 'Stop'
python -m pip install -r requirements.txt
python evaluation/run_verified_federated_baseline.py
pdflatex -halt-on-error -interaction=nonstopmode rivf2026.tex
pdflatex -halt-on-error -interaction=nonstopmode rivf2026.tex
New-Item -ItemType Directory -Force output\pdf | Out-Null
Copy-Item rivf2026.pdf output\pdf\rivf2026_submission.pdf -Force
Write-Host 'Reproduced metrics and output/pdf/rivf2026_submission.pdf'
