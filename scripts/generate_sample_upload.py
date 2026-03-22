"""
generate_sample_upload.py
Produce demo upload files from the holdout test_patients.json.

Outputs (in backend/model_artifacts/):
    sample_upload.csv   — 10 patients as a flat CSV row per patient
    sample_upload.json  — same 10 patients as a JSON array in /predict format
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ARTIFACT = Path(__file__).resolve().parents[1] / "backend" / "model_artifacts"

patients = json.loads((ARTIFACT / "test_patients.json").read_text())
meta     = json.loads((ARTIFACT / "metadata.json").read_text())
genes: list[str] = meta["selected_genes"]  # 50 genes in model order

# Pick 5 Alive + 5 Dead for variety
alive  = [p for p in patients if p["actual_status"] == "Alive"][:5]
dead   = [p for p in patients if p["actual_status"] == "Dead"][:5]
sample = alive + dead

# ── CSV ───────────────────────────────────────────────────────────────────
csv_path = ARTIFACT / "sample_upload.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["age", "gender"] + genes)
    for p in sample:
        row = [p["age"], p["gender"]] + [round(p["genes"].get(g, 0.0), 6) for g in genes]
        writer.writerow(row)
print(f"CSV  -> {csv_path}  ({len(sample)} rows)")

# ── JSON array ────────────────────────────────────────────────────────────
records = [
    {
        "age":    p["age"],
        "gender": p["gender"],
        "genes":  {g: round(p["genes"].get(g, 0.0), 6) for g in genes},
    }
    for p in sample
]
json_path = ARTIFACT / "sample_upload.json"
json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
print(f"JSON -> {json_path}  ({len(records)} patients)")

# ── Preview ───────────────────────────────────────────────────────────────
p0 = sample[0]
print(f"\nRow 1 preview: age={p0['age']}, gender={p0['gender']}, "
      f"actual={p0['actual_status']}")
top3 = list(p0["genes"].items())[:3]
print(f"  First 3 genes: {top3}")
