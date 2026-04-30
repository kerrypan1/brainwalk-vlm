#!/usr/bin/env python3
"""
File Summary
------------
Compute per-ID clip-average predictions, compare against GT from a CSV, and
report mean absolute error (MAE) across IDs.

Folder layout:
- ROOT_DIR contains subfolders named like: "00(id)_2" (e.g., "00123_2")
- Each subfolder contains 3 prediction text files (assumed *.txt), each containing
  key:value numeric pairs like:
    GT: speed: 2, assistive_device: 4, imbalance: 2, gait_deviation: 3.0,
        deviation_outside_walkway: 3, fga_score: 3
  (Prefix can be GT: or LABEL:; we just parse key:value pairs.)

GT CSV:
- CSV_PATH points to a CSV with columns:
  id,speed,assistive_device,imbalance,gait_deviation,deviation_outside_walkway,fga_score

What this script does:
1) For each ID folder: parse up to 3 clip prediction files and average each field.
2) Normalize IDs by stripping leading zeros (e.g., "0018" -> "18") for matching.
3) Compute per-field MAE across IDs and an overall MAE.
"""

from __future__ import annotations

import re
import csv
from pathlib import Path
from typing import Dict, List


# =======================
# HARD-CODE PATHS HERE
# =======================
ROOT_DIR = Path("./vlm_output/intern/clips_fps_4.0_length_4.0")  # contains 00(id)_2 subfolders
CSV_PATH = Path("./gt.csv")                                     # GT CSV
# =======================


FIELDS = [
    "speed",
    "assistive_device",
    "imbalance",
    "gait_deviation",
    "deviation_outside_walkway",
    "fga_score",
]

PAIR_RE = re.compile(r"([a-zA-Z_]+)\s*:\s*([-+]?\d+(?:\.\d+)?)")


def mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def norm_id(s: str) -> str:
    """Normalize ID strings by stripping leading zeros."""
    # Handles "0018" -> "18", "18" -> "18"
    return str(int(s))


def parse_scores_from_text(text: str) -> Dict[str, float]:
    """
    Extract key:value numeric pairs like 'speed: 2' from a text blob.
    Returns only known FIELDS.
    """
    found: Dict[str, float] = {}
    for k, v in PAIR_RE.findall(text):
        k = k.strip()
        if k in FIELDS:
            found[k] = float(v)
    return found


def read_pred_from_file(p: Path) -> Dict[str, float]:
    text = p.read_text(encoding="utf-8", errors="ignore")
    scores = parse_scores_from_text(text)
    missing = [f for f in FIELDS if f not in scores]
    if missing:
        raise ValueError(f"Missing fields {missing} in file: {p}")
    return scores


def extract_id_from_folder_name(name: str) -> str:
    """
    Folder names are like '00(id)_2'. We interpret ID as the first digit group.
    Example: '00123_2' -> '00123' -> normalized to '123'
    """
    digits = re.findall(r"\d+", name)
    if not digits:
        raise ValueError(f"Could not parse id from folder name: {name}")
    return norm_id(digits[0])


def load_gt_csv(csv_path: Path) -> Dict[str, Dict[str, float]]:
    """Load GT CSV into dict: normalized_id -> {field: float}."""
    gt: Dict[str, Dict[str, float]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing_cols = [c for c in (["id"] + FIELDS) if c not in (reader.fieldnames or [])]
        if missing_cols:
            raise ValueError(f"CSV missing required columns: {missing_cols}. Found: {reader.fieldnames}")

        for row in reader:
            raw = str(row["id"]).strip()
            if raw == "":
                continue
            sid = norm_id(raw)
            gt[sid] = {k: float(row[k]) for k in FIELDS}
    return gt


def main() -> None:
    if not ROOT_DIR.exists():
        raise SystemExit(f"ROOT_DIR does not exist: {ROOT_DIR}")
    if not CSV_PATH.exists():
        raise SystemExit(f"CSV_PATH does not exist: {CSV_PATH}")

    gt_map = load_gt_csv(CSV_PATH)

    # Gather predictions per ID from output subfolders.
    pred_avg_by_id: Dict[str, Dict[str, float]] = {}
    skipped: List[str] = []

    for sub in sorted([p for p in ROOT_DIR.iterdir() if p.is_dir()]):
        try:
            sid = extract_id_from_folder_name(sub.name)
        except Exception as e:
            skipped.append(f"{sub.name} (id parse error: {e})")
            continue

        txts = sorted(sub.glob("*.txt"))
        if len(txts) == 0:
            # If your prediction files have no .txt extension, change to: sub.glob("*")
            skipped.append(f"{sub.name} (no .txt files found)")
            continue

        # Evaluate up to first three clips per sample folder.
        used = txts[:3]
        try:
            per_clip = [read_pred_from_file(p) for p in used]
        except Exception as e:
            skipped.append(f"{sub.name} (parse error: {e})")
            continue

        # Average across clips
        avg = {k: mean([d[k] for d in per_clip]) for k in FIELDS}
        pred_avg_by_id[sid] = avg

    # Match prediction IDs and GT IDs after normalization.
    common_ids = sorted(set(pred_avg_by_id.keys()) & set(gt_map.keys()))
    missing_gt = sorted(set(pred_avg_by_id.keys()) - set(gt_map.keys()))
    missing_pred = sorted(set(gt_map.keys()) - set(pred_avg_by_id.keys()))

    if not common_ids:
        raise SystemExit(
            "No overlapping IDs between predictions and GT CSV.\n"
            f"Example pred IDs: {list(pred_avg_by_id.keys())[:5]}\n"
            f"Example GT IDs: {list(gt_map.keys())[:5]}"
        )

    # Per-field MAE across IDs
    mae_by_field: Dict[str, float] = {}
    for f in FIELDS:
        errs = [abs(pred_avg_by_id[i][f] - gt_map[i][f]) for i in common_ids]
        mae_by_field[f] = mean(errs)

    # Overall MAE (average of absolute errors over all fields and IDs)
    all_errs: List[float] = []
    for i in common_ids:
        for f in FIELDS:
            all_errs.append(abs(pred_avg_by_id[i][f] - gt_map[i][f]))
    overall_mae = mean(all_errs)

    # Print report
    print("=== Clip-average vs GT error report ===")
    print(f"Pred folder: {ROOT_DIR}")
    print(f"GT CSV:      {CSV_PATH}")
    print(f"IDs used:    {len(common_ids)}")
    if missing_gt:
        print(f"Pred IDs missing in GT CSV: {len(missing_gt)} (e.g., {missing_gt[:5]})")
    if missing_pred:
        print(f"GT IDs missing predictions: {len(missing_pred)} (e.g., {missing_pred[:5]})")
    if skipped:
        print(f"Skipped folders: {len(skipped)} (e.g., {skipped[:5]})")

    print("\nMAE by field (averaged over IDs):")
    for f in FIELDS:
        print(f"  {f:24s} {mae_by_field[f]:.4f}")

    print(f"\nOverall MAE (all fields, all IDs): {overall_mae:.4f}")


if __name__ == "__main__":
    main()
