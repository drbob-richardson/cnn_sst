#!/usr/bin/env python3
"""
collect_results.py
-------------------------------------------------
Scan the output folders created by learn_pixel_masks_loop_serial.py and
aggregate metrics (runtime, accuracy, precision, recall, F1, AUROC, AUPRC)
into tidy CSV/JSON summaries. Also prints a quick leaderboard to stdout.

Assumptions:
- Results are stored under a root like: results/lead_{lead}_{gate}/
- Each combo writes a metrics file: metrics_lead{lead}_{gate}.txt
- The metrics file contains lines or tokens of the form "key: value"
  (robust to missing newlines—parses "k1: v1k2: v2 ..." too).

Outputs:
- results_summary.csv                (one row per (lead, gate))
- results_by_gate.csv               (aggregate by gate, mean across leads)
- results_by_lead.csv               (aggregate by lead, mean across gates)
- results_summary.json              (JSON dump of per-run rows)
- Also prints a leaderboard (sorted by F1 desc) to stdout.
"""

import re
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

# ------------------------------ Parsing ----------------------------------

_METRIC_KEYS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auroc",
    "auprc",
    "runtime_sec",
]

def parse_metrics_text(text: str) -> Dict[str, float]:
    """
    Parse a metrics file that's expected to have entries like 'key: value'.
    Robust to missing newlines: it will find key:value pairs anywhere.
    """
    # Accept keys made of letters/underscores; value can be float or int (incl. 'nan')
    pattern = re.compile(r'([A-Za-z_]+)\s*:\s*([-+]?(\d+(\.\d+)?|\.\d+|nan|NaN|NAN))')
    metrics: Dict[str, float] = {}
    for m in pattern.finditer(text):
        k = m.group(1).lower()
        v_str = m.group(2)
        try:
            v = float(v_str) if v_str.lower() != "nan" else float("nan")
        except ValueError:
            continue
        metrics[k] = v
    return metrics

def parse_one_file(path: Path) -> Dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    metrics = parse_metrics_text(text)
    # Keep only known keys but don't fail if some are missing
    clean = {k: metrics.get(k, float("nan")) for k in _METRIC_KEYS}
    return clean

# ------------------------------ Discovery --------------------------------

def find_metric_files(root: Path) -> List[Tuple[int, str, Path]]:
    """
    Find files matching results/lead_{lead}_{gate}/metrics_lead{lead}_{gate}.txt
    Return list of (lead, gate, file_path).
    """
    out: List[Tuple[int, str, Path]] = []
    # Pattern for directory: lead_12_tile_2x2
    dir_rx = re.compile(r"^lead_(\d+)_([A-Za-z0-9_]+)$")
    for child in (root).glob("lead_*_*"):
        if not child.is_dir():
            continue
        m = dir_rx.match(child.name)
        if not m:
            continue
        lead = int(m.group(1))
        gate = m.group(2)
        # find metrics file inside
        for f in child.glob(f"metrics_lead{lead}_{gate}.txt"):
            out.append((lead, gate, f))
    return sorted(out, key=lambda t: (t[0], t[1]))

# ------------------------------ Aggregation ------------------------------

def rows_to_csv(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return
    # Ensure consistent column order
    cols = ["lead", "gate"] + _METRIC_KEYS
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in cols})

def aggregate_mean(rows: List[Dict[str, Any]], key_field: str) -> List[Dict[str, Any]]:
    """
    Aggregate numeric metrics by a single key field (e.g., 'gate' or 'lead').
    Returns list of dicts with the key_field and the mean of each metric.
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in rows:
        buckets[r[key_field]].append(r)
    agg_rows = []
    for k, group in buckets.items():
        out = {key_field: k}
        for mkey in _METRIC_KEYS:
            vals = [float(g.get(mkey, float("nan"))) for g in group if str(g.get(mkey, "")) != ""]
            # Filter out NaNs
            vals = [v for v in vals if v == v]
            out[mkey] = sum(vals)/len(vals) if vals else float("nan")
        agg_rows.append(out)
    # Sort by key_field where possible
    def sort_key(d):
        if key_field == "lead":
            try:
                return int(d["lead"])
            except Exception:
                return d["lead"]
        return d[key_field]
    return sorted(agg_rows, key=sort_key)

def print_leaderboard(rows: List[Dict[str, Any]], top_k: int = 20) -> None:
    print("\n== Leaderboard (by F1 desc) ==")
    sorted_rows = sorted(rows, key=lambda r: (float(r.get("f1", "-inf"))), reverse=True)
    for i, r in enumerate(sorted_rows[:top_k], 1):
        lead = r.get("lead")
        gate = r.get("gate")
        f1   = r.get("f1")
        acc  = r.get("accuracy")
        tsec = r.get("runtime_sec")
        print(f"{i:>2}. lead={lead:>2}  gate={gate:<8}  F1={f1:.4f}  Acc={acc:.4f}  Time={tsec:.1f}s")

# ------------------------------ Main -------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=str, default="results",
                    help="Root folder containing lead_*_* subfolders.")
    ap.add_argument("--out-dir", type=str, default=".",
                    help="Where to write CSV/JSON summaries (default: current dir).")
    ap.add_argument("--quiet", action="store_true", help="Suppress stdout leaderboard.")
    args = ap.parse_args()

    root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    found = find_metric_files(root)
    if not found:
        print(f"[collect_results] No metrics files found under: {root.resolve()}")
        print("Expected paths like: results/lead_12_tile_2x2/metrics_lead12_tile_2x2.txt")
        return 0

    rows: List[Dict[str, Any]] = []
    missing: List[str] = []

    for lead, gate, fpath in found:
        try:
            metrics = parse_one_file(fpath)
            row = {"lead": lead, "gate": gate}
            row.update(metrics)
            rows.append(row)
        except Exception as e:
            missing.append(f"{fpath}  ({e})")

    # Write per-run CSV/JSON
    csv_path = out_dir / "results_summary.csv"
    rows_to_csv(rows, csv_path)
    (out_dir / "results_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # Aggregates
    by_gate = aggregate_mean(rows, "gate")
    by_lead = aggregate_mean(rows, "lead")
    rows_to_csv(by_gate, out_dir / "results_by_gate.csv")
    rows_to_csv(by_lead, out_dir / "results_by_lead.csv")

    # Human-friendly prints
    if not args.quiet:
        print(f"\nWrote: {csv_path}")
        print(f"Wrote: {out_dir/'results_by_gate.csv'}")
        print(f"Wrote: {out_dir/'results_by_lead.csv'}")
        print_leaderboard(rows, top_k=50)

    if missing:
        print("\n[Warnings] Some files could not be parsed:")
        for m in missing:
            print("  -", m)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
