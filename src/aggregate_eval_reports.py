#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Set


def run_eval(evaluator: Path, jsonl: Path, T0: int, T1: int) -> Dict[str, Any]:
    """
    Runs the evaluator script on one JSONL and returns the parsed JSON report.
    Assumes evaluator supports: evaluator <jsonl> --T0 ... --T1 ... --out <file>
    """
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "report.json"
        cmd = [
            sys.executable, str(evaluator),
            str(jsonl),
            "--T0", str(T0),
            "--T1", str(T1),
            "--out", str(out_path),
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            raise RuntimeError(
                f"Evaluator failed for {jsonl}.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
            )
        return json.loads(out_path.read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser(
        description="Batch collect Affiliation F1 into a CSV (rows=pattern_types + OVERALL, cols=input files)."
    )
    p.add_argument(
        "--evaluator",
        type=Path,
        required=True,
        help="Path to your single-file evaluator script (the one you run on a JSONL).",
    )
    p.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    p.add_argument(
        "--T0", type=int, default=29, help="Domain start (inclusive). Default 29",
    )
    p.add_argument(
        "--T1", type=int, default=300, help="Domain end (exclusive). Default 300",
    )
    p.add_argument(
        "jsonls",
        nargs="+",
        type=Path,
        help="List of JSONL files to evaluate (each becomes a CSV column).",
    )
    args = p.parse_args()

    # Run evaluator for each JSONL
    reports: List[Dict[str, Any]] = []
    col_names: List[str] = []
    for j in args.jsonls:
        reports.append(run_eval(args.evaluator, j, args.T0, args.T1))
        col_names.append(j.name)

    # Collect all pattern types across reports
    all_pts: Set[str] = set()
    for rep in reports:
        all_pts.update(rep.get("by_pattern_type", {}).keys())

    # Stable ordering: alphabetical pattern types, then OVERALL
    row_order = sorted(all_pts)
    OVERALL = "OVERALL"
    row_order.append(OVERALL)

    # Build CSV
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pattern_type"] + col_names)

        def fmt(x: Any) -> str:
            if x is None:
                return "NA"
            try:
                return f"{float(x):.4f}"
            except Exception:
                return str(x)

        for pt in row_order:
            row = [pt]
            for rep in reports:
                if pt == OVERALL:
                    v = rep.get("overall", {}).get("f1")
                else:
                    v = rep.get("by_pattern_type", {}).get(pt, {}).get("f1")
                row.append(fmt(v))
            w.writerow(row)

    print(f"Wrote CSV -> {args.csv}")


if __name__ == "__main__":
    main()
