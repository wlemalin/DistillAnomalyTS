#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute per-file averages for BLEURT, ROUGE-L, and BERTScore from JSON files in a directory."
    )
    p.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Directory containing .json files (each file = one LLM).",
    )
    p.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    return p.parse_args()


def iter_key_numbers(obj: Any, target: str) -> Iterable[float]:
    """
    Recursively traverse obj (dict/list/values) and yield numeric values stored under key == target.
    Accepts int/float, and numeric strings that can be parsed as float.
    """
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            # If key present, try to coerce to float
            if target in cur:
                val = cur[target]
                f = _to_float(val)
                if f is not None:
                    yield f
            # Continue traversal
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
        # primitives: nothing to do


def _to_float(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.strip())
        except Exception:
            return None
    return None


def average(nums: List[float]) -> Optional[float]:
    return (sum(nums) / len(nums)) if nums else None


def fmt4(x: Optional[float]) -> str:
    return f"{x:.4f}" if x is not None else "NA"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_file_metrics(path: Path) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns (bleurt_avg, rougeL_avg, bertscore_avg) for this JSON file,
    searching for keys at any depth: 'bleurt_score', 'rougeL', 'bertscore'.
    """
    data = load_json(path)
    bleurts = list(iter_key_numbers(data, "bleurt_score"))
    rouges  = list(iter_key_numbers(data, "rougeL"))
    berts   = list(iter_key_numbers(data, "bertscore"))
    return average(bleurts), average(rouges), average(berts)


def main() -> None:
    args = parse_args()
    if not args.dir.is_dir():
        raise SystemExit(f"Not a directory: {args.dir}")

    files = sorted([p for p in args.dir.iterdir() if p.suffix.lower() == ".json"])
    if not files:
        raise SystemExit(f"No .json files found in: {args.dir}")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Model", "BLEURT", "ROUGE-L", "BERTScore"])
        for path in files:
            model = path.stem
            try:
                bleurt_avg, rouge_avg, bert_avg = collect_file_metrics(path)
            except Exception as e:
                # If a file is malformed JSON, write NA row but keep going
                bleurt_avg = rouge_avg = bert_avg = None
            w.writerow([model, fmt4(bleurt_avg), fmt4(rouge_avg), fmt4(bert_avg)])

    print(f"Wrote CSV -> {args.csv}")


if __name__ == "__main__":
    main()

