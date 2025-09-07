#!/usr/bin/env python3
# Example usage:
#   python evaluate_descriptions.py \
#       --gt UCR_gt_descr.csv \
#       ../vl_outsample_results/out_all_1440.jsonl \
#       TEST_vl_all_with_metrics.json
#
# Minimal (uses default GT path: UCR_gt_descr.csv):
#   python evaluate_descriptions.py out_all_1440.jsonl results_with_metrics.json

import argparse
import csv
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
from rouge_score import rouge_scorer
import bert_score


def load_ground_truth_csv(gt_csv_path: str):
    """
    Load ground-truth descriptions from a CSV with columns:
      series_id, ground_truth_description

    Handles leading spaces after commas and preserves leading zeros in IDs (e.g., "006").
    Also builds a numeric lookup (e.g., 6) so "6" will match "006".
    """
    gt_map = {}       # string id -> description (preserves leading zeros)
    gt_num_map = {}   # int id -> description (for numeric matching)

    with open(gt_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        # Ensure required columns exist
        fieldnames = [fn.strip() for fn in (reader.fieldnames or [])]
        if "series_id" not in fieldnames or "ground_truth_description" not in fieldnames:
            raise ValueError(
                f"CSV must contain 'series_id' and 'ground_truth_description' columns. "
                f"Found: {fieldnames}"
            )

        for row in reader:
            # DictReader respects skipinitialspace for values too
            sid_raw = (row.get("series_id") or "").strip()
            desc = (row.get("ground_truth_description") or "").strip()
            if not sid_raw or not desc:
                continue

            gt_map[sid_raw] = desc
            # If series_id is numeric (possibly with leading zeros), keep an int mapping too
            num_str = sid_raw.lstrip("0")
            if num_str == "" and sid_raw != "":  # e.g., "000" -> 0
                num_str = "0"
            if num_str.isdigit():
                gt_num_map[int(num_str)] = desc

    return gt_map, gt_num_map


def lookup_ref(qid, gt_map, gt_num_map):
    """
    Try to find the reference description by:
    1) exact string match (preserving leading zeros),
    2) integer match (so '6' matches '006' and vice versa).
    """
    qid_str = str(qid).strip()
    if not qid_str:
        return None
    if qid_str in gt_map:
        return gt_map[qid_str]
    num_str = qid_str.lstrip("0")
    if num_str == "" and qid_str != "":
        num_str = "0"
    if num_str.isdigit():
        qid_int = int(num_str)
        if qid_int in gt_num_map:
            return gt_num_map[qid_int]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated anomaly descriptions against CSV ground truth using BLEURT, ROUGE-L, BERTScore, and NLI."
    )
    parser.add_argument(
        "input",
        help="Path to the model output JSONL file to evaluate (each line is a JSON object).",
    )
    parser.add_argument(
        "output",
        help="Path to the output JSON file where metrics will be saved.",
    )
    parser.add_argument(
        "-g",
        "--gt",
        default="UCR_gt_desc.csv",
        help="Path to the CSV with columns: series_id, ground_truth_description (default: UCR_gt_desc.csv).",
    )
    args = parser.parse_args()

    # Load models (once)
    # BLEURT
    bleurt_model = BleurtForSequenceClassification.from_pretrained("lucadiliello/BLEURT-20-D12")
    bleurt_tokenizer = BleurtTokenizer.from_pretrained("lucadiliello/BLEURT-20-D12")
    bleurt_model.eval()

    # NLI (BART MNLI)
    nli_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
    nli_model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
    nli_model.eval()

    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # Load ground truth from CSV
    gt_map, gt_num_map = load_ground_truth_csv(args.gt)

    # Load and evaluate generated outputs
    results = []
    with open(args.input, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"❌ Generated JSON decode error on line {idx}: {e}")
                continue

            qid = item.get("id")
            ref = lookup_ref(qid, gt_map, gt_num_map)
            if not ref:
                continue  # skip if no ground truth description

            hyp = item.get("output_json", {}).get("anomalies", [{}])[0].get("description", "")
            if not hyp:
                continue

            out = item.copy()
            out["ref"] = ref

            # BLEURT
            inputs = bleurt_tokenizer([ref], [hyp], padding="longest", return_tensors="pt")
            with torch.no_grad():
                out["bleurt_score"] = bleurt_model(**inputs).logits.flatten().item()

            # ROUGE-L
            out["rougeL"] = rouge.score(ref, hyp)["rougeL"].fmeasure

            # BERTScore (CPU for portability)
            _, _, F1 = bert_score.score([hyp], [ref], lang="en", verbose=False, device="cpu")
            out["bertscore"] = F1[0].item()

            # NLI
            nli_inputs = nli_tokenizer(ref, hyp, return_tensors="pt", truncation=True)
            with torch.no_grad():
                logits = nli_model(**nli_inputs).logits
                probs = F.softmax(logits, dim=-1).squeeze().tolist()
            out["nli"] = dict(zip(["entailment", "neutral", "contradiction"], probs))

            results.append(out)

    # Save evaluated results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Evaluation complete. Saved to {args.output}")


if __name__ == "__main__":
    main()
