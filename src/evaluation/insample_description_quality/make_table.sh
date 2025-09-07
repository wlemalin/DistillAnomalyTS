#!/usr/bin/env bash

# --- Inputs (edit these paths/names) ---
T="src/evaluation/insample_description_quality/qwen_txt_metrics_desc.json"                     # Teacher/OpenAI 4o
Q1="src/evaluation/insample_description_quality/qwen_txt_metrics_desc.json"            # Qwen variant 1
Q1_NAME="Qwen2.5-1.5B-txt"                       # label for CSV header

Q2="src/evaluation/insample_description_quality/qwen_vl_ts_metrics_desc.json"         # Qwen variant 2 (NEW)
Q2_NAME="Qwen2.5-VL"                         # label for CSV header

OUT="results/insample_description_metrics.csv"                 # output CSV

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

# --- EXACT same aggregations you already run, now for Q1 and Q2 ---

echo "Teacher ROUGE:"
jq 'group_by(.pattern_type)
    | map({pattern_type: .[0].pattern_type,
           avg_rouge: (map(.teacher_rougeL) | add / length)})' "$T" \
  | tee "$tmpdir/t_rouge.json"

echo "Qwen ROUGE:"
jq 'group_by(.pattern_type)
    | map({pattern_type: .[0].pattern_type,
           avg_rouge: (map(.qwen_rougeL) | add / length)})' "$Q1" \
  | tee "$tmpdir/q1_rouge.json"

echo "Qwen2 ROUGE:"
jq 'group_by(.pattern_type)
    | map({pattern_type: .[0].pattern_type,
           avg_rouge: (map(.qwen_rougeL) | add / length)})' "$Q2" \
  | tee "$tmpdir/q2_rouge.json"

echo "Teacher BLEURT:"
jq 'group_by(.pattern_type)
    | map({pattern_type: .[0].pattern_type,
           avg_bleurt: (map(.teacher_bleurt_score) | add / length)})' "$T" \
  | tee "$tmpdir/t_bleurt.json"

echo "Qwen BLEURT:"
jq 'group_by(.pattern_type)
    | map({pattern_type: .[0].pattern_type,
           avg_bleurt: (map(.qwen_bleurt_score) | add / length)})' "$Q1" \
  | tee "$tmpdir/q1_bleurt.json"

echo "Qwen2 BLEURT:"
jq 'group_by(.pattern_type)
    | map({pattern_type: .[0].pattern_type,
           avg_bleurt: (map(.qwen_bleurt_score) | add / length)})' "$Q2" \
  | tee "$tmpdir/q2_bleurt.json"

echo "Teacher BERTscore:"
jq 'group_by(.pattern_type)
    | map({pattern_type: .[0].pattern_type,
           avg_bertscore: (map(.teacher_bertscore) | add / length)})' "$T" \
  | tee "$tmpdir/t_bert.json"

echo "Qwen BERTscore:"
jq 'group_by(.pattern_type)
    | map({pattern_type: .[0].pattern_type,
           avg_bertscore: (map(.qwen_bertscore) | add / length)})' "$Q1" \
  | tee "$tmpdir/q1_bert.json"

echo "Qwen2 BERTscore:"
jq 'group_by(.pattern_type)
    | map({pattern_type: .[0].pattern_type,
           avg_bertscore: (map(.qwen_bertscore) | add / length)})' "$Q2" \
  | tee "$tmpdir/q2_bert.json"

# --- Merge to CSV (safe heredoc; model names passed via --arg) ---

cat > "$tmpdir/merge.jq" <<'JQ'
def r4:
  if type=="number" then ((. * 10000 | round) / 10000) else null end;

def to_map(arr; prop; newkey):
  arr | map({ (.pattern_type): { (newkey): .[prop] } }) | add;

# Input order (9 arrays with -s):
# 0:t_rouge, 1:q1_rouge, 2:q2_rouge, 3:t_bleurt, 4:q1_bleurt, 5:q2_bleurt, 6:t_bert, 7:q1_bert, 8:q2_bert
(to_map(.[0]; "avg_rouge";     "t_rouge"))  as $tR |
(to_map(.[1]; "avg_rouge";     "q1_rouge")) as $q1R |
(to_map(.[2]; "avg_rouge";     "q2_rouge")) as $q2R |
(to_map(.[3]; "avg_bleurt";    "t_bleurt")) as $tB |
(to_map(.[4]; "avg_bleurt";    "q1_bleurt")) as $q1B |
(to_map(.[5]; "avg_bleurt";    "q2_bleurt")) as $q2B |
(to_map(.[6]; "avg_bertscore"; "t_bert"))   as $tS |
(to_map(.[7]; "avg_bertscore"; "q1_bert"))  as $q1S |
(to_map(.[8]; "avg_bertscore"; "q2_bert"))  as $q2S |

($tR * $q1R * $q2R * $tB * $q1B * $q2B * $tS * $q1S * $q2S) as $m |
($m | to_entries | sort_by(.key)) as $rows |
(
  ["Anomaly Type",
   "Teacher ROUGE","Teacher BLEURT","Teacher BERTScore",
   ($q1_name + " ROUGE"),($q1_name + " BLEURT"),($q1_name + " BERTScore"),
   ($q2_name + " ROUGE"),($q2_name + " BLEURT"),($q2_name + " BERTScore")],
  ($rows[] | [
    .key,
    (.value.t_rouge  | r4),
    (.value.t_bleurt | r4),
    (.value.t_bert   | r4),
    (.value.q1_rouge  | r4),
    (.value.q1_bleurt | r4),
    (.value.q1_bert   | r4),
    (.value.q2_rouge  | r4),
    (.value.q2_bleurt | r4),
    (.value.q2_bert   | r4)
  ])
)
| @csv
JQ

jq -r -s \
  --arg q1_name "$Q1_NAME" \
  --arg q2_name "$Q2_NAME" \
  -f "$tmpdir/merge.jq" \
  "$tmpdir/t_rouge.json"  "$tmpdir/q1_rouge.json" "$tmpdir/q2_rouge.json" \
  "$tmpdir/t_bleurt.json" "$tmpdir/q1_bleurt.json" "$tmpdir/q2_bleurt.json" \
  "$tmpdir/t_bert.json"   "$tmpdir/q1_bert.json"   "$tmpdir/q2_bert.json" \
  > "$OUT"



echo "Wrote $OUT"
