#!/usr/bin/env bash

# Add missing files to checkpoint to make it usable

# Usage check
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 CHECKPOINT_BASE_DIR" >&2
  exit 1
fi

checkpoint_base="${1%/}"  # strip trailing slash if present

if [[ ! -d "$checkpoint_base" ]]; then
  echo "Checkpoint base directory not found: $checkpoint_base" >&2
  exit 1
fi

# Find latest checkpoint-X directory by numeric X
latest_checkpoint="$(
  find "$checkpoint_base" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null \
    | sed -E 's/.*checkpoint-([0-9]+)/\1 &/' \
    | sort -n \
    | awk '{print $2}' \
    | tail -n 1
)"

if [[ -z "${latest_checkpoint:-}" ]]; then
  echo "No checkpoint directories found in $checkpoint_base" >&2
  exit 1
fi

echo "Latest checkpoint found: $latest_checkpoint"

# Source folder for files
src_dir="$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean"

# Files to copy
files=(
  "chat_template.json"
  "merges.txt"
  "preprocessor_config.json"
  "tokenizer.json"
  "tokenizer_config.json"
  "vocab.json"
)

# Copy files
for file in "${files[@]}"; do
  src="$src_dir/$file"
  if [[ -f "$src" ]]; then
    cp "$src" "$latest_checkpoint/"
    echo "Copied $file to $latest_checkpoint/"
  else
    echo "Warning: $file not found in $src_dir" >&2
  fi
done
