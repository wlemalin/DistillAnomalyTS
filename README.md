# DistillAnomalyTS

DistillAnomalyTS distills the anomaly‑detection and explanation skills of a large teacher model (GPT‑4o) into small student models. It generates synthetic time‑series data with controlled anomalies, annotates them via GPT‑4o, fine‑tunes LoRA‑based student models (text‑only and vision‑language) on the annotations, and evaluates them with a custom affiliation metric.

## Pipeline

1. **Generate data:** `1‑generate_data.sh` calls `create_synth_data.py` to create time series with range, point, frequency and trend anomalies (and noisy variants). It then uses `generate_csv.py` to add rolling mean/std and spectral features to each series.

2. **Annotate:** `2‑4o_annotate.sh` sends the synthetic series to GPT‑4o to produce natural‑language anomaly descriptions and interval labels. It generates multiple prompt variants and filters overlapping annotations.

3. **Train:**
   - **Text‑only:** `train_text.py` fine‑tunes a Qwen2.5‑1.5B model with LoRA (rank 8) on the annotated data. See `train_text.slurm` for examples on raw signals, signals with rolling stats and signals with local frequency.
   - **Vision‑Language:** a similar LoRA fine‑tuning process for Qwen2.5‑VL; evaluation is performed with `eval_lora_vl.py`.

4. **Evaluate:** `eval_affil.py` measures survival‑based precision and recall between predicted and ground‑truth anomaly intervals and reports F1 scores.

5. **Aggregate:** `5‑aggregate_results.sh` collects F1 scores and description‑quality metrics (BLEURT, ROUGE‑L, BERTScore) across models and datasets.

## Requirements

Python 3.10+, PyTorch, Transformers, datasets, peft, sentence‑transformers, SciPy and other common scientific packages (see `requirements.txt`). Training was realised using H100 GPU.
