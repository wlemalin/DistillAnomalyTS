import argparse
import json
import logging
import os
import re
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from sentence_transformers import SentenceTransformer, util
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, Trainer, TrainerCallback,
                          TrainingArguments)

# === ARGUMENTS ===
parser = argparse.ArgumentParser(
    description="LoRA fine-tuning script with semantic evaluation")
parser.add_argument("--dataset_path", type=str, required=True,
                    help="Path to the dataset JSONL file")
parser.add_argument("--output_dir", type=str, required=True,
                    help="Directory where training outputs will be saved")
parser.add_argument("--out_json", type=str, required=True,
                    help="Directory where eval outputs will be saved")
args = parser.parse_args()

# === CONFIGURATION ===
HOME = os.path.expanduser("~")

MODEL_NAME = os.path.join(HOME, "hf_models", "qwen2.5-1.5b")
DATASET_PATH = args.dataset_path

# output_dir fallback
if args.output_dir:
    OUTPUT_DIR = args.output_dir
else:
    model_name_parts = MODEL_NAME.split('/')
    model_short_name = model_name_parts[-1] if model_name_parts else "unknown_model"
    OUTPUT_DIR = f"./{model_short_name.lower()}-lora-output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Training outputs saved in:", OUTPUT_DIR)


# === LOADING JSONL DATASET ===
def load_jsonl_dataset(path):
    with open(path, 'r') as f:
        lines = [json.loads(l) for l in f]
    return Dataset.from_list(lines)


raw_dataset = load_jsonl_dataset(DATASET_PATH)
split_dataset = raw_dataset.train_test_split(test_size=0.10)
dataset = DatasetDict({
    "train": split_dataset["train"],
    "test": split_dataset["test"]
})

print(f"Train size: {len(dataset['train'])}")
print(f"Test size: {len(dataset['test'])}")

# === TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# === PREPROCESSING ===
def tokenize(example):
    prompt = example["input"]
    response = example["output"]

    # Tokenize prompt and answer
    prompt_ids = tokenizer(prompt, truncation=False)["input_ids"]
    response_ids = tokenizer(response, truncation=True,
                             max_length=256)["input_ids"]

    max_total_len = 2048
    max_prompt_len = max_total_len - len(response_ids)
    prompt_ids = prompt_ids[-max_prompt_len:]

    input_ids = prompt_ids + response_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + response_ids

    # Padding if needed
    pad_len = max_total_len - len(input_ids)
    if pad_len > 0:
        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        labels += [-100] * pad_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


print("Tokenisation des données...")
tokenized_dataset = dataset.map(
    tokenize, remove_columns=dataset["train"].column_names)
print("Tokenisation terminée.")

# === MODEL ===
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True
)

# === CONFIG LORA ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# === SEMANTIC SIMILARITY MODEL ===
model_st = SentenceTransformer(os.path.join(HOME, "hf_models", "all_MiniLM-L6-v2"))


# ===  SEMANTIC EVALUATION ===
def compute_semantic_similarity(model, tokenizer, dataset, output_file=None):
    model.eval()
    examples = dataset.to_list()
    inputs = [ex["input"] for ex in examples]
    gold_outputs = [ex["output"] for ex in examples]
    gt_anoms = [ex["ground_truth"] for ex in examples]
    pattern_types = [ex.get("pattern_type") for ex in examples]

    generated_outputs = []
    for inp in inputs:
        inputs_tokenized = tokenizer(inp, return_tensors="pt").to(model.device)
        prompt_len = inputs_tokenized.input_ids.shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs_tokenized,
                max_new_tokens=128,
                do_sample=False,
                temperature=0.7
            )

        generated_only_ids = output_ids[0][prompt_len:]
        output_text = tokenizer.decode(
            generated_only_ids, skip_special_tokens=True)
        generated_outputs.append(output_text)

    emb_generated = model_st.encode(generated_outputs, convert_to_tensor=True)
    emb_gold = model_st.encode(gold_outputs, convert_to_tensor=True)
    scores = torch.nn.functional.cosine_similarity(emb_generated, emb_gold)
    avg_score = float(scores.mean())

    if output_file:
        output_data = [{
            "input": inp,
            "generated_output": gen_out,
            "teacher_output": gold_out,
            "ground_truth": gt,
            "pattern_type": p_type,
            "score": float(score)
        } for inp, gen_out, gold_out, gt, p_type, score in zip(inputs, generated_outputs, gold_outputs, gt_anoms, pattern_types, scores)]

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=4)

    model.train()
    return avg_score

# === CALLBACK FOR CHECKPOINT EVALUATION ===
class SemanticSimilarityCallback(TrainerCallback):
    def __init__(self, model, tokenizer, test_dataset, output_dir):
        self.model = model
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.output_dir = output_dir

    def on_step_end(self, args, state, control, **kwargs):
        print(f"🔄 Étape terminée : {state.global_step}")

    def on_evaluate(self, args, state, control, **kwargs):
        print(
            "\n✨ Évaluation de la similarité sémantique à la sauvegarde du checkpoint ✨\n")
        if not hasattr(self, "model") or self.model is None:
            print("Erreur : self.model est None.")
            return

        # trainer.model.eval()
        step = state.global_step
        print(f"création du fichier d'évaluation pour le checkpoint {step}")
        output_file = os.path.join(
            self.output_dir, f"evaluation_checkpoint-{step}.json")
        print(f"fichier d'évaluation : {output_file}")

        print(f"calcul de la similarité sémantique pour le checkpoint {step}")
        score = compute_semantic_similarity(
            self.model,
            self.tokenizer,
            self.test_dataset,
            output_file=output_file
        )

        # trainer.model.train()
        print(f"\n✅ Checkpoint {step}: Similarité sémantique = {score:.4f}\n")


# === TRAINING ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=7,
    learning_rate=1e-3,
    bf16=True,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    save_strategy="steps",
    eval_strategy="steps",
    eval_steps=20,
    save_steps=20,
    save_total_limit=10,
    report_to="none",
    max_steps=160,
    disable_tqdm=False
)

# Logging config
logging.basicConfig(level=logging.INFO, force=True)


# Initialise Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    callbacks=[SemanticSimilarityCallback(
        model, tokenizer, dataset["test"], OUTPUT_DIR)],
)

# Ajouter le callback d'évaluation sémantique
# semantic_callback = SemanticSimilarityCallback(tokenizer, dataset["test"], OUTPUT_DIR)
# trainer.add_callback(semantic_callback)

# Évaluation initiale avant entraînement
print("\n🔍 Évaluation avant entraînement (similarité sémantique)...")
score_before = compute_semantic_similarity(
    model, tokenizer, dataset["test"], output_file=os.path.join(OUTPUT_DIR, "evaluation_avant.json"))
print(f"Score moyen avant entraînement : {score_before:.4f}")

# Training
trainer.train()

out_file = (Path(OUTPUT_DIR).resolve().parents[1] /
            "post_training_eval" /
            f"evaluation_apres_{args.out_json}.json")
out_file.parent.mkdir(parents=True, exist_ok=True)

# Post training evaluation
print("\n📊 Évaluation après entraînement (similarité sémantique)...")
score_after = compute_semantic_similarity(
    model, tokenizer, dataset["test"], output_file=str(out_file))
print(f"Score moyen après entraînement : {score_after:.4f}")

# Save final model
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
