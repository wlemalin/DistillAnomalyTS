#!/usr/bin/env python
"""
Train LoRA on Qwen2.5-VL with multi-image samples.

Run:

python vl_lora_split_jsonl.py \
  --model_dir ../hf_models/Qwen2.5-VL-3B-Instruct_clean \
  --data_json /path/to/all_samples.jsonl \
  --out_dir ./runs/qwen2_5_vl_lora_exp1 \
  --val_ratio 0.10 \
  --seed 42 \
  --epochs 3 \
  --batch_size 1 \
  --grad_accum 4 \
  --lr 1e-5 \
  --debug
"""

import argparse, json, os, sys, math, logging, warnings
from pathlib import Path
from typing import Dict, List, Sequence, Any, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model, TaskType

from datasets import Dataset as HFDataset, DatasetDict

from icecream import ic  # type: ignore
ic.configureOutput(prefix='🧊  ')


# -----------------------------
# CLI
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument
    # paths
    g("--model_dir",   required=True)
    g("--data_json",   required=True, help="Single JSONL with all samples; will be split into train/eval")
    g("--out_dir",     required=True, help="Directory to save checkpoints and LoRA adapters")
    # split params
    g("--val_ratio",   type=float, default=0.10, help="Fraction for eval/test split")
    g("--seed",        type=int,   default=42)
    # training
    g("--epochs",      type=int,   default=3)
    g("--lr",          type=float, default=1e-5)
    g("--batch_size",  type=int,   default=1)
    g("--grad_accum",  type=int,   default=4)
    g("--debug",       action="store_true")
    return p


def setup_logging(debug: bool):
    lvl = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        stream=sys.stdout,
        level=lvl,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_jsonl_dataset(path: str) -> HFDataset:
    with open(path, "r") as f:
        lines = [json.loads(l) for l in f]
    return HFDataset.from_list(lines)


# -----------------------------
# Helpers
# -----------------------------
def _even_grid_from_npatch(n: int) -> Tuple[int, int]:
    """
    Find (h, w) even integers such that h * w == n.
    Works for ViT token counts (multiple of 4). All images assumed same size.
    """
    root = int(math.isqrt(n))
    for w in range(root, 1, -1):
        if n % w == 0:
            h = n // w
            if (h % 2 == 0) and (w % 2 == 0):
                return h, w
    # fallbacks
    for w in (2, 4, 6, 8):
        if n % w == 0 and (n // w) % 2 == 0:
            return n // w, w
    raise ValueError(f"Cannot find even factors for {n} patches")


def _concat_with_truncation_keep_images_and_answer(
    img_tok_tensor: torch.Tensor,   # (1, sum_img_tokens_with_wrappers)
    prompt_ids: torch.Tensor,       # (1, P)
    answer_ids: torch.Tensor,       # (1, A)
    pad_id: int,
    max_len: int | None,
    debug: bool = False,
):
    """
    Concatenate [IMG][PROMPT][ANSWER].
    If max_len is set and exceeded, truncate ONLY from the *start of the prompt*,
    preserving ALL image tokens and ALL answer tokens.

    Returns: input_ids, attention_mask, labels
    """
    # Build preliminary
    input_ids = torch.cat([img_tok_tensor, prompt_ids, answer_ids], dim=1)  # (1, L)
    # Attention mask (1s everywhere initially)
    attn_mask = torch.ones_like(input_ids)
    # Labels: ignore image + prompt; learn on answer
    labels = torch.cat([
        torch.full_like(img_tok_tensor, -100),
        torch.full_like(prompt_ids, -100),
        answer_ids
    ], dim=1)

    if max_len is None:
        return input_ids.squeeze(0), attn_mask.squeeze(0), labels.squeeze(0)

    L = input_ids.size(1)
    if L <= max_len:
        return input_ids.squeeze(0), attn_mask.squeeze(0), labels.squeeze(0)

    # Overflow: trim from the *prompt* portion only
    overflow = L - max_len

    img_len = img_tok_tensor.size(1)
    prompt_len = prompt_ids.size(1)
    ans_len = answer_ids.size(1)

    trim_from_prompt = min(overflow, prompt_len)
    if trim_from_prompt < overflow:
        # If (image + answer) alone exceed context (should be rare): as last resort, trim from the start of answer
        # but keep at least 1 token of answer.
        remaining = overflow - trim_from_prompt
        trim_from_answer = min(remaining, max(0, ans_len - 1))
        if debug:
            logging.warning(
                "[TRUNCATION] Image+Answer exceed max context. "
                "Trimming %d tokens from answer start (kept at least 1).", trim_from_answer
            )
    else:
        trim_from_answer = 0

    # Slice components
    new_prompt_ids = prompt_ids[:, trim_from_prompt:] if trim_from_prompt > 0 else prompt_ids
    new_answer_ids = answer_ids[:, trim_from_answer:] if trim_from_answer > 0 else answer_ids

    # Rebuild
    input_ids = torch.cat([img_tok_tensor, new_prompt_ids, new_answer_ids], dim=1)
    attn_mask = torch.ones_like(input_ids)
    labels = torch.cat([
        torch.full_like(img_tok_tensor, -100),
        torch.full_like(new_prompt_ids, -100),
        new_answer_ids
    ], dim=1)

    if debug:
        kept = input_ids.size(1)
        logging.debug(
            "[TRUNCATION] L=%d > max=%d | overflow=%d | trimmed: prompt=%d answer=%d | kept=%d",
            L, max_len, overflow, trim_from_prompt, trim_from_answer, kept
        )

    return input_ids.squeeze(0), attn_mask.squeeze(0), labels.squeeze(0)


# -----------------------------
# Dataset
# -----------------------------
class QwenVLDataset(Dataset):
    """
    Accepts rec["image_paths"] as str or list[str].
    Builds:
      - pixel_values: concat of all image patch embeddings  (Σpatch, hid)
      - image_grid_thw: one row [1, H, W] per image         (n_img, 3)
      - text: chat-templated prompt (user) and answer (assistant)
      - sequence: [vision blocks (for all images)] + [prompt] + [answer]
      - controlled truncation: only prompt is truncated if needed
    """

    def __init__(
        self,
        image_processor,
        tokenizer,
        img_tok_id: int,
        merge: int,
        start_id: int | None,
        end_id: int | None,
        max_len: int | None,
        debug: bool,
        records: Sequence[Dict[str, Any]] | None = None,
        jsonl_path: str | None = None,
    ):
        if records is not None:
            self.recs = records
        elif jsonl_path is not None:
            self.recs = [json.loads(l) for l in open(jsonl_path)]
        else:
            raise ValueError("Provide either records or jsonl_path")

        self.proc = image_processor
        self.tok = tokenizer

        self.img_tok_id = img_tok_id
        self.merge = merge
        self.start_id = start_id
        self.end_id = end_id
        self.max_len = max_len
        self.debug = debug
        if debug:
            ic.configureOutput(prefix="🧊  ")

        # base for resolving relative paths (same as previous script)
        self.base = (Path(__file__).resolve().parent / "../../").resolve()

    def __len__(self):
        return len(self.recs)

    def _load_one_image(self, fname: str):
        path = Path(fname)
        if not path.is_absolute():
            path = (self.base / fname).resolve()
        img = Image.open(path).convert("RGB")
        pix = self.proc(img, return_tensors="pt")["pixel_values"].squeeze(0)  # (n_patch, hid)
        n_patch, hid = pix.shape

        # Compute even HxW grid
        h_grid, w_grid = _even_grid_from_npatch(n_patch)
        target = h_grid * w_grid
        if target > n_patch:
            pad = torch.zeros(target - n_patch, hid, dtype=pix.dtype)
            pix = torch.cat([pix, pad], 0)

        # number of <image> patch features this picture needs after spatial merge
        n_feat = (h_grid // self.merge) * (w_grid // self.merge)

        # Build the image special-token segment for this single image
        seg = torch.full((1, n_feat), self.img_tok_id, dtype=torch.long)
        if (self.start_id is not None) and (self.end_id is not None):
            seg = torch.cat([torch.tensor([[self.start_id]]), seg, torch.tensor([[self.end_id]])], dim=1)

        if self.debug:
            ic(str(path), f"n_patch={n_patch} h_grid={h_grid} w_grid={w_grid} n_feat={n_feat}")

        return pix, torch.tensor([1, h_grid, w_grid]), seg  # (n_patch, hid), (3,), (1, n_tokens_for_this_image)

    def __getitem__(self, idx):
        rec = self.recs[idx]
        names = rec["image_paths"] if isinstance(rec["image_paths"], list) else [rec["image_paths"]]

        # ---- vision ----
        pix_list: List[torch.Tensor] = []
        grid_list: List[torch.Tensor] = []
        seg_list: List[torch.Tensor] = []
        for n in names:
            p, g, s = self._load_one_image(n)
            pix_list.append(p); grid_list.append(g); seg_list.append(s)

        pixel_values   = torch.cat(pix_list, 0)         # (Σpatch, hid)
        image_grid_thw = torch.stack(grid_list, 0)      # (n_img, 3)
        img_tok_tensor = torch.cat(seg_list, 1)         # (1, Σfeat + wrappers)

        # ---- language (chat template) ----
        prompt_text = rec.get("input") or ""
        answer_text = rec.get("output")
        if answer_text is None:
            oj = rec.get("output_json")
            if oj is not None:
                answer_text = json.dumps(oj, ensure_ascii=False)
            else:
                raise KeyError(f"No 'output' or 'output_json' for sample idx {idx}")

        user_msgs = [{"role": "user", "content": prompt_text}]
        prompt_ids = self.tok.apply_chat_template(
            user_msgs,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )  # (1, P)

        # For training, we supervise the assistant message.
        assistant_msgs = [{"role": "assistant", "content": answer_text}]
        answer_ids = self.tok.apply_chat_template(
            assistant_msgs,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )  # (1, A)

        # ---- stitch and truncate if needed (keep images + answer) ----
        input_ids, attn_mask, labels = _concat_with_truncation_keep_images_and_answer(
            img_tok_tensor=img_tok_tensor,
            prompt_ids=prompt_ids,
            answer_ids=answer_ids,
            pad_id=self.tok.pad_token_id,
            max_len=self.max_len,
            debug=self.debug,
        )

        if self.debug:
            L = input_ids.numel()
            img_len = img_tok_tensor.size(1)
            P = prompt_ids.size(1)
            A = answer_ids.size(1)
            logging.debug(
                "[SEQ DEBUG] total=%d | img=%d | prompt=%d | answer=%d | max=%s | exceeded=%s",
                L, img_len, P, A, str(self.max_len), (self.max_len is not None and L > self.max_len)
            )

        return {
            "pixel_values":    pixel_values,   # (Σpatch, hid)
            "image_grid_thw":  image_grid_thw, # (n_img, 3)
            "input_ids":       input_ids,      # (L,)
            "attention_mask":  attn_mask,      # (L,)
            "labels":          labels,         # (L,)
        }


# -----------------------------
# Collator
# -----------------------------
def make_collate_fn(tokenizer):
    pad_id = tokenizer.pad_token_id

    def collate_fn(batch: List[Dict[str, torch.Tensor]]):
        # language
        ids  = torch.nn.utils.rnn.pad_sequence(
            [b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id)
        msk  = torch.nn.utils.rnn.pad_sequence(
            [b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
        lbls = torch.nn.utils.rnn.pad_sequence(
            [b["labels"] for b in batch], batch_first=True, padding_value=-100)

        # vision: pad to max seq_len across batch
        seq_max = max(b["pixel_values"].size(0) for b in batch)
        hid = batch[0]["pixel_values"].size(1)
        pix = torch.stack([
            torch.cat([
                b["pixel_values"],
                torch.zeros(seq_max - b["pixel_values"].size(0), hid, dtype=b["pixel_values"].dtype)
            ], 0) if b["pixel_values"].size(0) < seq_max else b["pixel_values"]
            for b in batch
        ], dim=0)  # (B, seq_max, hid)

        # For grid, keep original behavior: single tensor if B==1, else list (Qwen handles either).
        grids = [b["image_grid_thw"] for b in batch]
        grids = grids[0] if len(grids) == 1 else grids

        return {
            "input_ids": ids,
            "attention_mask": msk,
            "labels": lbls,
            "pixel_values": pix,
            "image_grid_thw": grids,
        }
    return collate_fn


# -----------------------------
# DebugTrainer
# -----------------------------
class DebugTrainer(Trainer):
    """Logs one-time shape/context info on first forward pass."""

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
        **kwargs,
    ):
        if debug and not getattr(self, "_inspected", False):
            self._inspected = True
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    logging.debug("%-15s : %-14s %s", k, tuple(v.shape), v.dtype)

            # Context length debug
            ids = inputs["input_ids"]  # (B, L)
            max_len = getattr(self.model.config, "max_position_embeddings", None)
            nonpad_per_sample = [int(m.sum().item()) for m in inputs["attention_mask"]]
            lbls = inputs["labels"]
            ans_tok_counts = [int((l != -100).sum().item()) for l in lbls]

            # Print an explicit EXCEEDED flag per sample
            exceeded = [
                (max_len is not None) and (int(ids.shape[1]) > int(max_len))
                for _ in range(ids.shape[0])
            ]

            logging.info(
                "[CTX DEBUG] shape=%s | max_pos=%s | nonpad=%s | answer_tokens=%s | exceeded_any=%s",
                tuple(ids.shape), str(max_len), nonpad_per_sample, ans_tok_counts, any(exceeded)
            )

        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
            **kwargs,
        )


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    args  = build_parser().parse_args()
    debug = args.debug
    (ic.enable if debug else ic.disable)()
    setup_logging(debug)

    os.makedirs(args.out_dir, exist_ok=True)
    logging.info("Checkpoints/adapters will be saved to: %s", args.out_dir)

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    warnings.filterwarnings("ignore", category=UserWarning, message=".*pinned memory.*")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Running on %s • Torch %s", device, torch.__version__)

    # Load model & processors
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        device_map="auto" if device == "cuda" else None,
        torch_dtype="auto",
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    image_processor = AutoImageProcessor.from_pretrained(args.model_dir, local_files_only=True)

    # Ensure right padding so truncation logic is straightforward
    tokenizer.padding_side = "right"

    # LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logging.info("Vision hidden size     : %d", model.config.vision_config.hidden_size)
    logging.info("Trainable parameters   : {:,} / {:,} ({:.2f}%)".format(
        trainable, total, 100.0 * trainable / total
    ))

    IMAGE_PATCH_ID = model.config.image_token_id
    MERGE = getattr(model.config.vision_config, "spatial_merge_size", 2)

    vision_start_id  = getattr(model.config, "vision_start_token_id", None)
    vision_end_id    = getattr(model.config, "vision_end_token_id", None)
    max_ctx = getattr(model.config, "max_position_embeddings", None)

    # Friendly print of special tokens
    def tok2(tid):
        try:
            return tokenizer.convert_ids_to_tokens([tid])[0]
        except Exception:
            return str(tid)

    print("image_patch_id :", IMAGE_PATCH_ID, tok2(IMAGE_PATCH_ID))
    if vision_start_id is not None:
        print("vision_start_id:", vision_start_id, tok2(vision_start_id))
    if vision_end_id is not None:
        print("vision_end_id  :", vision_end_id, tok2(vision_end_id))
    print("max_position_embeddings:", max_ctx)

    # Load & split
    ds_full: HFDataset = load_jsonl_dataset(args.data_json)
    split: DatasetDict = ds_full.train_test_split(test_size=args.val_ratio, seed=args.seed)
    logging.info("Dataset split — train: %d, eval: %d", len(split["train"]), len(split["test"]))

    # Datasets
    train_ds = QwenVLDataset(
        image_processor=image_processor,
        tokenizer=tokenizer,
        img_tok_id=IMAGE_PATCH_ID,
        merge=MERGE,
        start_id=vision_start_id, end_id=vision_end_id,
        max_len=max_ctx,
        debug=debug,
        records=split["train"],
    )
    val_ds = QwenVLDataset(
        image_processor=image_processor,
        tokenizer=tokenizer,
        img_tok_id=IMAGE_PATCH_ID,
        merge=MERGE,
        start_id=vision_start_id, end_id=vision_end_id,
        max_len=max_ctx,
        debug=debug,
        records=split["test"],
    )

    logging.info("Train samples: %d | Val samples: %d", len(train_ds), len(val_ds))

    # TrainingArguments
    targs = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=1,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        weight_decay=0.01,
        fp16=True,
        bf16=False,
        remove_unused_columns=False,  # very important for VL
        logging_steps=10,
        log_level="debug" if debug else "info",
    )

    trainer = DebugTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=make_collate_fn(tokenizer),
    )

    trainer.train()
