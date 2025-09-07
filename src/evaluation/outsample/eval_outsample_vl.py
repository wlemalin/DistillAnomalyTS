import os, re, json, argparse, math
from pathlib import Path
from typing import List, Dict

import torch
from PIL import Image
import pandas as pd
from tqdm import tqdm
from safetensors.torch import load_file as safe_load
from transformers import AutoTokenizer, AutoImageProcessor

# ========= OFFLINE =========
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ========= CLI =========
parser = argparse.ArgumentParser(description="Run VL anomaly detector with configurable image modes.")
parser.add_argument("--mode", "-m", choices=["stft", "ts", "ts3", "all"], default="ts",
                    help="Which images to feed: stft | ts | ts3 | all")
parser.add_argument("--lora-adapter-path", type=Path, default=Path("/home2020/home/beta/wlaemlin/repo/src/train_vision/qwen2.5-vl-3b-ft-lora-4o_filtered_gt/checkpoint-686"),
                    help="Path to a LoRA adapter directory containing adapter_config.json and adapter_model.safetensors")
parser.add_argument("--output-jsonl", type=str, default=None,
                    help="Output JSONL file path. Defaults to outsample_{mode}_outputs.jsonl")
parser.add_argument("--ban-tool-words", action="store_true",
                    help="Discourage tool-call style by banning words like 'action', 'tool', 'function', 'call'.")
parser.add_argument("--disable-lora", action="store_true",
                    help="Skip merging LoRA (use base model only) for A/B sanity checks.")
args = parser.parse_args()

IMG_MODE: str = args.mode
lora_adapter_path: Path = args.lora_adapter_path.resolve()
output_jsonl: str = args.output_jsonl or f"outsample_{IMG_MODE}_outputs.jsonl"

# ========= PATHS =========
csv_dir = Path("/home2020/home/beta/wlaemlin/repo/src/evaluation/outsample/dataset/csv_data/").resolve()
img_dir = Path("/home2020/home/beta/wlaemlin/repo/src/evaluation/outsample/dataset/figs/").resolve()
base_model_path = Path("/home2020/home/beta/wlaemlin/hf_models/Qwen2.5-VL-3B-Instruct_clean").resolve()

assert csv_dir.is_dir(), f"CSV directory not found: {csv_dir}"
assert img_dir.is_dir(), f"Image directory not found: {img_dir}"
assert base_model_path.is_dir(), f"Model directory not found: {base_model_path}"
if not args.disable_lora:
    assert lora_adapter_path.is_dir(), f"LoRA directory not found: {lora_adapter_path}"

# ========= IMPORT QWEN VL CLASS =========
try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration as QwenVLForCG
except Exception:
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration as QwenVLForCG

# ========= LOAD TOKENIZER / IMAGE PROCESSOR / MODEL =========
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    local_files_only=True,
)
tokenizer.padding_side = "right"

image_processor = AutoImageProcessor.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    local_files_only=True,
)

DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
model = QwenVLForCG.from_pretrained(
    str(base_model_path),
    torch_dtype=DTYPE,
    device_map="auto",
    local_files_only=True,
    low_cpu_mem_usage=True,
)
model.eval()
device = next(model.parameters()).device

# ========= LORA MERGE (OPTIONAL) =========
def _named_linear_modules(m):
    return {n: mod for n, mod in m.named_modules() if hasattr(mod, "weight") and getattr(mod, "weight") is not None}

def _best_match_module(target_suffix, name2mod):
    cands = [n for n in name2mod if n.endswith(target_suffix)]
    if not cands: return None
    cands.sort(key=len, reverse=True)
    return cands[0]

def merge_lora_inplace(model, adapter_dir: Path):
    cfg_path = adapter_dir / "adapter_config.json"
    st_path  = adapter_dir / "adapter_model.safetensors"
    if not (cfg_path.exists() and st_path.exists()):
        raise FileNotFoundError(f"LoRA files missing in {adapter_dir}")
    peft_cfg = json.loads(cfg_path.read_text())
    lora_alpha = float(peft_cfg.get("lora_alpha", 1))
    r_global   = peft_cfg.get("r", None)

    sd = safe_load(str(st_path))
    name2mod = _named_linear_modules(model)
    dev = next(model.parameters()).device

    merged, skipped = 0, 0
    for a_key in [k for k in sd.keys() if k.endswith(".lora_A.weight")]:
        base = a_key[:-len(".lora_A.weight")]
        b_key = base + ".lora_B.weight"
        if b_key not in sd:
            skipped += 1; continue
        A = sd[a_key].to(device=dev, dtype=torch.float32)
        B = sd[b_key].to(device=dev, dtype=torch.float32)
        r = A.shape[0]
        scale = lora_alpha / float(r_global if r_global else r)

        cleaned = base
        for pref in ("base_model.model.", "base_model.", "model.", ""):
            if cleaned.startswith(pref):
                cleaned = cleaned[len(pref):]; break
        mod_name = _best_match_module(cleaned, name2mod)
        if mod_name is None:
            skipped += 1; continue
        linear = name2mod[mod_name]

        delta = torch.matmul(B, A) * scale
        if tuple(delta.shape) != tuple(linear.weight.shape):
            skipped += 1; continue
        with torch.no_grad():
            linear.weight += delta.to(dtype=linear.weight.dtype)
        merged += 1
    print(f"✅ LoRA merge complete: merged {merged} layers, skipped {skipped}.")

if not args.disable_lora:
    try:
        merge_lora_inplace(model, lora_adapter_path)
    except Exception as e:
        print(f"⚠️ LoRA merge failed ({e}). Proceeding with base model only.")

# ========= PROMPT =========
IMAGE_MODE_DESCRIPTIONS: Dict[str, str] = {
    "text": "",
    "stft": " and the STFT spectrogram image",
    "ts": " and the time-series plot image",
    "ts3": " and the following plot images: raw values, moving average, and moving standard deviation",
    "all": " and the following plot images: raw values, moving average, moving standard deviation, and the STFT spectrogram",
}

SYSTEM_PROMPT = (
    "You are a strict JSON generator. Respond with a single JSON object only. "
    "Do NOT call tools or functions. Do NOT use keys other than exactly those specified. "
    "If no anomaly exists, reply with {{\"anomalies\": []}} only."
)

# NOTE: All literal braces are escaped ({{ }}) so .format() doesn't treat them as placeholders.
PROMPT_TEMPLATE = (
    "Given the time series below{images_note}, determine whether there is an anomalous interval.\n"
    "If the series is entirely normal, return the empty JSON template. Otherwise, detect and describe the nature of the anomaly.\n\n"
    "Return ONLY a JSON object formatted exactly as follows, with no extra keys or text:\n\n"
    "Empty (no anomaly):\n"
    "{{\n  \"anomalies\": []\n}}\n\n"
    "Non-empty:\n"
    "{{\n  \"anomalies\": [\n    {{\"start\": <int>, \"end\": <int>, \"description\": <string>}}\n  ]\n}}\n\n"
    "Rules:\n"
    "* If there is no anomaly, use the empty array (\"anomalies\": []).\n"
    "* At most one anomaly per series.\n"
    "* Each description must be one short sentence explaining how the data deviates from normal.\n"
    "* Do NOT invent anomalies when none exist.\n\n"
    "Time series (len: {length}):\n"
    "# Columns per line: timestamp : value, moving_average, moving_std\n"
    "{series}"
)

def format_series(df: pd.DataFrame) -> str:
    return "\n".join(
        f"{int(r.time_step)} : {int(r.sensor_1)}, {int(r.sensor_1_mean)}, {int(r.sensor_1_std)}"
        for r in df.itertuples(index=False)
    )

# Prefer JSON objects that actually contain "anomalies"
def extract_best_json(text: str) -> str:
    objs = []
    stack = []
    for i, ch in enumerate(text):
        if ch == '{':
            stack.append(i)
        elif ch == '}' and stack:
            start = stack.pop()
            try:
                s = text[start:i+1]
                o = json.loads(s)
                objs.append((start, i, o))
            except Exception:
                pass
    for _, _, o in reversed(objs):
        if isinstance(o, dict) and "anomalies" in o:
            return json.dumps(o)
    if objs:
        return json.dumps(objs[-1][2])
    return ""

def build_image_paths(base_stem: str, mode: str) -> List[Path]:
    paths: List[Path] = []
    if mode in ("ts", "ts3", "all"):
        paths.append(img_dir / f"{base_stem}_raw.png")
    if mode in ("ts3", "all"):
        paths.append(img_dir / f"{base_stem}_mean.png")
        paths.append(img_dir / f"{base_stem}_std.png")
    if mode in ("stft", "all"):
        paths.append(img_dir / f"{base_stem}_stft.png")
    return paths

def get_series_id(filename: str) -> str:
    stem = Path(filename).stem
    return stem.split("_", 1)[0]

# ========= SAME GRID LOGIC AS TRAINING =========
def _even_grid_from_npatch(n: int) -> tuple[int, int]:
    root = int(math.isqrt(n))
    for w in range(root, 1, -1):
        if n % w == 0:
            h = n // w
            if (h % 2 == 0) and (w % 2 == 0):
                return h, w
    for w in (2, 4, 6, 8):
        if n % w == 0 and (n // w) % 2 == 0:
            return n // w, w
    raise ValueError(f"Cannot find even factors for {n} patches")

# ========= CONSTANTS FROM CONFIG =========
IMG_PATCH_ID = model.config.image_token_id
MERGE = getattr(model.config.vision_config, "spatial_merge_size", 2)
START_ID = getattr(model.config, "vision_start_token_id", None)
END_ID   = getattr(model.config, "vision_end_token_id", None)

# ========= MAIN =========
results = []

# Optional bad-words to discourage tool call style
bad_words_ids = None
if args.ban_tool_words:
    bad_words = ["action", "tool", "function", "call"]
    toks = tokenizer(bad_words, add_special_tokens=False).input_ids  # list[list[int]]
    bad_words_ids = toks

for fname in tqdm(sorted(os.listdir(csv_dir))):
    if not fname.endswith(".csv"):
        continue

    m = re.search(r'_(\d+)_(\d+)\.csv$', fname)
    if not m:
        print(f"Skipping unexpected CSV name: {fname}")
        continue
    ground_truth = [[int(m.group(1)), int(m.group(2))]]

    df = pd.read_csv(csv_dir / fname)
    base_stem = Path(fname).with_suffix("").name
    image_paths = build_image_paths(base_stem, IMG_MODE)

    series_id = get_series_id(fname)

    missing = [p for p in image_paths if not p.exists()]
    if missing:
        print(f"⚠️ Missing images for {fname}: {', '.join(str(p) for p in missing)}")
        continue

    # ---- Prepare images (replicate training: factor even grid, pad if needed) ----
    images = [Image.open(p).convert("RGB") for p in image_paths]
    pix_list: List[torch.Tensor] = []
    thw_list: List[torch.Tensor] = []
    seg_tokens: List[int] = []

    hidden_dim = None

    for im in images:
        out = image_processor(images=im, return_tensors="pt")
        pv = out["pixel_values"]
        # Accept both (1, n_patches, hidden) and (n_patches, hidden)
        if pv.ndim == 3 and pv.shape[0] == 1:
            pv = pv.squeeze(0)                  # -> (n_patches, hidden)
        elif pv.ndim == 2:
            pass                                # already (n_patches, hidden)
        else:
            raise RuntimeError(f"Unexpected pixel_values shape: {tuple(pv.shape)}")

        n_patch, hid = pv.shape
        if hidden_dim is None:
            hidden_dim = hid
        else:
            assert hid == hidden_dim, f"Hidden dim mismatch: {hidden_dim} vs {hid}"

        h_grid, w_grid = _even_grid_from_npatch(n_patch)
        target = h_grid * w_grid
        if target > n_patch:
            pad = torch.zeros(target - n_patch, hid, dtype=pv.dtype)
            pv = torch.cat([pv, pad], 0)

        # #features after spatial merge
        n_feat = (h_grid // MERGE) * (w_grid // MERGE)

        # Build image token segment like training
        if START_ID is not None and END_ID is not None:
            seg_tokens.append(START_ID)
            seg_tokens.extend([IMG_PATCH_ID] * int(n_feat))
            seg_tokens.append(END_ID)
        else:
            seg_tokens.extend([IMG_PATCH_ID] * int(n_feat))

        pix_list.append(pv)
        thw_list.append(torch.tensor([1, h_grid, w_grid], dtype=torch.int64))

    pixel_values = torch.cat(pix_list, dim=0).to(device=device, dtype=torch.float32)  # (Σpatch, hid)
    image_grid_thw = torch.stack(thw_list, dim=0).to(device)  # (n_img, 3)

    # ---- Build prompt with chat template (TEXT ONLY; images are prepended as tokens) ----
    series_txt = format_series(df)
    images_note = IMAGE_MODE_DESCRIPTIONS.get(IMG_MODE, "")
    prompt_txt = PROMPT_TEMPLATE.format(length=len(df), series=series_txt, images_note=images_note)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_txt},
    ]
    chat_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(device)  # (1, L)

    pre_ids_tensor = torch.tensor([seg_tokens], dtype=torch.long, device=device)
    input_ids = torch.cat([pre_ids_tensor, chat_ids], dim=1)
    attn_mask = torch.ones_like(input_ids)

    # ---- GENERATE ----
    gen_kwargs = dict(
        max_new_tokens=256,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    if bad_words_ids:
        gen_kwargs["bad_words_ids"] = bad_words_ids

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **gen_kwargs,
        )

    # Decode only new tokens
    prompt_len = input_ids.shape[1]
    decoded = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
    decoded_clean = decoded.strip().strip("`").strip()
    llm_json = extract_best_json(decoded_clean)

    out_json = {}
    if llm_json:
        try:
            out_json = json.loads(llm_json)
        except json.JSONDecodeError:
            print(f"⚠️ JSON parse failed for {fname}")

    results.append({
        "id": series_id,
        "ground_truth": ground_truth,
        "input": prompt_txt,
        "image_paths": [str(p) for p in image_paths],
        "output": llm_json or decoded_clean,
        "output_json": out_json,
    })
    print("processed one")

with open(output_jsonl, "w") as f:
    for row in results:
        json.dump(row, f); f.write("\n")

print(f"✅ Done. Saved: {output_jsonl}")
