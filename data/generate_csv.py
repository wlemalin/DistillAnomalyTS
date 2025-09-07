#!/usr/bin/env python
# ─────────────────────────── generate_csv.py ────────────────────────────
#
#  Walk any folder tree, find every  data.pkl  produced by SyntheticDataset.save,
#  and write **one CSV per time-series** into a sibling  csv_data/  directory.
#
#  Usage examples
#    python generate_csv.py data/synthetic
#    python generate_csv.py data/synthetic/flat-trend/train --window 50
#
# ───────────────────────────────────────────────────────────────────────────────
from scipy.signal import spectrogram
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# ═════════════════════════════════ helper functions ═══════════════════════════
def load_pkl(pkl_path: Path) -> dict:
    """Return the dictionary saved by SyntheticDataset.save()."""
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    if "series" not in data:
        raise KeyError(f"'series' key not found in {pkl_path}")
    return data


def scale_round_and_pad(arr: np.ndarray) -> np.ndarray:
    """
    Multiply by 100, round to nearest integer, pad magnitude to three digits.
    NaNs are kept as empty strings so they appear blank in the CSV.
    """
    # keep NaN so we can recognise it later
    scaled = np.where(np.isnan(arr), np.nan, np.rint(arr * 100))

    def _fmt(v) -> str:
        if np.isnan(v):
            return ""
        v_int = int(v)
        if v_int < 0:
            return f"-{abs(v_int):02d}"   # "-007", "-100", …
        else:
            return f"{v_int:02d}"  # "000",  "096", "100"
    vfmt = np.vectorize(_fmt, otypes=[object])
    return vfmt(scaled)


def series_to_dataframe(series: np.ndarray, window: int) -> pd.DataFrame:
    """
    Build a DataFrame containing, for each sensor:
      - sensor_X
      - sensor_X_mean
      - sensor_X_std
      - sensor_X_centroid

    All columns are then multiplied by 100, rounded, and zero‐padded to 3 digits.
    """
    # 1) Rescale raw into [0,1]
    series = (series / 2) + 0.5

    # 2) Prepare DataFrame of raw values
    if series.ndim == 1:
        df = pd.DataFrame(series, columns=["sensor_1"])
    else:
        cols = [f"sensor_{i+1}" for i in range(series.shape[1])]
        df = pd.DataFrame(series, columns=cols)

    # 3) Rolling mean & std
    for col in df.columns:
        df[f"{col}_mean"] = df[col].rolling(window, min_periods=window).mean()
        df[f"{col}_std"] = df[col].rolling(window, min_periods=window).std()

    # 4) Spectral centroid per sensor
    #    (STFT with nperseg=2*window, noverlap=window)
    for col in df.columns[: series.shape[1]]:  # only raw cols
        x = df[col].to_numpy(dtype=float)
        freqs, times, Sxx = spectrogram(
            x, fs=1.0, window="hann", nperseg=window*2, noverlap=window
        )
        P = np.abs(Sxx)
        cent = np.sum(freqs[:, None] * P, axis=0) / (np.sum(P, axis=0) + 1e-10)
        # interpolate frames back to full length
        frame_pos = np.linspace(0, len(x) - 1, num=len(cent))
        cent_full = np.interp(np.arange(len(x)), frame_pos, cent)
        df[f"{col}_centroid"] = cent_full * 10

    # 5) Format all columns (scale×100, round, pad to 3 digits)
    formatted = scale_round_and_pad(df.to_numpy())
    df_fmt = pd.DataFrame(formatted, columns=df.columns, dtype="string")
    df_fmt.index.name = "time_step"

    return df_fmt


# ═════════════════════════════════ core workers ═══════════════════════════════
def pkl_to_csv(pkl_path: Path, window: int):
    """Write every series in a data.pkl to individual CSVs."""
    root_dir = pkl_path.parent            # …/train   or   …/eval
    csv_dir = root_dir / "csv_data"
    csv_dir.mkdir(exist_ok=True)

    data = load_pkl(pkl_path)
    series_list = data["series"]

    bar_desc = f"→ {pkl_path.relative_to(Path.cwd())}"
    for idx, series in enumerate(tqdm(series_list, desc=bar_desc)):
        df = series_to_dataframe(np.asarray(series), window)
        csv_path = csv_dir / f"series_{idx+1:05d}.csv"
        df.to_csv(csv_path, index=True)


def walk_and_convert(start_dir: Path, window: int):
    """Recursively find every data.pkl under start_dir and convert it."""
    for pkl_path in start_dir.rglob("data.pkl"):
        pkl_to_csv(pkl_path, window)


# ═════════════════════════════════ entry-point ════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Convert SyntheticDataset data.pkl files to per-series CSVs "
                     "including rolling mean & std-dev, all scaled/padded."),
    )
    parser.add_argument(
        "path",
        type=str,
        help="Root directory (e.g. v2/data/synthetic or any sub-folder).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Rolling window length for mean/std calculations (default: 30).",
    )
    args = parser.parse_args()

    start_dir = Path(args.path).expanduser().resolve()
    if start_dir.is_file() and start_dir.name == "data.pkl":
        pkl_to_csv(start_dir, args.window)
    else:
        walk_and_convert(start_dir, args.window)
