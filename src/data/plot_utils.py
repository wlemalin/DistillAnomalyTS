from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import spectrogram

# ---------- small helper for consistent ticks + grid ----------
def _apply_time_ticks_and_grid(ax, xmax: int, step: int = 25) -> None:
    ax.set_xticks(np.arange(0, max(0, int(xmax)) + 1, step))
    ax.grid(True, which='both', color='lightgray', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
# --------------------------------------------------------------


def compute_rolling_stats_with_centroid(
    series: np.ndarray,
    window: int = 30,
    fs: float = 1.0,
) -> pd.DataFrame:
    # (unchanged logic; no plotting here)
    df = pd.DataFrame({'value': series})

    mean_col = f'rolling_mean_{window}'
    std_col  = f'rolling_std_{window}'
    df[mean_col] = df['value'].rolling(window, min_periods=1).mean()
    df[std_col]  = df['value'].rolling(window, min_periods=1).std(ddof=0)

    nperseg  = window * 2
    noverlap = window
    freqs, times, Sxx = spectrogram(
        series, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap
    )
    P = np.abs(Sxx)
    cent = np.sum(freqs[:, None] * P, axis=0) / (np.sum(P, axis=0) + 1e-10)

    frame_positions = np.linspace(0, len(series)-1, num=len(cent))
    cent_full = np.interp(np.arange(len(series)), frame_positions, cent)
    lo, hi = cent_full.min(), cent_full.max()
    cent_scaled = (cent_full - lo) / (hi - lo) * 99 if hi > lo else np.zeros_like(cent_full)
    centroid_col = f'stft_centroid_{window}'
    df[centroid_col] = np.round(cent_scaled).astype(int).clip(0, 99)

    return df


def create_color_generator(exclude_color='blue'):
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][1:]
    filtered_colors = [color for color in default_colors if color != exclude_color]
    return (color for color in filtered_colors)


def plot_series_and_predictions(
    series: np.ndarray,
    gt_anomaly_intervals: list[list[tuple[int, int]]] | None,
    anomalies: Optional[dict] = None,
    single_series_figsize: tuple[int, int] = (20, 3),
):
    plt.figure(figsize=single_series_figsize)

    color_generator = create_color_generator()

    def get_next_color(gen):
        try:
            return next(gen)
        except StopIteration:
            return next(create_color_generator())

    num_anomaly_methods = len(anomalies) if anomalies else 0
    ymin_max = [
        (i / num_anomaly_methods * 0.5 + 0.25, (i + 1) / num_anomaly_methods * 0.5 + 0.25)
        for i in range(num_anomaly_methods)
    ][::-1] if num_anomaly_methods else []

    for i in range(series.shape[1]):
        plt.ylim((-1.1, 1.1))
        plt.plot(series[:, i], color='steelblue')

        if gt_anomaly_intervals is not None:
            for start, end in gt_anomaly_intervals[i]:
                plt.axvspan(start, end, alpha=0.2, color=gt_color)

        if anomalies is not None:
            for idx, (method, anomaly_values) in enumerate(anomalies.items()):
                if anomaly_values.shape == series.shape:
                    anomaly_values = np.nonzero(anomaly_values[:, i])[0].flatten()
                ymin, ymax = ymin_max[idx]
                c = get_next_color(color_generator)
                for anomaly in anomaly_values:
                    plt.axvspan(anomaly, anomaly + 1, ymin=ymin, ymax=ymax,
                                alpha=0.5, color=c, lw=0)
                plt.plot([], [], color=c, label=method)

    _apply_time_ticks_and_grid(plt.gca(), xmax=len(series) - 1, step=25)

    plt.tight_layout()
    if anomalies is not None:
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    return plt.gcf()


def plot_and_save_rolling_plots(
    df: pd.DataFrame,
    window: int,
    std_plot_path: str,
    mean_plot_path: str,
    *,
    series: Optional[np.ndarray] = None,
    fs: float = 1.0,
    stft_plot_path: Optional[str] = None,
) -> None:
    """
    Generate and save:
      - Rolling std (index > window) → std_plot_path
      - Rolling mean (index > window) → mean_plot_path
      - (optional) STFT spectrogram of `series` → stft_plot_path
    """
    Path(std_plot_path).parent.mkdir(parents=True, exist_ok=True)
    Path(mean_plot_path).parent.mkdir(parents=True, exist_ok=True)
    if stft_plot_path:
        Path(stft_plot_path).parent.mkdir(parents=True, exist_ok=True)

    mean_col = f'rolling_mean_{window}'
    std_col  = f'rolling_std_{window}'
    color_mean, color_std = 'tab:green', 'tab:orange'

    filtered = df[df.index > window]

    # 1) Rolling std plot
    fig_std, ax_std = plt.subplots(figsize=(10, 5))
    ax_std.plot(filtered.index, filtered[std_col], color=color_std)
    ax_std.set_xlabel('Index')
    ax_std.set_ylabel(std_col)
    ax_std.set_title(f'{window}-pt Rolling σ (index > {window})')
    xmax = int(filtered.index.max()) if not filtered.empty else 0
    _apply_time_ticks_and_grid(ax_std, xmax=xmax, step=25)
    fig_std.tight_layout()
    fig_std.savefig(std_plot_path, dpi=60)
    plt.close(fig_std)

    # 2) Rolling mean plot
    fig_mean, ax_mean = plt.subplots(figsize=(10, 5))
    ax_mean.plot(filtered.index, filtered[mean_col], color=color_mean)
    ax_mean.set_xlabel('Index')
    ax_mean.set_ylabel(mean_col)
    ax_mean.set_title(f'{window}-pt Rolling μ (index > {window})')
    _apply_time_ticks_and_grid(ax_mean, xmax=xmax, step=25)
    fig_mean.tight_layout()
    fig_mean.savefig(mean_plot_path, dpi=60)
    plt.close(fig_mean)

    # 3) (optional) Spectrogram plot saved alongside the others
    if stft_plot_path is not None and series is not None and len(series) > 0:
        nperseg  = window * 2
        noverlap = window
        freqs, times, Sxx = spectrogram(
            series, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap
        )
        P_db = 10 * np.log10(np.abs(Sxx) + 1e-12)

        fig_sp, ax_sp = plt.subplots(figsize=(10, 4))
        pcm = ax_sp.pcolormesh(times, freqs, P_db, shading='gouraud', cmap='magma')
        cbar = fig_sp.colorbar(pcm, ax=ax_sp)
        cbar.set_label('Power (dB)')

        ax_sp.set_title(f'STFT Spectrogram (window={window}, nperseg={nperseg}, overlap={noverlap})')
        ax_sp.set_xlabel('Time (samples)')
        ax_sp.set_ylabel('Frequency (Hz)')

        # Ticks/grid to match your convention
        _apply_time_ticks_and_grid(ax_sp, xmax=len(series) - 1, step=25)

        fig_sp.tight_layout()
        fig_sp.savefig(stft_plot_path, dpi=100)
        plt.close(fig_sp)
