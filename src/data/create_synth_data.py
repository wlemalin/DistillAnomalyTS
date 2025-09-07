# synthesize.py (Standalone Version using original plotting function structure)

import argparse
import os
import pickle
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from plot_utils import plot_series_and_predictions, compute_rolling_stats_with_centroid, plot_and_save_rolling_plots
from scipy import stats
from torch.utils.data import Dataset
from tqdm import trange

# =============================================================================
# == Synthetic Data Generation Functions (Copied from original script) =======
# =============================================================================

# --- Helper Function ---


def add_anomalies_to_univariate_series(
    x: np.ndarray,
    normal_duration_rate: float,
    anomaly_duration_rate: float,
    anomaly_size_range: tuple[float, float],
    minimum_anomaly_duration: int,
    minimum_normal_duration: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Add anomalies to a given time series."""
    is_dummy_range = (anomaly_size_range[0]
                      == 0 and anomaly_size_range[1] == 1)
    if not is_dummy_range and anomaly_size_range[0] >= anomaly_size_range[1]:
        raise ValueError(
            f"The anomaly size range {anomaly_size_range} should be strictly increasing."
        )
    x_copy = x.copy()
    N = len(x_copy)
    distr_duration_normal = stats.expon(scale=normal_duration_rate)
    distr_duration_anomalous = stats.expon(scale=anomaly_duration_rate)
    max_number_of_intervals = 8
    location = 0
    anomaly_intervals = []
    for _ in range(max_number_of_intervals):
        random_states = np.random.randint(0, np.iinfo(np.int32).max, size=2)
        norm_dur = distr_duration_normal.rvs(random_state=random_states[0])
        norm_dur = max(norm_dur, minimum_normal_duration)
        anom_start = location + int(norm_dur)
        if anom_start >= N:
            break
        anom_dur = distr_duration_anomalous.rvs(random_state=random_states[1])
        anom_dur = max(anom_dur, minimum_anomaly_duration)
        anom_end = min(N, anom_start + int(anom_dur))
        if anom_start >= anom_end:
            location = anom_end
            continue
        if not is_dummy_range:
            shift_sign = 1 if np.random.randint(low=0, high=2) == 1 else -1
            shift_magnitude = np.random.uniform(
                anomaly_size_range[0], anomaly_size_range[1], size=anom_end - anom_start
            )
            shift = shift_sign * shift_magnitude
            x_copy[anom_start:anom_end] += shift
        location = anom_end
        anomaly_intervals.append((anom_start, anom_end))
        if location >= N:
            break
    return x_copy, anomaly_intervals


# --- Point Anomalies ---
def synthetic_dataset_with_point_anomalies(
    n_samples: int = 300, number_of_sensors: int = 1, frequency: float = 0.03,
    normal_duration_rate: float = 240.0, anomaly_duration_rate: float = 30.0,
    minimum_anomaly_duration: int = 5, minimum_normal_duration: int = 100,
    anomaly_std: float = 0.5, ratio_of_anomalous_sensors: float = 1.0,
    seed: Optional[int] = None
) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    """Generate point anomalies."""
    if seed is not None:
        np.random.seed(seed)
    t = np.arange(n_samples)
    x = np.array([np.sin(2 * np.pi * (frequency + 0.01 * i) * t)
                 for i in range(number_of_sensors)]).T
    if number_of_sensors > 0:
        num_anom_sensors = max(1 if ratio_of_anomalous_sensors > 0 else 0, int(
            round(number_of_sensors * ratio_of_anomalous_sensors)))
        num_anom_sensors = min(number_of_sensors, num_anom_sensors)
        sensors_with_anomalies = np.random.choice(
            number_of_sensors, num_anom_sensors, replace=False) if num_anom_sensors > 0 else []
    else:
        sensors_with_anomalies = []
    anomaly_intervals = [[] for _ in range(number_of_sensors)]
    for sensor in sensors_with_anomalies:
        _, intervals = add_anomalies_to_univariate_series(
            x[:,
                sensor], normal_duration_rate, anomaly_duration_rate, (0, anomaly_std * 2),
            minimum_anomaly_duration, minimum_normal_duration
        )
        original_sensor_data = np.sin(
            2 * np.pi * (frequency + 0.01 * sensor) * t)  # Store original
        modified_sensor_data = original_sensor_data.copy()
        sensor_anomalies = []
        for start, end in intervals:
            if start < end:  # Ensure interval is valid
                anomaly = np.random.normal(0, anomaly_std, end - start)
                modified_sensor_data[start:end] = anomaly  # Replace segment
                sensor_anomalies.append((start, end))
        x[:, sensor] = modified_sensor_data  # Apply modifications
        anomaly_intervals[sensor] = sensor_anomalies
    return x, anomaly_intervals


# --- Frequency Anomalies ---
def synthetic_dataset_with_frequency_anomalies(
    n_samples: int = 300, number_of_sensors: int = 1, frequency: float = 0.03,
    normal_duration_rate: float = 135.0, anomaly_duration_rate: float = 15.0,
    minimum_anomaly_duration: int = 7, minimum_normal_duration: int = 100,
    frequency_multiplier: float = 3.0, ratio_of_anomalous_sensors: float = 1.0,
    seed: Optional[int] = None
) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    """Generate frequency anomalies."""
    if seed is not None:
        np.random.seed(seed)
    t = np.arange(n_samples)
    x = np.zeros((n_samples, number_of_sensors))
    if number_of_sensors > 0:
        num_anom_sensors = max(1 if ratio_of_anomalous_sensors > 0 else 0, int(
            round(number_of_sensors * ratio_of_anomalous_sensors)))
        num_anom_sensors = min(number_of_sensors, num_anom_sensors)
        sensors_with_anomalies = np.random.choice(
            number_of_sensors, num_anom_sensors, replace=False) if num_anom_sensors > 0 else []
    else:
        sensors_with_anomalies = []
    anomaly_intervals = [[] for _ in range(number_of_sensors)]
    for sensor in range(number_of_sensors):
        base_freq = frequency + 0.01 * sensor
        freq_function = np.full(n_samples, base_freq)
        if sensor in sensors_with_anomalies:
            current_time = 0
            while current_time < n_samples:
                normal_duration = max(minimum_normal_duration, int(
                    stats.expon(scale=normal_duration_rate).rvs()))
                current_time += normal_duration
                if current_time >= n_samples:
                    break
                anomaly_duration = max(minimum_anomaly_duration, int(
                    stats.expon(scale=anomaly_duration_rate).rvs()))
                anomaly_end = min(n_samples, current_time + anomaly_duration)
                if current_time >= anomaly_end:
                    current_time = anomaly_end
                    continue
                multiplier = frequency_multiplier if np.random.random() < 0.5 else (
                    1.0 / frequency_multiplier if frequency_multiplier != 0 else 1.0)
                freq_function[current_time:anomaly_end] *= multiplier
                freq_function[current_time:anomaly_end] = np.maximum(
                    1e-9, freq_function[current_time:anomaly_end])
                anomaly_intervals[sensor].append((current_time, anomaly_end))
                current_time = anomaly_end
        phase = np.cumsum(2 * np.pi * freq_function)
        x[:, sensor] = np.sin(phase)
    return x, anomaly_intervals

# --- Trend Anomalies Helper ---


def generate_abnormal_slope(normal_slope: float, abnormal_slope_range: tuple[float, float], inverse_ratio: float) -> float:
    """Generate abnormal slope (simplified logic from previous version)."""
    min_slope, max_slope = abnormal_slope_range
    if np.isinf(max_slope):
        max_slope = max(abs(normal_slope) * 10, min_slope *
                        2) if normal_slope != 0 else min_slope * 2
    if max_slope <= min_slope:
        max_slope = min_slope + 1.0
    if np.random.random() > inverse_ratio:  # Same direction-ish
        lower = max(abs(normal_slope), min_slope)
        upper = max_slope
        if lower >= upper:
            return np.sign(normal_slope) * lower if normal_slope != 0 else lower
        magnitude = np.random.uniform(lower, upper)
        return np.sign(normal_slope) * magnitude if normal_slope != 0 else magnitude
    else:  # Inverse direction-ish
        lower = -max_slope
        upper = -min_slope
        if lower >= upper:
            # Needs better logic for inverse range
            return np.sign(normal_slope) * lower if normal_slope != 0 else lower
        return np.random.uniform(lower, upper)


# --- Trend Anomalies ---
def synthetic_dataset_with_trend_anomalies(
    n_samples: int = 300, number_of_sensors: int = 1, frequency: float = 0.02,
    normal_duration_rate: float = 510.0, anomaly_duration_rate: float = 100.0,
    minimum_anomaly_duration: int = 50, minimum_normal_duration: int = 150,
    ratio_of_anomalous_sensors: float = 1.0, normal_slope: float = 3.0,
    abnormal_slope_range: tuple[float, float] = (6.0, 20.0), inverse_ratio: float = 0.0,
    seed: Optional[int] = None
) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    """Generate trend anomalies."""
    if seed is not None:
        np.random.seed(seed)
    t = np.arange(n_samples)
    x = np.zeros((n_samples, number_of_sensors))
    if number_of_sensors > 0:
        num_anom_sensors = max(1 if ratio_of_anomalous_sensors > 0 else 0, int(
            round(number_of_sensors * ratio_of_anomalous_sensors)))
        num_anom_sensors = min(number_of_sensors, num_anom_sensors)
        sensors_with_anomalies = np.random.choice(
            number_of_sensors, num_anom_sensors, replace=False) if num_anom_sensors > 0 else []
    else:
        sensors_with_anomalies = []
    anomaly_intervals = [[] for _ in range(number_of_sensors)]
    for sensor in range(number_of_sensors):
        base_freq = frequency + 0.01 * sensor
        trend = np.zeros(n_samples)
        current_value = 0.0
        current_time = 0
        if sensor in sensors_with_anomalies:
            _, intervals = add_anomalies_to_univariate_series(
                np.zeros(
                    n_samples), normal_duration_rate, anomaly_duration_rate, (0, 1),
                minimum_anomaly_duration, minimum_normal_duration
            )
            last_interval_end = 0
            for start, end in intervals:
                start = max(last_interval_end, start)
                end = max(start, end)
                if start >= n_samples:
                    break
                end = min(n_samples, end)
                if start >= end:
                    continue
                if start > current_time:
                    time_segment = t[current_time:start] - t[current_time]
                    trend[current_time:start] = current_value + \
                        normal_slope * time_segment / n_samples
                    if start > current_time:
                        current_value = trend[start - 1]
                abnormal_slope = generate_abnormal_slope(
                    normal_slope, abnormal_slope_range, inverse_ratio)
                time_segment = t[start:end] - t[start]
                trend[start:end] = current_value + \
                    abnormal_slope * time_segment / n_samples
                if end > start:
                    current_value = trend[end - 1]
                current_time = end
                anomaly_intervals[sensor].append((start, end))
                last_interval_end = end
        if current_time < n_samples:
            time_segment = t[current_time:] - t[current_time]
            trend[current_time:] = current_value + \
                normal_slope * time_segment / n_samples
        x[:, sensor] = np.sin(2 * np.pi * base_freq * t) + trend
        min_val, max_val = np.min(x[:, sensor]), np.max(x[:, sensor])
        if max_val > min_val:
            x[:, sensor] = 2 * (x[:, sensor] - min_val) / \
                (max_val - min_val) - 1
        elif np.any(x[:, sensor]):
            x[:, sensor] = 0
    return x, anomaly_intervals

# --- Flat Trend Anomalies ---


def synthetic_dataset_with_flat_trend_anomalies(**args):
    """Generate flat trend anomalies."""
    flat_args = {'normal_slope': 3.0, 'abnormal_slope_range': (
        0.1, 1.5), 'inverse_ratio': 0.0}
    flat_args.update(args)
    return synthetic_dataset_with_trend_anomalies(**flat_args)

# --- Out-of-Range Anomalies ---


def synthetic_dataset_with_out_of_range_anomalies(
    number_of_sensors: int = 1, train_size: int = 5_000, test_size: int = 300,
    nominal_data_mean: float = 0.0, nominal_data_std: float = 0.1,
    normal_duration_rate: float = 240.0, anomaly_duration_rate: float = 20.0,
    anomaly_size_range: tuple = (0.5, 0.8), minimum_anomaly_duration: int = 5,
    minimum_normal_duration: int = 100, ratio_of_anomalous_sensors: float = 1.0,
    seed: Optional[int] = None
) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    """Generate out-of-range anomalies."""
    if seed is not None:
        np.random.seed(seed)
    test_data = np.random.normal(
        nominal_data_mean, nominal_data_std, size=(test_size, number_of_sensors))
    if number_of_sensors > 0:
        num_anom_sensors = max(1 if ratio_of_anomalous_sensors > 0 else 0, int(
            round(number_of_sensors * ratio_of_anomalous_sensors)))
        num_anom_sensors = min(number_of_sensors, num_anom_sensors)
        sensors_with_anomalies = np.random.choice(
            number_of_sensors, num_anom_sensors, replace=False) if num_anom_sensors > 0 else []
    else:
        sensors_with_anomalies = []
    all_anomaly_intervals = [[] for _ in range(number_of_sensors)]
    for idx in sensors_with_anomalies:
        modified_series, anomaly_locations = add_anomalies_to_univariate_series(
            test_data[:, idx], normal_duration_rate, anomaly_duration_rate, anomaly_size_range,
            minimum_anomaly_duration, minimum_normal_duration
        )
        test_data[:, idx] = modified_series
        all_anomaly_intervals[idx] = anomaly_locations
    return test_data, all_anomaly_intervals

# =============================================================================
# == SyntheticDataset Class (Uses integrated plotting function) ==============
# =============================================================================


class SyntheticDataset(Dataset):
    def __init__(
        self,
        data_dir="data_stft/synthetic/range/",
        synthetic_func_name="synthetic_dataset_with_out_of_range_anomalies",
    ):
        self.data_dir = data_dir
        self.figs_dir = os.path.join(data_dir, 'figs')
        self.series = []
        self.anom = []

        # create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.figs_dir, exist_ok=True)
        # os.makedirs(os.path.join(self.figs_dir, 'rolling'), exist_ok=True)

        try:
            self.synthetic_func = globals()[synthetic_func_name]
            print(f"Using generation function: {synthetic_func_name}")
        except KeyError:
            available = [k for k, v in globals().items()
                         if k.startswith('synthetic_dataset_with_') and callable(v)]
            print(f"ERROR: '{synthetic_func_name}' not found. Available: {available}")
            exit(1)

    def generate(self, num_series=400, seed=42, add_noise=False, window=30):
        if seed is not None:
            np.random.seed(seed)
            print(f"Using fixed random seed: {seed}")
        else:
            np.random.seed(None)
            print("Using non-fixed random seed.")

        print(f"Generating {num_series} series using {self.synthetic_func.__name__}...")
        for i in trange(num_series, desc="Generating Series & Plots"):
            gen_args = {'number_of_sensors': 1, 'ratio_of_anomalous_sensors': 1.0, 'seed': None}
            if self.synthetic_func.__name__ == 'synthetic_dataset_with_out_of_range_anomalies':
                gen_args['test_size'] = 300
            else:
                gen_args['n_samples'] = 300

            try:
                data, anomaly_locations = self.synthetic_func(**gen_args)
            except Exception as e:
                print(f" Error generating series {i}: {e}")
                continue

            if add_noise:
                data += np.random.normal(0, 0.08, data.shape)

            self.series.append(data)
            self.anom.append(anomaly_locations)

            # original plot
            try:
                fig = plot_series_and_predictions(
                    series=data,
                    single_series_figsize=(10, 1.5),
                    gt_anomaly_intervals=None,
                    anomalies=None,
                )
            except Exception as e:
                print(f" Error plotting series {i}: {e}")
                plt.close()
                continue

            fig_path = os.path.join(self.figs_dir, f'{i+1:d}.png')
            try:
                fig.savefig(fig_path)
            except Exception as e:
                print(f" Warning: could not save {fig_path}: {e}")
            plt.close(fig)

            # rolling-stats plots for sensor 0
            df_stats = compute_rolling_stats_with_centroid(data[:, 0], window=window, fs=1.0)

            std_p   = os.path.join(self.figs_dir, f'{i+1:d}_std.png')
            mean_p  = os.path.join(self.figs_dir, f'{i+1:d}_mean.png')
            stft_p  = os.path.join(self.figs_dir, f'{i+1:d}_stft.png')  # NEW

            plot_and_save_rolling_plots(
                df_stats,
                window,
                std_p,
                mean_p,
                series=data[:, 0],     # NEW: enables spectrogram save
                fs=1.0,
                stft_plot_path=stft_p  # NEW
            )

        self.save()
        print(f" Finished generating {num_series} series and plots.")

    def save(self):
        data_dict = {'series': self.series, 'anom': self.anom}
        save_path = os.path.join(self.data_dir, 'data.pkl')
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(data_dict, f)
            print(f"Data saved successfully to {save_path}")
        except Exception as e:
            print(f" Warning: Error saving data to {save_path}: {e}")

    def load(self):
        load_path = os.path.join(self.data_dir, 'data.pkl')
        print(f"Attempting to load data from: {load_path}")
        try:
            if not os.path.exists(load_path):
                raise FileNotFoundError(f"File not found: {load_path}")
            with open(load_path, 'rb') as f:
                data_dict = pickle.load(f)
            if 'series' not in data_dict:
                raise ValueError("Loaded data missing 'series' key.")
            self.series = data_dict['series']
            self.anom = data_dict.get('anom', [])
            if not isinstance(self.series, list) or not isinstance(self.anom, list):
                raise TypeError("Loaded data structure invalid.")
            print(f"Loaded dataset '{os.path.basename(os.path.normpath(self.data_dir))}' "
                  f"with {len(self.series)} series.")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.series, self.anom = [], []

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):
        if idx >= len(self.series):
            raise IndexError(f"Index {idx} out of range ({len(self.series)})")
        series_data = self.series[idx]
        series_tensor = torch.tensor(series_data, dtype=torch.float32)
        # anomaly intervals for sensor 0 only
        intervals = self.anom[idx][0] if idx < len(self.anom) and self.anom[idx] else []
        anom_tensor = torch.tensor(intervals, dtype=torch.float32) if intervals else torch.empty((0, 2), dtype=torch.float32)
        return anom_tensor, series_tensor

    def few_shots(self, num_shots=5, idx=None):
        if not self.series:
            return []
        n = len(self.series)
        if num_shots > n:
            num_shots = n
        indices = [idx] if isinstance(idx, int) else (
            np.random.choice(n, num_shots, replace=False) if idx is None else idx
        )
        result = []
        for i in indices:
            try:
                anom_tensor, series_tensor = self[i]
            except IndexError:
                continue
            anoms = []
            if anom_tensor.numel() > 0:
                anoms = [{'start': int(s), 'end': int(e)} for s, e in anom_tensor]
            result.append((series_tensor, anoms))
        return result

# =============================================================================
# == Main Execution Block (Keep EXACTLY as original) ==========================
# =============================================================================


def main(args):
    """Parses arguments and runs the dataset generation or loading."""
    dataset = SyntheticDataset(args.data_dir, args.synthetic_func)
    if args.generate:
        dataset.generate(args.num_series, args.seed, args.add_noise, args.window)
    else:
        dataset.load()
        if not dataset.series:
            print(f"No data loaded from {args.data_dir}. Use --generate.")
            return
        print(
            f"Dataset loaded successfully. Contains {len(dataset.series)} series.")
        num_with_anom_info = sum(
            1 for lst in dataset.anom if lst and lst[0]) if dataset.anom else 0
        print(
            f"Anomaly interval information loaded for {num_with_anom_info} series (first sensor).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate/load synthetic dataset & plots (plots generated WITHOUT GT highlighting).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--num_series", type=int, default=400,
                        help="Number of series to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--data_dir", type=str, default='data_stft/synthetic/range/',
                        help="Directory to save/load data and plots")
    parser.add_argument("--generate", action="store_true",
                        help="Generate new data instead of loading")
    parser.add_argument("--add_noise", action="store_true",
                        help="Add Gaussian noise (std=0.08) to generated data")

    available_funcs = [k for k, v in globals().items() if k.startswith(
        'synthetic_dataset_with_') and callable(v)]
    default_func = "synthetic_dataset_with_out_of_range_anomalies"
    if not available_funcs:
        available_funcs = [default_func]  # Fallback
    elif default_func not in available_funcs:
        default_func = sorted(available_funcs)[0]

    parser.add_argument("--synthetic_func", type=str, default=default_func,
                        choices=sorted(list(set(available_funcs))),
                        help="Name of the synthetic function to use")
    parser.add_argument("--window", type=int, default=30,
                        help="Window size for rolling mean/std plots")

    args = parser.parse_args()

    if args.synthetic_func not in globals() or not callable(globals()[args.synthetic_func]):
        print(
            f"\nFATAL ERROR: Selected function '{args.synthetic_func}' not found/callable.")
        exit(1)

    try:
        main(args)
        print("\nScript finished successfully.")
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        print("\nScript finished with errors.")

