import os
import time
import json
import random
import torch
import numpy as np
import matplotlib
from sprite_core.config import Config
from netCDF4 import Dataset as NetCDFFile
import math

matplotlib.use("Agg")  # For non-interactive environment saving
from adjustText import adjust_text
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from functools import lru_cache
from torch.utils.data import DataLoader, Dataset, Subset

# =========================
# Some global constants
# =========================
TIME_INTERVAL_MIN = 5
DELTA_T_HR = TIME_INTERVAL_MIN / 60.0
BASE_PATH = ""
START_OFFSET = 0

torch.set_num_threads(32)

if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)


# =========================
# Collate FN
# =========================
def skip_none_collate_fn(batch):
    """
    Filter out None items to prevent an entire batch from failing if one item fails to load.
    batch: List[ (frames_tensor, (year_str, month_str)) ], length = batch_size
    """
    # 1) Filter out None
    filtered_batch = [x for x in batch if x is not None]
    if len(filtered_batch) == 0:
        return None

    # 2) Collect frames_list and meta_info_list
    frames_list = []
    year_month_list = []
    for frames_tensor, (year_str, month_str) in filtered_batch:
        frames_list.append(frames_tensor)
        year_month_list.append((year_str, month_str))

    # 3) Stack frames_list into a tensor of shape [batch_size, ...]
    #    Must ensure consistent shape
    frames_batch = torch.stack(frames_list, dim=0)

    # 4) Return (frames_batch, year_month_list),
    #    where year_month_list is a list of length = batch_size, each item is (year_str, month_str)
    return frames_batch, year_month_list


def stratified_random_sample(total_size, sample_size, num_segments):
    """
    Stratified random sampling to avoid out-of-range issues.
    """
    # Use math.ceil to avoid fraction
    segment_size = math.ceil(total_size / num_segments)

    # Final index storage
    all_indices = []
    # Track how many have been sampled
    used_samples = 0

    for i in range(num_segments):
        seg_start = i * segment_size
        seg_end = min((i + 1) * segment_size, total_size)  # Do not exceed total_size

        # How many segments are left
        remain_segments = num_segments - i
        # How many samples left to pick
        remain_samples = sample_size - used_samples
        # Evenly allocate for this segment
        this_segment_needed = remain_samples // remain_segments
        if this_segment_needed <= 0:
            break  # If already allocated enough

        # Randomly pick this_segment_needed from [seg_start, seg_end)
        indices_seg = np.random.choice(np.arange(seg_start, seg_end), size=this_segment_needed, replace=False)
        all_indices.append(indices_seg)
        used_samples += this_segment_needed

    # Concatenate
    sampled_indices = np.hstack(all_indices)
    return sampled_indices


def parse_year_month_from_filename(file_path):
    """
    Suppose a filename like: 20220821_002018.nc
    0~3 digits: Year  -> 2022
    4~5 digits: Month -> 08
    The rest might be day/time, but we only care about year, month
    """
    basename = os.path.basename(file_path)
    year_str = basename[0:4]
    month_str = basename[4:6]
    return year_str, month_str


# =========================
# Example Dataset
# =========================
class NetCDFDataset(Dataset):
    def __init__(self, base_dir, splits, sequence_length=1, read_mode="sequential", start_offset=0):
        super().__init__()
        self.splits = splits
        self.sequence_length = sequence_length
        self.read_mode = read_mode
        self.local_folder_paths = [os.path.join(base_dir, split) for split in splits]
        if self.read_mode == "sequential":
            self.start_offset = start_offset
        else:
            self.start_offset = 0

        self.all_sequences = self._get_all_sequences()

    def _get_all_sequences(self):
        """
        Traverse directories and gather all .nc file paths.
        """
        sequences = []
        for folder_path in self.local_folder_paths:
            if os.path.exists(folder_path):
                for root, _, files in os.walk(folder_path):
                    file_paths = [os.path.join(root, f) for f in files if f.endswith(".nc")]
                    sequences.extend(file_paths)

        sequences = sorted(sequences)

        unique_paths = []
        seen_basenames = set()
        for path in sequences:
            base = os.path.basename(path)
            if base in seen_basenames:
                # Already encountered a file with the same base name, treat as duplicate -> skip
                continue
            seen_basenames.add(base)
            unique_paths.append(path)

        if self.start_offset > 0:
            unique_paths = unique_paths[self.start_offset :]

        if self.start_offset < 0:
            unique_paths = unique_paths[: self.start_offset]

        return unique_paths

    def __len__(self):
        if len(self.all_sequences) == 0:
            return 0
        return len(self.all_sequences) - self.sequence_length + 1

    @staticmethod
    @lru_cache(maxsize=128)
    def _load_frame(file_path, retries=1):
        for _ in range(retries):
            try:
                with NetCDFFile(file_path, "r") as nc_data:
                    rrdata = nc_data.variables.get("RRdata", None)
                    if rrdata is None:
                        raise KeyError(f"Variable 'RRdata' not found in file {file_path}")
                    return np.ma.filled(rrdata[...], 0)
            except (OSError, KeyError) as e:
                print(f"Error encountered while loading file {file_path}: {e}. Retrying...")
        return None

    def __getitem__(self, idx):
        if idx + self.sequence_length > len(self.all_sequences):
            raise IndexError(f"Index {idx} out of range (total={len(self.all_sequences)}).")

        if self.read_mode == "random":
            return self._getitem_random(idx)
        elif self.read_mode == "sequential":
            return self._getitem_sequential(idx)
        else:
            raise ValueError(f"Unknown read_mode: {self.read_mode}")

    def _getitem_sequential(self, idx):
        if idx + self.sequence_length > len(self.all_sequences):
            raise IndexError("Index out of range for sequence extraction (sequential mode).")

        frame_paths = self.all_sequences[idx : idx + self.sequence_length]
        year_str, month_str = parse_year_month_from_filename(frame_paths[0])

        frames = []
        for fp in frame_paths:
            frame_data = self._load_frame(fp)
            if frame_data is None:
                print(f"Warning: Skipping unreadable file {fp}.")
                return None
            frames.append(frame_data)

        frames_tensor = torch.tensor(np.stack(frames), dtype=torch.float64)
        #
        # unique = frames_tensor.unique()
        # if unique.numel() == 1:
        #     print(f"###################Unique: {unique} #######################\n")
        #
        return frames_tensor, (year_str, month_str)

    def _getitem_random(self, idx):
        max_attempts = 5
        attempts = 0
        while attempts < max_attempts:
            if idx + self.sequence_length > len(self.all_sequences):
                raise IndexError("Index out of range for sequence extraction (random mode).")

            last_frame_path = self.all_sequences[idx + self.sequence_length - 1]
            last_frame_data = self._load_frame(last_frame_path)
            if last_frame_data is not None:
                frame_paths = self.all_sequences[idx : idx + self.sequence_length]
                year_str, month_str = parse_year_month_from_filename(frame_paths[0])
                frames = [self._load_frame(fp) for fp in frame_paths]
                if all(f is not None for f in frames):
                    frames_tensor = torch.tensor(np.stack(frames), dtype=torch.float64)
                    return frames_tensor, (year_str, month_str)
                else:
                    print(
                        f"One or more frames are None for idx range {idx}~{idx + self.sequence_length - 1}, "
                        "retrying with random index..."
                    )
                    idx = random.randint(0, len(self.all_sequences) - self.sequence_length)
                    attempts += 1
            else:
                print(f"Last frame is None for idx {idx}, retrying with random index...")
                idx = random.randint(0, len(self.all_sequences) - self.sequence_length)
                attempts += 1
        raise ValueError("No valid sequence found after multiple attempts in random mode.")


# =========================
# Utility to print bucket distribution
# =========================
def print_distribution_formatted(distribution_stats, intervals):
    print(f"{'Interval':<25}{'Count':>12}{'Percentage':>12}")
    for (lower, upper), (count_val, pct_val) in distribution_stats.items():
        if lower == upper:
            # e.g. (0.0, 0.0)
            interval_str = f"= {lower}"
        elif upper == float("inf"):
            interval_str = f"({lower}, ∞)"
        else:
            interval_str = f"({lower}, {upper}]"
        print(f"{interval_str:<25}{count_val:>12}{pct_val:>12.2f}")


# =========================
# Bucket trend visualization function
# =========================
def plot_bucket_trend_over_batches(json_path, out_fig=BASE_PATH + "bucket_trend.png", skip_bucket_indices=None):
    if skip_bucket_indices is None:
        skip_bucket_indices = []

    with open(json_path) as f:
        debug_data = json.load(f)

    intervals = [tuple(x) for x in debug_data["intervals"]]
    batch_info = debug_data["batches"]

    batch_info_sorted = sorted(batch_info, key=lambda x: x.get("batch_index", 0))
    batch_indices = [entry["batch_index"] for entry in batch_info_sorted]
    n_buckets = len(intervals)

    bucket_values = [[] for _ in range(n_buckets)]
    for entry in batch_info_sorted:
        dist_diff = entry.get("distribution_diff", None)
        if dist_diff is None:
            for j in range(n_buckets):
                bucket_values[j].append(0.0)
            continue
        for j in range(n_buckets):
            val = dist_diff[j] if j < len(dist_diff) else 0.0
            bucket_values[j].append(val)

    plt.figure(figsize=(10, 6))
    texts = []
    points = []  # if we need scatter for arrow

    for j in range(n_buckets):
        if j in skip_bucket_indices:
            continue

        y_values = bucket_values[j]
        max_value = max(y_values)
        max_index = y_values.index(max_value)
        max_batch = batch_indices[max_index]

        (line,) = plt.plot(batch_indices, y_values, label=f"Bucket {intervals[j]}")
        line_color = line.get_color()

        # Place text near the highest point
        txt = plt.text(
            max_batch,
            max_value,
            f"{max_value:.2f}",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="bottom",
            color=line_color,
        )
        texts.append(txt)

    plt.title("Distribution Diff Over Batches")
    plt.xlabel("Batch Index")
    plt.ylabel("Bucket Increment")
    plt.legend()
    plt.grid(True)

    # Use adjust_text to reduce overlap
    adjust_text(
        texts,
        x=[p[0] for p in points],
        y=[p[1] for p in points],
        arrowprops={"arrowstyle": "->", "color": "gray", "lw": 0.5},
    )
    adjust_text(texts)

    plt.savefig(out_fig)
    plt.close()
    print(f"[Info] Bucket trend plot saved to {out_fig}")


def plot_aggregate_bucket_distribution(
    json_path, out_fig=BASE_PATH + "aggregate_bucket_distribution.png", skip_bucket_indices=None
):
    """
    From debug_info JSON, aggregate distribution_diff of all batches to get total counts in each bucket,
    and draw a bar chart for an intuitive look at data distribution across buckets.
    """
    if skip_bucket_indices is None:
        skip_bucket_indices = []

    with open(json_path) as f:
        debug_data = json.load(f)

    intervals = [tuple(x) for x in debug_data["intervals"]]
    batch_info = debug_data["batches"]

    n_buckets = len(intervals)
    total_counts = [0.0] * n_buckets

    # Sum the increments from each batch
    for entry in batch_info:
        dist_diff = entry.get("distribution_diff", None)
        if dist_diff is None:
            continue
        for j in range(n_buckets):
            total_counts[j] += dist_diff[j]

    # Skip buckets we do not want to plot
    final_intervals = []
    final_counts = []
    for j in range(n_buckets):
        if j in skip_bucket_indices:
            continue
        final_intervals.append(intervals[j])
        final_counts.append(total_counts[j])

    # Draw a bar chart
    plt.figure(figsize=(10, 6))

    x_labels = []
    for lower, upper in final_intervals:
        if lower == upper:
            x_labels.append(f"= {lower}")
        elif upper == float("inf"):
            x_labels.append(f"({lower}, ∞)")
        else:
            x_labels.append(f"({lower}, {upper}]")

    x_positions = range(len(final_counts))
    plt.bar(x_positions, final_counts, color="skyblue", alpha=0.8)

    plt.xticks(x_positions, x_labels, rotation=45, ha="right")
    plt.title("Aggregate Bucket Distribution (All Batches Summed)")
    plt.xlabel("Rainfall Intervals")
    plt.ylabel("Total Counts")
    plt.grid(axis="y")
    plt.tight_layout()

    plt.savefig(out_fig)
    plt.close()
    print(f"[Info] Aggregate bucket distribution chart saved to {out_fig}")


def plot_month_year_3d(
    json_path,
    out_fig=BASE_PATH + "month_year_3d.png",
    plot_metric="yearly_cumulative",  # "avg_rainfall"/"avg_intensity"/"yearly_cumulative"
    skip_bucket_indices=None,
    figsize=(12, 8),
    offset_scale=0.10,
):
    if skip_bucket_indices is None:
        skip_bucket_indices = []

    with open(json_path) as f:
        debug_data = json.load(f)

    intervals = [tuple(x) for x in debug_data["intervals"]]
    batches = debug_data["batches"]

    # 1) Depending on plot_metric, prepare year_month -> Z
    year_month_to_value = {}

    if plot_metric == "yearly_cumulative":
        # Looking for "batch_index":"final" entry
        final_entry = None
        for binfo in batches:
            if binfo.get("monthly_accumulation_stats") is not None:
                final_entry = binfo
                break
        if not final_entry:
            print("[Warning] No final monthly accumulation found in debug_info!")
            return

        stats_dict = final_entry.get("monthly_accumulation_stats", {})
        # We'll read "year_month_cumulative_rainfall"
        cum_dict = stats_dict.get("year_month_cumulative_rainfall", {})
        # cum_dict like: { "2022-08": 1234.5, ... }
        year_month_to_value = cum_dict

    else:
        # avg_rainfall or avg_intensity: must traverse year_month_updates
        aggregator_sum = defaultdict(float)
        aggregator_count = defaultdict(float)

        for binfo in batches:
            year_month_updates = binfo.get("year_month_updates", {})
            for ym_key, upd in year_month_updates.items():
                if plot_metric == "avg_rainfall":
                    val_sum = upd.get("sum_increment_rainfall", 0.0)
                    val_count = upd.get("count_increment_rainfall", 0.0)
                else:  # "avg_intensity"
                    val_sum = upd.get("sum_increment", 0.0)
                    val_count = upd.get("count_increment", 0.0)

                aggregator_sum[ym_key] += val_sum
                aggregator_count[ym_key] += val_count

        # compute year_month_to_value
        for ym_key in aggregator_sum:
            s = aggregator_sum[ym_key]
            c = aggregator_count[ym_key]
            if c > 0:
                year_month_to_value[ym_key] = s / c
            else:
                year_month_to_value[ym_key] = 0.0

    # 2) Gather all years, months
    all_years = set()
    all_months = set()
    for ym_key in year_month_to_value:
        year_str, month_str = ym_key.split("-")
        all_years.add(int(year_str))
        all_months.add(int(month_str))

    sorted_years = sorted(all_years)
    sorted_months = sorted(all_months)

    # 3) Build x=month, y=year, z= year_month_to_value
    year_to_y = {y: idx for idx, y in enumerate(sorted_years)}
    month_to_x = {m: idx for idx, m in enumerate(sorted_months)}

    # 4) Prepare colors (cmap) + skip_bucket_indices
    n_buckets = len(intervals)
    cmap = plt.cm.get_cmap("rainbow", n_buckets)
    color_list = [cmap(i) for i in range(n_buckets)]

    def find_bucket_idx(z):
        # find the bucket in which z falls
        for idx, (lower, upper) in enumerate(intervals):
            if idx in skip_bucket_indices:
                pass
            if math.isclose(lower, upper, abs_tol=1e-9):
                if math.isclose(z, lower, abs_tol=1e-9):
                    return idx
            if z > lower and z <= upper:
                return idx
        if z <= intervals[0][0]:
            return 0
        return n_buckets - 1

    # 5) Plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    for ym_key, z_val in year_month_to_value.items():
        year_str, month_str = ym_key.split("-")
        y_val = int(year_str)
        m_val = int(month_str)

        if y_val not in year_to_y or m_val not in month_to_x:
            continue

        xx = month_to_x[m_val]
        yy = year_to_y[y_val]

        bucket_idx = find_bucket_idx(z_val)
        if bucket_idx in skip_bucket_indices:
            continue

        color = color_list[bucket_idx]

        ax.scatter(xx, yy, z_val, c=[color], alpha=0.8, s=100, edgecolors="k")

    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    metric_title = {
        "avg_rainfall": "Average Rainfall (mm)",
        "avg_intensity": "Average Intensity (mm/hr)",
        "yearly_cumulative": "Yearly Cumulative Rainfall (mm)",
    }.get(plot_metric, plot_metric)

    ax.set_zlabel(metric_title)

    # set ticks
    ax.set_xticks([month_to_x[m] for m in sorted_months])
    ax.set_xticklabels([f"{m:02d}" for m in sorted_months])
    ax.set_yticks([year_to_y[y] for y in sorted_years])
    ax.set_yticklabels([str(y) for y in sorted_years])

    # build legend
    legend_patches = []
    for j, (lower, upper) in enumerate(intervals):
        if j in skip_bucket_indices:
            continue
        if math.isclose(lower, upper, abs_tol=1e-9):
            label_str = f"= {lower}"
        elif math.isinf(upper):
            label_str = f"({lower}, ∞)"
        else:
            label_str = f"({lower}, {upper}]"
        patch = mpatches.Patch(color=color_list[j], label=label_str)
        legend_patches.append(patch)

    ax.legend(handles=legend_patches, title="Buckets", loc="upper left", bbox_to_anchor=(1.05, 1.0))

    plt.tight_layout()
    plt.savefig(out_fig, bbox_inches="tight")
    plt.close()
    print(f"[Info] 3D plot with metric='{plot_metric}' saved to {out_fig}")


# =========================
# Core Statistics Class
# =========================
class RainfallDistributionCalculator:
    def __init__(self, data, device="cpu"):
        # define bucket ranges
        self.intervals = [
            (0.0, 0.0),
            (0.0, 0.1),
            (0.1, 1.0),
            (1.0, 4.0),
            (4.0, 10.0),
            (10.0, 15.0),
            (15.0, 25.4),
            (25.4, 50.8),
            (50.8, 76.2),
            (76.2, 101.6),
            (101.6, 128.0),
            (128.0, float("inf")),
            # (127.0, 152.4),
            # (152.4, 177.8),
            # (177.8, 203.2),
            # (203.2, float('inf'))
        ]
        # self.intervals = [
        #     (0.0, 14.514285714285714),
        #     (14.514285714285714, 29.02857142857143),
        #     (29.02857142857143, 43.542857142857144),
        #     (43.542857142857144, 58.05714285714286),
        #     (58.05714285714286, 72.57142857142857),
        #     (72.57142857142857, 87.08571428571429),
        #     (87.08571428571429, 101.6),
        #     (101.6, 116.11428571428571),
        #     (116.11428571428571, 130.62857142857143),
        #     (130.62857142857143, 145.14285714285714),
        #     (145.14285714285714, 159.65714285714284),
        #     (159.65714285714284, 174.17142857142858),
        #     (174.17142857142858, 188.68571428571428),
        #     (188.68571428571428, 203.2),
        #     (203.2, float('inf'))
        # ]

        self.device = device
        self.data = data
        self.num_intervals = len(self.intervals)

        # bucket counts
        self.distribution = torch.zeros(self.num_intervals, dtype=torch.int64, device=self.device)
        self.invalid_values = []
        self.total_count = 0

        # monthly stats
        self.month2sum = defaultdict(float)
        self.month2count = defaultdict(float)
        self.month2sum_rainfall = defaultdict(float)
        self.month2count_rainfall = defaultdict(float)

        # debug info
        self.debug_info = []
        self.batch_index = 0

    def calculate_distribution(self, num_workers=4, convert_to_rainfall=False):
        """
        Traverse the dataset, do bucket statistics, and optionally do monthly rainfall stats (mm).
        """
        loader = DataLoader(
            self.data, batch_size=1, num_workers=num_workers, collate_fn=skip_none_collate_fn, pin_memory=False
        )

        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue
                frames, year_month_list = batch
                frames = frames.to(self.device, non_blocking=True)

                # 1) Simple log: min, max, negative count, >128
                batch_min = float(frames.min().item())
                batch_max = float(frames.max().item())
                batch_neg_count = int((frames < 0).sum().item())
                batch_gt128_count = int((frames > 128).sum().item())

                old_total_count = self.total_count

                # 2) monthly stats (clip negative->0, clip >128->128)
                frames_pos = torch.where(frames > 0, frames, torch.zeros_like(frames))
                frames_clamped = torch.clamp(frames_pos, max=127)

                # 2.1) bucket
                # distribution_diff = self._count_intervals(frames_clamped)
                distribution_diff = self.distribution
                self.total_count += frames_clamped.numel()

                if frames_clamped.ndim == 5:
                    batch_sum_pos = frames_clamped.sum(dim=[1, 2, 3, 4])
                    batch_count_pos = (frames_clamped > 0).sum(dim=[1, 2, 3, 4])
                else:
                    batch_sum_pos = frames_clamped.sum(dim=[1])
                    batch_count_pos = (frames_clamped > 0).sum(dim=[1])

                if convert_to_rainfall:
                    frames_rainfall = frames_clamped * DELTA_T_HR
                    if frames_rainfall.ndim == 5:
                        batch_sum_rainfall = frames_rainfall.sum(dim=[1, 2, 3, 4])
                        batch_count_rainfall = (frames_rainfall > 0).sum(dim=[1, 2, 3, 4])
                    else:
                        batch_sum_rainfall = frames_rainfall.sum(dim=[1])
                        batch_count_rainfall = (frames_rainfall > 0).sum(dim=[1])
                else:
                    batch_sum_rainfall = None
                    batch_count_rainfall = None

                # 4) Update month2sum for each sample
                year_month_updates = {}
                for i, (year_str, month_str) in enumerate(year_month_list):
                    if isinstance(year_str, bytes):
                        year_str = year_str.decode()
                    if isinstance(month_str, bytes):
                        month_str = month_str.decode()

                    val_sum = float(batch_sum_pos[i].item())
                    val_count = float(batch_count_pos[i].item())
                    self.month2sum[month_str] += val_sum
                    self.month2count[month_str] += val_count

                    rr_sum, rr_count = 0.0, 0.0
                    if convert_to_rainfall and (batch_sum_rainfall is not None):
                        rr_sum = float(batch_sum_rainfall[i].item())
                        rr_count = float(batch_count_rainfall[i].item())
                        self.month2sum_rainfall[month_str] += rr_sum
                        self.month2count_rainfall[month_str] += rr_count

                    ym_key = f"{year_str}-{month_str}"
                    if ym_key not in year_month_updates:
                        year_month_updates[ym_key] = {
                            "sum_increment": 0.0,
                            "count_increment": 0.0,
                            "sum_increment_rainfall": 0.0,
                            "count_increment_rainfall": 0.0,
                        }
                    year_month_updates[ym_key]["sum_increment"] += val_sum
                    year_month_updates[ym_key]["count_increment"] += val_count
                    if convert_to_rainfall:
                        year_month_updates[ym_key]["sum_increment_rainfall"] += rr_sum
                        year_month_updates[ym_key]["count_increment_rainfall"] += rr_count

                # 5) Store to debug_info
                batch_log = {
                    "batch_index": self.batch_index,
                    "frames_shape": list(frames.shape),
                    "batch_min": batch_min,
                    "batch_max": batch_max,
                    "num_negatives": batch_neg_count,
                    "num_over_128": batch_gt128_count,
                    "distribution_diff": distribution_diff,
                    "old_total_count": old_total_count,
                    "new_total_count": self.total_count,
                    "year_month_updates": year_month_updates,
                }
                self.debug_info.append(batch_log)
                self.batch_index += 1

            self.finalize_monthly_accumulation()

    # def _count_intervals(self, frames: torch.Tensor):
    #     data = frames.flatten()
    #
    #     # Take all values == 0
    #     zero_mask = (data == 0.0)
    #     zero_count = zero_mask.sum().item()
    #     self.distribution[0] += zero_count
    #
    #     # get all value != 0
    #     data_nonzero = data[~zero_mask]
    #     if data_nonzero.numel() == 0:
    #         return
    #
    #     # intervals[1:] to cover (0.0, 0.1), (0.1, 1.0), ..., (128.0, inf)
    #     # so that distribution[1] is corresponded to intervals[1],
    #     # distribution[2] is corresponded to intervals[2], ...
    #     interval_bounds = torch.tensor(
    #         [up for (_, up) in self.intervals[1:]],
    #         device=self.device, dtype=torch.float64
    #     )
    #
    #     # ----④ bucketize + scatter_add
    #     hist = torch.bucketize(data_nonzero, interval_bounds, right=True)
    #     valid_mask = (hist >= 0) & (hist < (self.num_intervals - 1))
    #     hist_valid = hist[valid_mask]
    #
    #     self.distribution[1:].scatter_add_(
    #         0,
    #         hist_valid,
    #         torch.ones_like(hist_valid, dtype=torch.float64)
    #     )
    #
    #     invalid_data = data_nonzero[~valid_mask]
    #     if invalid_data.numel() > 0:
    #         self._collect_invalid_data(invalid_data)

    def _count_intervals(self, frames: torch.Tensor):
        data = frames.flatten()

        # count all 0 pixels
        zero_mask = data == 0.0
        zero_count = zero_mask.sum().item()
        self.distribution[0] += zero_count

        # count all none-0 pixels
        data_nonzero = data[~zero_mask]
        if data_nonzero.numel() == 0:
            return

        # intervals[1:] to replace (0.0, 0.1], (0.1, 1.0], ..., (128.0, inf)
        interval_bounds = torch.tensor([up for (_, up) in self.intervals[1:]], device=self.device, dtype=torch.float64)

        data_nonzero = data_nonzero.to(torch.float64)
        hist = torch.bucketize(data_nonzero, interval_bounds, right=True)
        valid_mask = (hist >= 0) & (hist < (self.num_intervals - 1))
        hist_valid = hist[valid_mask].to(torch.int64)

        self.distribution[1:].scatter_add_(0, hist_valid, torch.ones_like(hist_valid, dtype=torch.int64))

        invalid_data = data_nonzero[~valid_mask]
        if invalid_data.numel() > 0:
            self._collect_invalid_data(invalid_data)

    def _collect_invalid_data(self, invalid_data: torch.Tensor):
        if invalid_data.numel() > 0:
            self.invalid_values.append(invalid_data.cpu().numpy())

    def get_distribution_stats(self):
        """
        :return: { (lower, upper): (count, pct), ... }
        """
        if self.total_count == 0:
            # If no data counted yet
            return dict.fromkeys(self.intervals, (0, 0.0))

        total_counts_in_buckets = float(self.distribution.sum().item())
        dist_dict = {}
        for interval, count_val in zip(self.intervals, self.distribution):
            c = int(count_val.item())
            pct = round((c / total_counts_in_buckets) * 100, 2)
            dist_dict[interval] = (c, pct)
        return dist_dict

    def finalize_monthly_accumulation(self):
        """
        1) Collect year_month_updates from each batch in self.debug_info,
           to get monthly (year-month) total rainfall.
        2) Then compute average for the same 'month' across different years.
        3) Save results into self.debug_info for reference in debug_info.json
        """

        # We store "year-month" -> total rainfall
        year_month_rainfall = defaultdict(float)
        year_month_count = defaultdict(float)

        # Go through all batches in self.debug_info
        for batch_info in self.debug_info:
            year_month_updates = batch_info.get("year_month_updates", {})
            for ym_key, updates_dict in year_month_updates.items():
                # sum_increment = increment for this (year-month)
                rr_sum = updates_dict.get("sum_increment", 0.0)
                rr_count = updates_dict.get("count_increment", 0.0)
                year_month_rainfall[ym_key] += rr_sum
                year_month_count[ym_key] += rr_count

        for ym_key in year_month_rainfall:
            s = year_month_rainfall[ym_key]
            c = year_month_count[ym_key]
            if c > 0:
                year_month_rainfall[ym_key] = s / c
            else:
                year_month_rainfall[ym_key] = 0.0

        # 2) For the same month but different years, compute an average
        month_rain_agg = defaultdict(list)
        for ym_key, total_val in year_month_rainfall.items():
            # e.g. "2022-08"
            year_str, month_str = ym_key.split("-")
            month_rain_agg[month_str].append(total_val)

        month_rain_avg = {}
        for m_str, val_list in month_rain_agg.items():
            if len(val_list) == 0:
                continue
            avg_val = sum(val_list) / len(val_list)
            month_rain_avg[m_str] = avg_val

        # 3) Append a special entry in debug_info
        final_stat = {
            "year_month_cumulative_rainfall": dict(year_month_rainfall),
            "month_rain_avg_across_years": dict(month_rain_avg),
        }

        self.debug_info.append(
            {
                "batch_index": 999999,  # Mark as summary info
                "monthly_accumulation_stats": final_stat,
            }
        )
        print("[Info] finalize_monthly_accumulation done, appended stats to debug_info.")

    # -----------------------
    # Print & save invalid data
    # -----------------------
    def print_invalid_data_examples(self, max_examples=20):
        if not self.invalid_values:
            print("[Info] No invalid data found.")
            return
        all_invalid_data = np.concatenate(self.invalid_values)
        count_invalid = len(all_invalid_data)
        print(f"\n[Invalid Data Report] total invalid data: {count_invalid}")
        print(f"  - First {max_examples} examples: {all_invalid_data[:max_examples]}")

    def save_invalid_data_to_pkl(self, pkl_path=BASE_PATH + "invalid_data.pkl"):
        print(f"[Info] Invalid data saved to {pkl_path}")

    # -----------------------
    # Save & load stats
    # -----------------------
    def save_stats_to_pkl(self, pkl_path=BASE_PATH + "stats.pkl"):
        print(f"[Info] Stats saved to {pkl_path}")

    def load_stats_from_pkl(self, pkl_path=BASE_PATH + "stats.pkl"):
        print(f"[Info] Stats loaded from {pkl_path}")

    # -----------------------
    # Save debug_info to JSON and intervals
    # -----------------------
    def save_debug_info(self, json_path=BASE_PATH + "debug_info.json"):
        """
        Convert intervals to list[list], save with debug_info
        """
        intervals_list = [list(tup) for tup in self.intervals]
        output_dict = {"intervals": intervals_list, "batches": self.debug_info}
        with open(json_path, "w") as f:
            json.dump(output_dict, f, indent=2)
        print(f"[Info] Debug info (including intervals) saved to {json_path}")

    # -----------------------
    # Plot distribution / invalid data / monthly stats
    # -----------------------
    def plot_distribution_scatter(self, title="Rainfall Distribution", out_fig=BASE_PATH + "distribution_scatter.png"):
        dist_stats = self.get_distribution_stats()
        x_vals, y_vals = [], []
        for (lower, upper), (_count_val, pct) in dist_stats.items():
            if lower == upper:
                mid = 0.0
            elif upper == float("inf"):
                mid = lower + 10
            else:
                mid = (lower + upper) / 2
            x_vals.append(mid)
            y_vals.append(pct)

        plt.figure(figsize=(8, 5))
        plt.scatter(x_vals, y_vals, color="b", alpha=0.7)
        plt.title(title)
        plt.xlabel("Rainfall interval (approx. mid-point)")
        plt.ylabel("Percentage (%)")
        plt.grid(True)
        plt.savefig(out_fig)
        plt.close()
        print(f"[Info] Distribution scatter plot saved to {out_fig}")

    def plot_invalid_scatter(self, title="Invalid Data Scatter", out_fig=BASE_PATH + "invalid_scatter.png"):
        if not self.invalid_values:
            print("[Info] No invalid data found, skipping invalid data scatter plot.")
            return
        all_invalid_data = np.concatenate(self.invalid_values)
        max_points = 20000
        if len(all_invalid_data) > max_points:
            print(
                f"[Warning] Too many invalid data points ({len(all_invalid_data)}), only plotting first {max_points}."
            )
            all_invalid_data = all_invalid_data[:max_points]

        indices = np.arange(len(all_invalid_data))
        plt.figure(figsize=(8, 5))
        plt.scatter(indices, all_invalid_data, s=2, color="r", alpha=0.5)
        plt.title(title)
        plt.xlabel("Index")
        plt.ylabel("Invalid Rainfall Value")
        plt.grid(True)
        plt.savefig(out_fig)
        plt.close()
        print(f"[Info] Invalid data scatter plot saved to {out_fig}")

    def plot_monthly_average_intensity(self, out_fig=BASE_PATH + "monthly_average_intensity.png"):
        """
        Plot monthly average rainfall intensity (mm/hr)
        """
        if not self.month2sum:
            print("[Info] No monthly data available to plot.")
            return
        sorted_months = sorted(self.month2sum.keys())
        months_str = []
        avg_values = []
        for m in sorted_months:
            total_sum = self.month2sum[m]
            total_count = self.month2count[m]
            avg_val = total_sum / total_count if total_count > 0 else 0.0
            months_str.append(m)
            avg_values.append(avg_val)

        plt.figure(figsize=(8, 5))
        plt.bar(months_str, avg_values, color="skyblue", alpha=0.7, label="Avg Rainfall Intensity (bar)")
        plt.plot(months_str, avg_values, color="red", marker="o", label="Avg Rainfall Intensity (line)")
        plt.ylabel("Monthly Avg Rainfall Intensity (mm/hr)")
        plt.xlabel("Month")
        plt.title("Monthly Average Rainfall Intensity")
        plt.grid(True)
        plt.legend()
        plt.savefig(out_fig)
        plt.close()
        print(f"[Info] Monthly average intensity plot saved to {out_fig}")

    def plot_monthly_average_rainfall(self, out_fig=BASE_PATH + "monthly_average_rainfall.png"):
        """
        Plot monthly average rainfall (mm). Need convert_to_rainfall=True in calculate_distribution.
        """
        if not self.month2sum_rainfall:
            print("[Info] No monthly rainfall data available to plot.")
            return
        sorted_months = sorted(self.month2sum_rainfall.keys())
        months_str = []
        avg_values = []
        for m in sorted_months:
            total_sum = self.month2sum_rainfall[m]
            total_count = self.month2count_rainfall[m]
            avg_val = total_sum / total_count if total_count > 0 else 0.0
            months_str.append(m)
            avg_values.append(avg_val)

        plt.figure(figsize=(8, 5))
        plt.bar(months_str, avg_values, color="skyblue", alpha=0.7, label="Avg Rainfall Amount (bar)")
        plt.plot(months_str, avg_values, color="red", marker="o", label="Avg Rainfall Amount (line)")
        plt.ylabel("Monthly Avg Rainfall (mm)")
        plt.xlabel("Month")
        plt.title("Monthly Average Rainfall")
        plt.grid(True)
        plt.legend()
        plt.savefig(out_fig)
        plt.close()
        print(f"[Info] Monthly average rainfall plot saved to {out_fig}")


# =========================
# Build datasets
# =========================
def build_datasets(all_frame_dir, importance_sampled_dir, use_full_data=True, sample_size=10000):
    """
    Build all_frame_dataset and importance_sampled_dataset and decide whether to do random sampling.
    """
    # Build the original dataset
    all_frame_dataset = NetCDFDataset(
        base_dir=all_frame_dir, splits=["train"], read_mode="sequential", start_offset=START_OFFSET
    )
    importance_sampled_dataset = NetCDFDataset(
        base_dir=importance_sampled_dir, splits=["train"], read_mode="sequential", start_offset=START_OFFSET
    )

    if use_full_data:
        data_for_all_frame = all_frame_dataset
        data_for_importance = importance_sampled_dataset
    else:
        all_frame_size = len(all_frame_dataset)
        importance_size = len(importance_sampled_dataset)
        sample_size_all = min(sample_size, all_frame_size)
        sample_size_importance = min(sample_size, importance_size)

        idx_all_frame = stratified_random_sample(all_frame_size, sample_size_all, 5)
        idx_importance = stratified_random_sample(importance_size, sample_size_importance, 5)

        data_for_all_frame = Subset(all_frame_dataset, idx_all_frame)
        data_for_importance = Subset(importance_sampled_dataset, idx_importance)

    return data_for_all_frame, data_for_importance


# =========================
# Compute stats & plot for single dataset
# =========================
def compute_distribution(dataset, device="cpu", convert_to_rainfall=True, pkl_path=None, load_if_exists=False):
    """
    Compute or load stats. Returns a RainfallDistributionCalculator instance.
    """
    calculator = RainfallDistributionCalculator(data=dataset, device=device)

    if load_if_exists and pkl_path and os.path.exists(pkl_path):
        print(f"[Info] Loading stats from {pkl_path} ...")
        calculator.load_stats_from_pkl(pkl_path)
    else:
        print("[Info] Calculating distribution from scratch...")
        # num_workers depends on hardware
        calculator.calculate_distribution(num_workers=32, convert_to_rainfall=convert_to_rainfall)
        if pkl_path:
            calculator.save_stats_to_pkl(pkl_path)

    return calculator


def plot_distribution_info(calculator: RainfallDistributionCalculator, prefix="all_frame"):
    """
    Print bucket stats, invalid data, and draw scatter + invalid data scatter
    """
    # Print bucket stats
    dist_stats = calculator.get_distribution_stats()
    print(f"\n[{prefix.upper()}] Data Distribution:")
    print_distribution_formatted(dist_stats, calculator.intervals)

    # Invalid data
    calculator.print_invalid_data_examples(max_examples=20)
    calculator.save_invalid_data_to_pkl(pkl_path=f"{BASE_PATH + prefix}_invalid_data.pkl")

    # Distribution scatter
    calculator.plot_distribution_scatter(
        title=f"{prefix.capitalize()} Distribution Scatter", out_fig=f"{BASE_PATH + prefix}_distribution_scatter.png"
    )
    # Invalid data scatter
    calculator.plot_invalid_scatter(
        title=f"{prefix.capitalize()} Invalid Data Scatter", out_fig=f"{BASE_PATH + prefix}_invalid_scatter.png"
    )


def plot_monthly_stats(calculator: RainfallDistributionCalculator, prefix="all_frame", convert_to_rainfall=True):
    """
    Plot monthly average rainfall intensity and monthly average rainfall
    """
    calculator.plot_monthly_average_intensity(out_fig=f"{BASE_PATH + prefix}_monthly_avg_intensity.png")
    if convert_to_rainfall:
        calculator.plot_monthly_average_rainfall(out_fig=f"{BASE_PATH + prefix}_monthly_avg_rainfall.png")


# =========================
# Pipeline function
# =========================
def run_analysis_pipeline(
    dataset,
    prefix="all_frame",
    device="cpu",
    convert_to_rainfall=True,
    stats_pkl_path=None,
    load_stats_if_exists=False,
    debug_json_path=None,
):
    """
    Perform:
    1) Compute or load stats
    2) Distribution visualization
    3) Monthly stats visualization
    4) Save debug_info + draw bucket_trend
    """
    # 1) Compute stats
    calculator = compute_distribution(
        dataset=dataset,
        device=device,
        convert_to_rainfall=convert_to_rainfall,
        pkl_path=stats_pkl_path,
        load_if_exists=load_stats_if_exists,
    )

    # 2) Distribution visualization
    plot_distribution_info(calculator, prefix=prefix)

    # 3) Monthly visualization
    plot_monthly_stats(calculator, prefix=prefix, convert_to_rainfall=convert_to_rainfall)

    # 4) Save debug_info and draw incremental bucket trend
    if debug_json_path:
        calculator.save_debug_info(debug_json_path)
        # Directly call the previously defined plot function
        skip_bucket_indices = []
        plot_bucket_trend_over_batches(
            json_path=BASE_PATH + "all_frame_debug.json",
            out_fig=f"{BASE_PATH}all_frame_bucket_trend_skip{skip_bucket_indices}.png",
            skip_bucket_indices=skip_bucket_indices,
        )
        print(f"Bucket trend plotting done (skip the {skip_bucket_indices}th bucket)!")

        json_path = BASE_PATH + "all_frame_debug.json"
        out_fig = BASE_PATH + "all_frame_bucket_distribution_scatter.png"
        plot_month_year_3d(json_path=json_path, out_fig=out_fig, skip_bucket_indices=skip_bucket_indices)


# =========================
# main() Entry
# =========================
def main():
    start_time = time.time()
    print("Starting analysis...")

    # Configuration
    all_frame_dir = Config.ORIG_DATA_DIR
    importance_sampled_dir = Config.DATA_DIR
    use_full_data = True
    sample_size = 1000
    convert_to_rainfall = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Build dataset
    data_for_all_frame, data_for_importance = build_datasets(
        all_frame_dir, importance_sampled_dir, use_full_data=use_full_data, sample_size=sample_size
    )

    # 2) all_frame dataset
    run_analysis_pipeline(
        dataset=data_for_all_frame,
        prefix="all_frame",
        device=device,
        convert_to_rainfall=convert_to_rainfall,
        stats_pkl_path=BASE_PATH + "all_frame_stats.pkl",
        load_stats_if_exists=False,
        debug_json_path=BASE_PATH + "all_frame_debug.json",
    )

    # 3) importance_sampled dataset
    run_analysis_pipeline(
        dataset=data_for_importance,
        prefix="importance_sampled",
        device=device,
        convert_to_rainfall=convert_to_rainfall,
        stats_pkl_path=BASE_PATH + "importance_stats.pkl",
        load_stats_if_exists=False,
        debug_json_path=BASE_PATH + "importance_debug.json",
    )

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")


# =========================
# Example: only draw bucket_trend from JSON
# =========================
def example_only_plot_bucket_trend():
    """
    Demonstration: if we already have a debug_info file (like "all_frame_debug.json"), we want to draw
    the incremental bucket changes only, skipping the 0th bucket (which is usually (0.0, 0.0)).
    """
    skip_bucket_indices = [0]
    plot_bucket_trend_over_batches(
        json_path=BASE_PATH + "importance_debug.json",
        out_fig=f"{BASE_PATH}Importance_frame_bucket_trend_skip{skip_bucket_indices}.png",
        skip_bucket_indices=skip_bucket_indices,
    )
    print(f"Bucket trend plotting done (skip the {skip_bucket_indices}th bucket)!")


def example_only_plot_aggregate():
    """
    Suppose we already have a debug_info file, we want to sum distribution_diff across all batches,
    and draw a bar chart, skipping the first 2 buckets.
    """
    json_path = BASE_PATH + "all_frame_debug.json"
    out_fig = BASE_PATH + "all_frame_aggregate_distribution.png"
    skip_bucket_indices = []

    plot_aggregate_bucket_distribution(json_path=json_path, out_fig=out_fig, skip_bucket_indices=skip_bucket_indices)
    print("Aggregate distribution done!")


def example_only_plot_scatter():
    json_path = BASE_PATH + "importance_debug.json"
    out_fig = BASE_PATH + "Importance_frame_bucket_distribution_scatter.png"
    skip_bucket_indices = []
    plot_month_year_3d(json_path=json_path, out_fig=out_fig, skip_bucket_indices=skip_bucket_indices)


if __name__ == "__main__":
    # # If we want the full pipeline, call main()
    main()

    # # start with offset
    # for i in range(7):
    #     BASE_PATH=f"{i * 100000}_start_offset/"
    #     START_OFFSET = i * 100000
    #     try:
    #         os.makedirs(BASE_PATH, exist_ok=True)
    #         print(f"{BASE_PATH} created")
    #     except OSError as e:
    #         print(f"creating base path error: {e}")
    #     main()

    # # If we only want to demonstrate "draw bucket_trend from existing debug_info JSON"
    # # while skipping the 0th bucket, we can do:
    # example_only_plot_bucket_trend()
    # example_only_plot_aggregate()
    # example_only_plot_scatter()
