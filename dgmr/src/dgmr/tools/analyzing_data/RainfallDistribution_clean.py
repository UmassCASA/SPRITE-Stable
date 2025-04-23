import time
import os
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from netCDF4 import Dataset
import random
from sprite_core.config import Config
from functools import lru_cache

# Set data directory paths
all_frame_dir = Config.ORIG_DATA_DIR
importance_sampled_dir = Config.DATA_DIR


# NetCDF data reading class
class NetCDFDataset(torch.utils.data.Dataset):
    """Class for loading the dataset from the netCDF files."""

    def __init__(self, base_dir, splits, sequence_length=15):
        super().__init__()
        self.splits = splits
        self.sequence_length = sequence_length
        self.local_folder_paths = [os.path.join(base_dir, split) for split in splits]
        self.all_sequences = self._get_all_sequences()

    def __len__(self):
        return len(self.all_sequences) - self.sequence_length + 1

    def _get_all_sequences(self):
        sequences = []
        for folder_path in self.local_folder_paths:
            if os.path.exists(folder_path):
                for root, _, files in os.walk(folder_path):
                    sequences += [os.path.join(root, f) for f in files if f.endswith(".nc")]
        return sorted(sequences)

    @staticmethod
    @lru_cache(maxsize=128)
    def _load_frame(file_path, retries=1):
        for _ in range(retries):
            try:
                with Dataset(file_path, "r") as nc_data:
                    # Attempt to access RRdata variable
                    rrdata = nc_data.variables.get("RRdata", None)
                    if rrdata is None:
                        raise KeyError(f"Variable 'RRdata' not found in file {file_path}")
                    return np.ma.filled(rrdata[...], 0)
            except (OSError, KeyError) as e:
                # Handle both file-related and missing-variable errors
                print(f"Error encountered while loading file {file_path}: {e}. Retrying...")
        return None

    def __getitem__(self, idx):
        max_attempts = 5  # Reduce maximum attempts
        attempts = 0
        while attempts < max_attempts:
            if idx + self.sequence_length > len(self.all_sequences):
                raise IndexError("Index out of range for sequence extraction")
            # Only check if the 15th frame exists
            frame_path = self.all_sequences[idx + self.sequence_length - 1]
            frame = self._load_frame(frame_path)
            if frame is not None:
                # If the 15th frame exists, load all frames at once
                frame_paths = self.all_sequences[idx : idx + self.sequence_length]
                frames = [self._load_frame(fp) for fp in frame_paths]
                if all(f is not None for f in frames):
                    return torch.tensor(np.stack(frames), dtype=torch.float32)
                else:
                    print(
                        f"One or more frames not found for index range {idx} to {idx + self.sequence_length - 1},"
                        "retrying with a new random index..."
                    )
                    idx = random.randint(0, len(self.all_sequences) - self.sequence_length)
                    attempts += 1
            else:
                print(f"15th frame not found for index {idx}, retrying with a new random index...")
                idx = random.randint(0, len(self.all_sequences) - self.sequence_length)
                attempts += 1
        raise ValueError("No valid sequence found after multiple attempts")


def format_interval(lower, upper, special_case=False):
    """
    Format the interval to match the structure in the image:
    - Single value: "= value"
    - Interval: (lower, upper] for all other cases.
    """
    if special_case:  # Special case for "= 0.0"
        return f"= {lower}"
    return f"({lower}, {upper}]"


def print_distribution_formatted(distribution, intervals):
    """Print the rainfall distribution with formatted intervals."""
    print(f"{'Interval':<20}{'Percentage':>10}")
    for _, (lower, upper) in enumerate(intervals):
        special_case = lower == upper  # Handle "= 0.0" case
        interval = format_interval(lower, upper, special_case=special_case)
        percentage = distribution.get((lower, upper), 0.0)
        print(f"{interval:<20}{percentage:>10.2f}")


class RainfallDistributionCalculator:
    def __init__(self, data, device="cpu"):
        self.intervals = [
            (0.0, 0.0),  # For "= 0.0", handled as a special case
            (0.0, 0.1),
            (0.1, 1.0),
            (1.0, 4.0),
            (4.0, 10.0),
            (10.0, 15.0),
            (15.0, 25.4),
            (25.4, 50.8),
            (50.8, 76.2),
            (76.2, 101.6),
            (101.6, 127.0),
            (127.0, 152.4),
            (152.4, 177.8),
            (177.8, 203.2),
            (203.2, float("inf")),  # Optional if you want to capture extreme outliers
        ]
        self.total_count = 0
        self.device = device
        self.data = data
        self.num_intervals = len(self.intervals)
        self.distribution = torch.zeros(len(self.intervals), device=self.device)

    def count_intervals(self, data):
        """Count the number of occurrences in each precipitation interval."""
        data = data.flatten()
        # Count special case for "= 0.0"
        zero_count = torch.sum(data == 0.0).item()
        self.distribution[0] += zero_count

        # For other intervals, exclude 0.0 and bucketize the remaining data
        data = data[data > 0.0]
        data = data.unsqueeze(1)  # Reshape data to (N, 1)
        interval_bounds = torch.tensor(
            [interval[0] for interval in self.intervals[1:]] + [self.intervals[-1][1]],
            device=self.device,
            dtype=torch.float32,
        )
        hist = torch.bucketize(data, interval_bounds, right=True) - 1  # Use `right=True` for (lower, upper]
        valid_indices = (hist >= 0) & (hist < self.num_intervals - 1)  # Ignore invalid indices
        hist = hist[valid_indices]
        self.distribution[1:].scatter_add_(0, hist, torch.ones_like(hist, dtype=torch.float32))

    def calculate_distribution(self):
        """Iterate over all data in the dataset to calculate the interval distribution."""
        with torch.no_grad():  # Use no_grad to avoid gradient calculation
            for frames in DataLoader(self.data, batch_size=8, num_workers=1, persistent_workers=True):
                frames = frames.to(self.device, non_blocking=True)
                self.count_intervals(frames)
                self.total_count += frames.numel()

    def get_distribution_percentage(self):
        """Get the distribution percentage for each interval."""
        if self.total_count == 0:
            return dict.fromkeys(self.intervals, 0.0)
        return {
            interval: round(float(count.cpu().item() / self.distribution.sum().cpu()) * 100, 2)
            for interval, count in zip(self.intervals, self.distribution)
        }


if __name__ == "__main__":
    start_time = time.time()  # Start timing
    print("Starting Rainfall Distribution Calculation...")

    # Create DataLoader for all-frame and sampled datasets
    all_frame_dataset = NetCDFDataset(base_dir=all_frame_dir, splits=["train", "test"])
    importance_sampled_dataset = NetCDFDataset(base_dir=importance_sampled_dir, splits=["train", "test"])

    all_frame_data_loader = DataLoader(all_frame_dataset, batch_size=1)
    importance_sampled_data_loader = DataLoader(importance_sampled_dataset, batch_size=1)

    # Create rainfall distribution calculators for all-frame and sampled datasets
    idx_all_fram = random.sample(range(len(all_frame_data_loader.dataset)), 10000)
    idx_sampled_fram = random.sample(range(len(importance_sampled_data_loader.dataset)), 10000)

    sampled_all_frame_dataset = Subset(all_frame_data_loader.dataset, idx_all_fram)
    sampled_importance_dataset = Subset(importance_sampled_data_loader.dataset, idx_sampled_fram)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_frame_calculator = RainfallDistributionCalculator(device=device, data=sampled_all_frame_dataset)
    importance_sampled_calculator = RainfallDistributionCalculator(device=device, data=sampled_importance_dataset)

    # Calculate all-frame data distribution
    all_frame_calculator.calculate_distribution()
    all_frame_distribution = all_frame_calculator.get_distribution_percentage()
    print("All Frame Data Distribution (in percentage):")
    print_distribution_formatted(all_frame_distribution, all_frame_calculator.intervals)

    # Calculate sampled data distribution
    importance_sampled_calculator.calculate_distribution()
    importance_sampled_distribution = importance_sampled_calculator.get_distribution_percentage()
    print("\nImportance Sampled Data Distribution (in percentage):")
    print_distribution_formatted(importance_sampled_distribution, importance_sampled_calculator.intervals)

    end_time = time.time()  # End timing
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
