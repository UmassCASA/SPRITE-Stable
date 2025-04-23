import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import csv


class TFDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        super().__init__()
        self.reader = load_dataset("openclimatefix/nimrod-uk-1km", "sample", split=split, streaming=True)
        self.iter_reader = self.reader

    def __len__(self):
        return 1000

    def __getitem__(self, item):
        try:
            row = next(self.iter_reader)
        except Exception:
            rng = np.random.default_rng()
            self.iter_reader = iter(self.reader.shuffle(seed=rng.integers(low=0, high=100000), buffer_size=1000))
            row = next(self.iter_reader)
        radar_frame = row["radar_frames"][0]
        # Extract the first input frame's max value
        max_value = np.max(radar_frame)
        return max_value


def process_and_save_to_csv(dataset, output_csv_path):
    """Processes the dataset to find maximum values and save to CSV."""
    with open(output_csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "max_value"])
        i = 0
        for frames_max in DataLoader(dataset, batch_size=1):
            writer.writerow([i, frames_max.item()])
            # print(f"Processed and wrote max value: {frames_max.item()} to CSV")
            i += 1

            if i % 100 == 0:
                print(f"Processed {i} records.")


# Initialize dataset
dataset = TFDataset(split="train")

# Define output path
output_csv_path = "/work/pi_mzink_umass_edu/SPRITE/skillful_nowcasting/output/max_values_nimrod.csv"

# Process dataset and save to CSV
process_and_save_to_csv(dataset, output_csv_path)

print("Completed writing max values to CSV.")
