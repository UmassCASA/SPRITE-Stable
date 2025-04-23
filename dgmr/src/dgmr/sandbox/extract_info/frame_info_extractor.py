# frame_info_extractor.py

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from datasets import load_dataset


NUM_INPUT_FRAMES = 4
NUM_TARGET_FRAMES = 4

# Get the directory of the file
directory = os.path.dirname(os.path.abspath(__file__))


def extract_input_and_target_frames(radar_frames):
    """Extract input and target frames from a dataset row's radar_frames."""
    input_frames = radar_frames[-NUM_TARGET_FRAMES - NUM_INPUT_FRAMES : -NUM_TARGET_FRAMES]
    target_frames = radar_frames[-NUM_TARGET_FRAMES:]
    return input_frames, target_frames


class TFDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, split):
        super().__init__()
        self.reader = load_dataset("openclimatefix/nimrod-uk-1km", "sample", split=split, streaming=True)
        self.iter_reader = self.reader

    def __len__(self):
        return 100

    def __getitem__(self, item):
        try:
            row = next(self.iter_reader)
        except Exception:
            rng = np.random.default_rng()
            self.iter_reader = iter(self.reader.shuffle(seed=rng.integers(low=0, high=100000), buffer_size=1000))
            row = next(self.iter_reader)
        input_frames, target_frames = extract_input_and_target_frames(row["radar_frames"])
        timestamp = row["end_time_timestamp"]
        radar_mask = row["radar_mask"]

        # get 'osgb_extent_top', 'osgb_extent_bottom', 'osgb_extent_left', 'osgb_extent_right'
        osgb_extent_top = row["osgb_extent_top"]
        osgb_extent_bottom = row["osgb_extent_bottom"]
        osgb_extent_left = row["osgb_extent_left"]
        osgb_extent_right = row["osgb_extent_right"]
        print(osgb_extent_top, osgb_extent_bottom, osgb_extent_left, osgb_extent_right)

        # get

        # print all columns in row
        print(row.keys())

        return (
            np.moveaxis(input_frames, [0, 1, 2, 3], [0, 2, 3, 1]),
            np.moveaxis(target_frames, [0, 1, 2, 3], [0, 2, 3, 1]),
            timestamp,
            radar_mask,
        )


def create_dataframe(dataloader):
    data = {
        "Timestamp": [],
        "Frame Type": [],
        "Frame Data": [],
        "Radar Mask": [],
        "Shape": [],
    }
    for batch in dataloader:
        input_frames, target_frames, timestamps, radar_masks = batch

        for i in range(len(input_frames)):
            for j, frame in enumerate(input_frames[i]):
                data["Timestamp"].append(timestamps[i].item())
                data["Frame Type"].append("Input")
                data["Frame Data"].append(frame.numpy())
                data["Radar Mask"].append(radar_masks[i][j].numpy())
                # add shape
                data["Shape"].append(frame.shape)
            for j, frame in enumerate(target_frames[i]):
                data["Timestamp"].append(timestamps[i].item())
                data["Frame Type"].append("Target")
                data["Frame Data"].append(frame.numpy())
                data["Radar Mask"].append(radar_masks[i][j + len(input_frames[i])].numpy())
                # add shape
                data["Shape"].append(frame.shape)
    return pd.DataFrame(data)


def main():
    test_dataset = TFDataset(split="test")
    test_dataloader = DataLoader(test_dataset, batch_size=10)  # Adjust batch_size as needed
    frame_info_df = create_dataframe(test_dataloader)
    print(frame_info_df.head())

    # save dataframe to csv to the current directory
    frame_info_df.to_csv(os.path.join(directory, "frame_info.csv"), index=False)


if __name__ == "__main__":
    main()
