import numpy as np
from numpy.random import default_rng
from datasets import load_dataset
import torch.utils.data.dataset

NUM_INPUT_FRAMES = 4
NUM_TARGET_FRAMES = 18


def extract_input_and_target_frames(radar_frames):
    """Extract input and target frames from a dataset row's radar_frames."""
    # We align our targets to the end of the window, and inputs precede targets.
    input_frames = radar_frames[-NUM_TARGET_FRAMES - NUM_INPUT_FRAMES : -NUM_TARGET_FRAMES]
    target_frames = radar_frames[-NUM_TARGET_FRAMES:]
    return input_frames, target_frames


class NIMRODDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, split):
        super().__init__()
        self.reader = load_dataset(
            "openclimatefix/nimrod-uk-1km", "sample", split=split, streaming=True, trust_remote_code=True
        )
        self.iter_reader = self.reader

    def __len__(self):
        return 1000

    def __getitem__(self, item):
        try:
            row = next(self.iter_reader)
        except Exception:
            rng = default_rng()
            self.iter_reader = iter(self.reader.shuffle(seed=rng.integers(low=0, high=100000), buffer_size=1000))
            row = next(self.iter_reader)

        input_frames, target_frames = extract_input_and_target_frames(row["radar_frames"])

        # Rearrange from (4, 256, 256, 1) to (4, 1, 256, 256)
        return np.moveaxis(input_frames, [0, 1, 2, 3], [0, 2, 3, 1]), np.moveaxis(
            target_frames, [0, 1, 2, 3], [0, 2, 3, 1]
        )
