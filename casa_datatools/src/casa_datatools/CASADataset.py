import netCDF4
from torch.utils.data import Dataset
import os
import numpy as np
from datetime import datetime


class CASADataset(Dataset):
    """
    Dataset returns a sequence of input and target frames.
    DataModule should have batch_size=1, since we are returning a batch of frames in __getitem__ method.
    """

    def __init__(
        self, split, num_input_frames=4, num_target_frames=18, include_datetimes=False, ensure_2d=False, data_dir=None
    ):
        """
        Args:
            split (str): The dataset split (e.g., 'train', 'validation', 'test').
            num_input_frames (int): Number of input frames.
            num_target_frames (int): Number of target frames.
            include_datetimes (bool): If True, include frame datetimes in the output.
            ensure_2d (bool): If True, ensure output frames are 2D.
            data_dir (str): Path to the data directory. Must be a valid directory path.
        """
        super().__init__()
        if data_dir is None:
            raise ValueError("data_dir must be provided")
        if not os.path.isdir(data_dir):
            raise ValueError(f"data_dir '{data_dir}' is not a valid directory")

        self.split = split
        self.include_datetimes = include_datetimes
        self.num_input_frames = num_input_frames
        self.num_target_frames = num_target_frames
        self.total_frames = self.num_input_frames + self.num_target_frames
        self.data_dir = data_dir
        self.local_folder_path = os.path.join(self.data_dir, split)
        self.all_sequences = self._get_all_sequences()
        self.ensure_2d = ensure_2d

    def __len__(self):
        """Returns the number of sequences."""
        return len(self.all_sequences)

    def _get_all_sequences(self):
        """Returns a list of all sequence paths."""
        sequences = sorted([d for d in os.listdir(self.local_folder_path) if d.startswith("seq-")])
        return [os.path.join(self.local_folder_path, seq) for seq in sequences]

    def _check_batch(self, frames, batch_idx, frame_type):
        """Checks if all frames are not a type of masked array."""
        if all(isinstance(f, np.ma.MaskedArray) for f in frames):
            raise ValueError(f"Frame(s) of batch {batch_idx} have masked array in {frame_type} frames")

    def _load_frame(self, file_path):
        """Loads a frame from a file and extracts the datetime."""
        # Extract date and time from file name
        file_name = os.path.basename(file_path)
        date_str, time_str = file_name.split("_")[0], file_name.split("_")[1].split(".")[0]
        moment_datetime = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")

        # Load the netCDF data
        with netCDF4.Dataset(file_path, "r") as nc_data:
            data = np.ma.filled(nc_data.variables["RRdata"][:], 0)

        ###### Specific for UNet ######
        # Ensure data is 2D [H, W]
        if self.ensure_2d:
            # Remove singleton dimensions | Originally [1, H, W]
            data = np.squeeze(data)

            # Ensure data has the correct shape
            if data.shape != (256, 256):
                raise ValueError(f"Unexpected data shape {data.shape} in file {file_path}")

        return data, moment_datetime

    def __getitem__(self, idx):
        """Returns one sequence (input and target frames)."""
        seq_path = self.all_sequences[idx]
        frame_paths = sorted([os.path.join(seq_path, f) for f in os.listdir(seq_path)])

        frames = []
        frame_datetimes = []
        for fp in frame_paths:
            frame, frame_datetime = self._load_frame(fp)
            frames.append(frame)
            frame_datetimes.append(frame_datetime)

        input_frames, target_frames = self._get_input_and_target_frames(frames=frames, idx=idx)

        if self.include_datetimes:
            return input_frames, target_frames, frame_datetimes
        else:
            return input_frames, target_frames

    def _get_input_and_target_frames(self, frames, idx):
        # Select input and target frames
        input_frames = frames[: self.num_input_frames]
        target_frames = frames[self.num_input_frames : self.total_frames]

        # Concatenate along the channel dimension
        input_frames = np.stack(input_frames, axis=0)  # Shape: [C * num_input_frames, H, W]
        target_frames = np.stack(target_frames, axis=0)  # Shape: [C * num_target_frames, H, W]

        # Checks for the entire batch
        self._check_batch(input_frames, idx, "input")
        self._check_batch(target_frames, idx, "target")

        return input_frames, target_frames
