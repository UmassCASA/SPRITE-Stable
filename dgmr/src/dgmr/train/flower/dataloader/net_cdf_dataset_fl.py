import os
import warnings

import numpy as np
from netCDF4 import Dataset

from sprite_core.config import Config

import torch.utils.data.dataset

import cv2


# TODO: adapt to casa-datatool
class NetCDFDataset(torch.utils.data.dataset.Dataset):
    """
    Typically, dataset returns an individual item from the dataset in __getitem__ method.
    Also, it should return the number of items in the dataset in __len__ method.

    Here, dataset returns a batch of frames in __getitem__ method, therefore __len__ method returns the number of
    batches.
    Also, DGMRDataModule should have batch_size=1, since we are returning a batch of frames in __getitem__ method.
    """

    def __init__(
        self, split, num_input_frams, num_total_frams, client_id=0, num_clients=1, data_shard=0, total_shards=1
    ):
        super().__init__()
        self.split = split
        self.local_folder_path = os.path.join(Config.DATA_DIR, split)
        self.all_sequences = self._get_all_sequences()
        self.num_input_frams = num_input_frams
        self.num_total_frams = num_total_frams

        # TODO: adapt new data split strategy for clients
        self.client_id = client_id
        self.num_clients = num_clients
        self.data_shard = data_shard
        self.total_shards = total_shards

        # Check if the input image size can be divided by the number of clients
        self.image_size = 256  # Assuming the original image size is 256x256
        self.num_splits = int(np.sqrt(num_clients))
        if self.image_size % self.num_splits != 0:
            raise ValueError(f"Image size {self.image_size} cannot be evenly divided by {self.num_splits}")

        self.crop_size = self.image_size // self.num_splits

        self.client_sequences = self._partition_data()
        self.local_folder_path = os.path.join(Config.DATA_DIR, split)
        self.all_sequences = self._get_all_sequences()
        # split data by `client_id` and `num_clients`
        self.client_sequences = self._partition_data()

    def _partition_data(self):
        # TODO: adapt new data split strategy for clients
        # Simply divide the sequence list by the number of clients, and number of data_shard
        total_sequences = len(self.all_sequences)
        sequences_per_shard = total_sequences // self.total_shards

        start_idx = self.data_shard * sequences_per_shard
        end_idx = start_idx + sequences_per_shard

        # prevent over-index
        if end_idx > total_sequences:
            end_idx = total_sequences

        return self.all_sequences[start_idx:end_idx]

    def __len__(self):
        """
        Return size of the data set for DataLoader, but if Dataset gives complete batches
        and not individual items from the dataset, then it should return total number of batches
        """
        # return len(self.all_sequences)
        return len(self.client_sequences)

    def _get_all_sequences(self):
        sequences = sorted([d for d in os.listdir(self.local_folder_path) if d.startswith("seq-")])
        return [os.path.join(self.local_folder_path, seq) for seq in sequences]

    def _check_batch(self, frames, batch_idx, frame_type):
        # Check if all frames are not a type of masked array
        if all(isinstance(f, np.ma.MaskedArray) for f in frames):
            raise ValueError(f"Frame(s) of batch {batch_idx} have masked array in {frame_type} frames")

    def _load_frame(self, file_path):
        with Dataset(file_path, "r") as nc_data:
            return np.ma.filled(nc_data.variables["RRdata"][:], 0)

    def _crop_and_resize(self, image, client_id):
        """
        Crop the image based on client_id and then resize back to original size.
        """
        if self.num_clients == 1:
            return image

        # print(f"image.shape: {image.shape}")
        # If the image has a batch dimension (e.g., [1, 256, 256]), remove it for processing
        if len(image.shape) == 3 and image.shape[0] == 1:
            image = image[0]  # Remove batch dimension, image shape becomes [256, 256]

        # Calculate the row and column indices for cropping
        row_idx = client_id // self.num_splits
        col_idx = client_id % self.num_splits
        # print(f"row_idx: {row_idx}, col_idx: {col_idx}")

        # Crop the image to the corresponding part
        start_row = row_idx * self.crop_size
        start_col = col_idx * self.crop_size
        # print(f"start_row: {start_row}, start_col: {start_col}")
        cropped_image = image[start_row : start_row + self.crop_size, start_col : start_col + self.crop_size]

        # print(f"cropped_image.shape: {cropped_image.shape}")
        # Resize the cropped image back to original size using bilinear interpolation
        resized_image = cv2.resize(cropped_image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        # Add the batch dimension back to match the original shape
        resized_image = np.expand_dims(resized_image, axis=0)

        return resized_image

    def __getitem__(self, idx):
        """Returns one sequence(batch) of frames"""
        # seq_path = self.all_sequences[idx]
        seq_path = self.client_sequences[idx]
        frame_paths = sorted([os.path.join(seq_path, f) for f in os.listdir(seq_path)])

        frames = [self._load_frame(fp) for fp in frame_paths]

        # check if frames contains empty datat
        empty_indices = [i for i, frame in enumerate(frames) if frame.size == 0]

        if empty_indices:
            for empty_idx in empty_indices:
                # looking for previous/next none-empty data
                prev_idx = next((i for i in range(empty_idx - 1, -1, -1) if frames[i].size != 0), None)
                next_idx = next((i for i in range(empty_idx + 1, len(frames)) if frames[i].size != 0), None)

                if prev_idx is not None and next_idx is not None:
                    # Replacement of null data with a weighted average of pre- and post-null data
                    dist_prev = empty_idx - prev_idx
                    dist_next = next_idx - empty_idx
                    frames[empty_idx] = (frames[prev_idx] * dist_next + frames[next_idx] * dist_prev) / (
                        dist_prev + dist_next
                    )
                elif prev_idx is not None:
                    frames[empty_idx] = frames[prev_idx]
                elif next_idx is not None:
                    frames[empty_idx] = frames[next_idx]
                else:
                    raise ValueError(f"All frames are empty in sequence: {seq_path}")

            warnings.warn(f"Empty frame(s) found and replaced in sequence: {seq_path}", stacklevel=2)

        # input_frames = np.stack(frames[:self.num_input_frams])
        # target_frames = np.stack(frames[self.num_input_frams:self.num_total_frams])
        input_frames = np.stack(
            [self._crop_and_resize(frame, self.client_id) for frame in frames[: self.num_input_frams]]
        )
        target_frames = np.stack(
            [
                self._crop_and_resize(frame, self.client_id)
                for frame in frames[self.num_input_frams : self.num_total_frams]
            ]
        )

        # Checks for the entire batch
        self._check_batch(input_frames, idx, "input")
        self._check_batch(target_frames, idx, "target")

        return input_frames, target_frames
