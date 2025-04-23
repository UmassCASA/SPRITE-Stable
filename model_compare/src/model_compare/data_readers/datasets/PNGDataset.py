import os.path

import numpy as np

from pysteps import io, utils
import torch.utils.data.dataset
from sprite_core.config import Config

from model_compare.data_readers.pysteps_customized_importers.PNGDataImporter import PNGDataImporter


class PNGDataset(torch.utils.data.dataset.Dataset):
    """
    Typically, dataset returns an individual item from the dataset in __getitem__ method.
    Also, it should return the number of items in the dataset in __len__ method.

    Dataset returns a batch of frames in __getitem__ method, therefore __len__ method returns the number of batches.
    Also, DGMRDataModule should have batch_size=1, since we are returning a batch of frames in __getitem__ method.
    """

    def __init__(self, date, timestep=5, num_prev_files=8, num_next_files=8, predictionStep=8, split="test"):
        super().__init__()
        self.split = split
        self.date = date
        self.root_path = os.path.join(Config.DATA_DIR, self.split)
        self.path_fmt = "%Y%m%d"
        self.fn_pattern = "%Y%m%d_%H%M"
        self.fn_ext = "png"
        self.importer = PNGDataImporter().import_custom_png
        self.importer_kwargs = {}
        self.timestep = timestep
        self.num_prev_files = num_prev_files
        self.num_next_files = num_next_files
        self.predictionStep = predictionStep

    def __len__(self):
        """
        Return size of the data set for DataLoader, but if Dataset gives complete batches
        and not individual items from the dataset, then it should return total number of batches
        """
        return len(self.fileNames)

    def _get_all_sequences(self):
        return self.num_prev_files + self.num_next_files + 1

    def _check_batch(self, frames, batch_idx, frame_type):
        # Check if all frames are not a type of masked array
        # if all([isinstance(f, np.ma.MaskedArray) for f in frames]):
        #     raise ValueError(f"Frame(s) of batch {batch_idx} have masked array in {frame_type} frames")
        pass

    def _load_data(self):
        # Find files for the specified date
        filenames = io.archive.find_by_date(
            self.date,
            self.root_path,
            self.path_fmt,
            self.fn_pattern,
            self.fn_ext,
            timestep=self.timestep,
            num_prev_files=self.num_prev_files,
            num_next_files=self.num_next_files,
        )
        # Read time series data
        rainrate_field, quality, metadata = io.read_timeseries(filenames, self.importer, **self.importer_kwargs)
        # Convert to precipitation rate if necessary
        rainrate_field, metadata = utils.to_rainrate(rainrate_field, metadata)

        # Set the refobs_field as the last frames equal to the predictionStep
        refobs_field = rainrate_field[-self.predictionStep :]

        return rainrate_field, refobs_field, metadata

    def __getitem__(self, idx):
        """Returns one sequence(batch) of frames"""
        input_frames, target_frames, _ = self._load_data()
        input_frames = np.nan_to_num(input_frames, nan=0.0)
        target_frames = np.nan_to_num(target_frames, nan=0.0)
        input_frames[input_frames > 128] = 128
        target_frames[target_frames > 128] = 128
        return input_frames, target_frames
