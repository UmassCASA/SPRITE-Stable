import os
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset
from pysteps import io, utils

from sprite_core.config import Config
from model_compare.data_readers.pysteps_customized_importers.NetCDFImporter import NetCDFImporter


class NetCDFDataset(Dataset):
    """
    A Dataset that can work in two modes:
    1) "seq" (date=None): scans seq-* folders, each containing multiple .nc,
       and returns each folder's timeseries as one item (CASA-like).
    2) "date" (date not None): scans ALL seq-*/.nc, merges them into a global timeline,
       finds the nearest file to the given date, and returns that date's neighborhood frames
       as a single item (PNGDataset-like).
    """

    def __init__(
        self,
        split="test",
        num_prev_files=4,
        num_next_files=18,
        include_datetimes=False,
        to_rainrate=True,
        max_val=128.0,
        date=None,  # <-- new param for "PNGDataset style" usage
    ):
        """
        Args:
            split: "train", "test", "validation", etc.
            num_prev_files: how many frames before the 'center' frame
            num_next_files: how many frames for the target
            include_datetimes: whether to return the timestamps
            to_rainrate: if True, convert reflectivity to rain rate
            max_val: clip data above this
            date: if not None, switch to 'date' mode, otherwise 'seq' mode
        """
        super().__init__()
        self.split = split
        self.root_path = os.path.join(Config.DATA_DIR, self.split)

        # for "seq" mode
        self.num_prev_files = num_prev_files
        self.num_next_files = num_next_files
        self.total_frames = num_prev_files + num_next_files
        self.include_datetimes = include_datetimes
        self.to_rainrate = to_rainrate
        self.max_val = max_val

        # new param
        self.date = date

        # Register the importer function (assuming we call it directly)
        self.import_custom_netcdf = NetCDFImporter().import_custom_netcdf
        self.importer_kwargs = {}

        # Decide which mode
        if self.date is None:
            self.mode = "seq"
            # gather all seq-* directories
            self.seq_dirs = sorted(
                [
                    os.path.join(self.root_path, d)
                    for d in os.listdir(self.root_path)
                    if d.startswith("seq-") and os.path.isdir(os.path.join(self.root_path, d))
                ]
            )
        else:
            self.mode = "date"
            # We'll scan ALL .nc files in all seq-* folders, building a global sorted list
            self.all_files = self._scan_all_nc_files()
            # find the center index near `self.date`
            self.center_idx = self._find_center_index(self.all_files, self.date)
            # If we only want to return 1 item for the entire date-based chunk, set length=1
            self.len = 1

    def __len__(self):
        if self.mode == "seq":
            return len(self.seq_dirs)
        else:
            return self.len  # typically 1

    # ----------------------------------------------------------------
    #  Mode "SEQ": scanning one folder => read_time_series => slice
    # ----------------------------------------------------------------

    def _parse_datetime_from_filename(self, fname):
        """
        e.g. '20230514_093337.nc' -> datetime(2023,05,14,09,33,37)
        """
        base = os.path.splitext(fname)[0]  # => '20230514_093337'
        date_str, time_str = base.split("_")
        return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")

    def _get_sorted_nc_files_in_seq(self, seq_path):
        """
        Return a sorted list of .nc file paths in the given seq folder
        """
        entries = []
        for fname in os.listdir(seq_path):
            if fname.endswith(".nc"):
                dt = self._parse_datetime_from_filename(fname)
                full_path = os.path.join(seq_path, fname)
                entries.append((dt, full_path))
        entries.sort(key=lambda x: x[0])  # sort by dt
        return [item[1] for item in entries]

    # ----------------------------------------------------------------
    #  Mode "DATE": scanning all seq-* => build global timeline
    # ----------------------------------------------------------------

    def _scan_all_nc_files(self):
        """Scan all seq-* folders, gather (dt, path) in a big list, sorted by dt."""
        all_entries = []
        for d in os.listdir(self.root_path):
            seq_path = os.path.join(self.root_path, d)
            if d.startswith("seq-") and os.path.isdir(seq_path):
                for fname in os.listdir(seq_path):
                    if fname.endswith(".nc"):
                        full_path = os.path.join(seq_path, fname)
                        dt = self._parse_datetime_from_filename(fname)
                        all_entries.append((dt, full_path))
        all_entries.sort(key=lambda x: x[0])
        return all_entries

    def _find_center_index(self, sorted_files, target_date):
        """
        Find the index i in sorted_files s.t. sorted_files[i] is closest to target_date.
        sorted_files is a list of (dt, path).
        """
        best_i = 0
        best_diff = abs(sorted_files[0][0] - target_date)
        for i, (dt, _) in enumerate(sorted_files):
            diff = abs(dt - target_date)
            if diff < best_diff:
                best_diff = diff
                best_i = i
        return best_i

    # ----------------------------------------------------------------
    #  read, convert, slice, etc.
    # ----------------------------------------------------------------

    def _nan_and_clip(self, arr):
        arr = np.nan_to_num(arr, nan=0.0)
        arr[arr > self.max_val] = self.max_val
        return arr

    def __getitem__(self, idx):
        if self.mode == "seq":
            # 1) pick the seq-* folder
            seq_path = self.seq_dirs[idx]
            # 2) get sorted .nc file list
            sorted_entries = self._get_sorted_nc_files_in_seq(seq_path)
            # 3) read timeseries with our importer
            sorted_datetimes = [
                datetime.strptime(e.split("/")[-1].split(".")[0].strip(), "%Y%m%d_%H%M%S") for e in sorted_entries
            ]
            sorted_filepaths = sorted_entries
            filepaths_and_times = (sorted_filepaths, sorted_datetimes)
            data_array, quality, metadata = io.read_timeseries(
                filepaths_and_times, self.import_custom_netcdf, **self.importer_kwargs
            )
            # 4) optionally convert to rainrate
            # if self.to_rainrate:
            #     data_array, metadata = utils.to_rainrate(data_array, metadata)

            # 5) check length
            if data_array.shape[0] < self.total_frames + 1:
                raise ValueError(
                    f"Not enough frames in {seq_path}: have {data_array.shape[0]}, need {self.total_frames + 1}"
                )

            # 6) slice input (first num_prev_files) & target (next num_next_files)
            #    This matches your current approach, e.g. input=[:4], target=[4:22]
            input_frames = data_array[: self.total_frames + 1]
            target_frames = data_array[(self.num_prev_files + 1) : (self.total_frames + 1)]

            # 7) handle missing or large
            input_frames = self._nan_and_clip(input_frames)
            target_frames = self._nan_and_clip(target_frames)

            input_frames = np.squeeze(input_frames, axis=1)
            target_frames = np.squeeze(target_frames, axis=1)

            if self.include_datetimes:
                chosen_files = sorted_filepaths[: self.total_frames + 1]
                frame_datetimes = [self._parse_datetime_from_filename(os.path.basename(p)) for p in chosen_files]
                return input_frames, target_frames, metadata, frame_datetimes
            else:
                return input_frames, target_frames, metadata

        else:
            # == "date" mode ==
            # We only have 1 item => idx=0
            # center on self.date
            center_i = self.center_idx

            # We want total_needed = num_prev_files + num_next_files + 1
            # Because PNGDataset style => input[: (num_prev_files+1)], target[-num_next_files:]
            total_needed = self.num_prev_files + self.num_next_files + 1

            # pick from [center_i - num_prev_files : center_i + num_next_files + 1)
            start_i = center_i - self.num_prev_files
            end_i = center_i + self.num_next_files + 1  # exclusive
            if start_i < 0:
                start_i = 0
            if end_i > len(self.all_files):
                end_i = len(self.all_files)

            chosen_entries = self.all_files[start_i:end_i]  # list of (dt, path)
            chosen_paths = [item[1] for item in chosen_entries]
            chosen_datetimes = [item[0] for item in chosen_entries]

            filepaths_and_times = (chosen_paths, chosen_datetimes)

            # read them
            data_array, quality, metadata = io.read_timeseries(
                filepaths_and_times, self.import_custom_netcdf, **self.importer_kwargs
            )
            if self.to_rainrate:
                data_array, metadata = utils.to_rainrate(data_array, metadata)

            T = data_array.shape[0]
            if T < total_needed:
                raise ValueError(f"Not enough frames around date {self.date}, have {T}, need {total_needed}")

            # Now slice per PNGDataset logic:
            # input_frames = data_array[: (num_prev_files+1)]
            # target_frames = data_array[-num_next_files:]
            input_frames = data_array
            target_frames = data_array[-self.num_next_files :]

            input_frames = self._nan_and_clip(input_frames)
            target_frames = self._nan_and_clip(target_frames)

            input_frames = np.squeeze(input_frames, axis=1)
            target_frames = np.squeeze(target_frames, axis=1)

            if self.include_datetimes:
                # also gather the matching datetimes
                # we do have "chosen_entries" in ascending order, but let's ensure we sort by dt
                combined = list(zip(chosen_entries, data_array))
                # sort by dt if needed
                combined.sort(key=lambda x: x[0][0])
                # re-extract dt and frames
                chosen_dts = [x[0][0] for x in combined]
                # now the input = first (num_prev_files+1), target= last num_next_files
                # input_dts  = chosen_dts[: (self.num_prev_files + 1)]
                # target_dts = chosen_dts[-self.num_next_files:]

                # you could unify them into one list, or keep them separate
                frame_datetimes = chosen_dts
                return input_frames, target_frames, metadata, frame_datetimes
            else:
                return input_frames, target_frames, metadata
