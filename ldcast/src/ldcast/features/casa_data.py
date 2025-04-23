import os
import numpy as np
import torch

from casa_datatools.CASADataModule import CASADataModule
from casa_datatools.CASADataset import CASADataset


class CasaDatasetWithTimeList(CASADataset):
    def __init__(self, *args, **kwargs):
        self.dt = kwargs.pop("dt")
        self.batch_size = kwargs.pop("batch_size")
        self.for_ae = kwargs.pop("for_ae")
        super().__init__(*args, **kwargs)

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

        if self.for_ae:
            all_frames_out = frames[: self.total_frames]
            all_frames_out = np.stack(all_frames_out, axis=0)
            self._check_batch(all_frames_out, idx, "all_frames_out")

            input_frames = all_frames_out
            target_frames = all_frames_out

            num_t_relative = self.total_frames
        else:
            input_frames, target_frames = self._get_input_and_target_frames(frames=frames, idx=idx)

            num_t_relative = self.num_input_frames

        t_relative = np.arange(num_t_relative, dtype=np.float32) * self.dt - (self.num_input_frames - 1) * self.dt
        # t_relative = np.broadcast_to(t_relative, (self.batch_size, num_t_relative))

        input_frames = torch.from_numpy(input_frames).float().permute(1, 0, 2, 3)
        target_frames = torch.from_numpy(target_frames).float().permute(1, 0, 2, 3)
        # input_frames = torch.from_numpy(input_frames).float().squeeze()
        # target_frames = torch.from_numpy(target_frames).float().squeeze()

        if self.include_datetimes:
            return [(input_frames, t_relative)], target_frames, frame_datetimes
        else:
            return [(input_frames, t_relative)], target_frames


class CasaDataModuleWithTimeList(CASADataModule):
    def __init__(self, *args, **kwargs):
        self.for_ae = kwargs.pop("for_ae")
        self.dt = kwargs.pop("dt")
        super().__init__(*args, **kwargs)

    def setup(self, stage=None):
        # Initialize dataset attributes with dynamic frame counts
        self.train_dataset = self.dataset(
            split="train",
            num_input_frames=self.num_input_frames,
            num_target_frames=self.num_target_frames,
            include_datetimes=False,
            ensure_2d=self.ensure_2d,
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            for_ae=self.for_ae,
            dt=self.dt,
        )
        self.val_dataset = self.dataset(
            split="validation",
            num_input_frames=self.num_input_frames,
            num_target_frames=self.num_target_frames,
            include_datetimes=False,
            ensure_2d=self.ensure_2d,
            data_dir=self.data_dir,
            batch_size=self.val_batch_size,
            for_ae=self.for_ae,
            dt=self.dt,
        )
        self.test_dataset = self.dataset(
            split="test",
            num_input_frames=self.num_input_frames,
            num_target_frames=self.num_target_frames,
            include_datetimes=False,
            ensure_2d=self.ensure_2d,
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            for_ae=self.for_ae,
            dt=self.dt,
        )
