from .CASADataset import CASADataset
from torch.utils.data import DataLoader, RandomSampler
from lightning.pytorch import LightningDataModule


class CASADataModule(LightningDataModule):
    def __init__(
        self,
        num_workers=0,
        pin_memory=True,
        batch_size=16,
        val_batch_size=4,
        num_input_frames=4,
        num_target_frames=18,
        ensure_2d=False,
        data_dir=None,
        persistent_workers=True,
        dataset=CASADataset,
    ):
        """
        Args:
            num_workers (int): Number of workers for data loading.
            pin_memory (bool): Whether to pin memory.
            batch_size (int): Batch size for training.
            val_batch_size (int): Batch size for validation.
            num_input_frames (int): Number of input frames.
            num_target_frames (int): Number of target frames.
        """
        super().__init__()
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_input_frames = num_input_frames
        self.num_target_frames = num_target_frames
        self.num_sequences_per_epoch = 100
        self.ensure_2d = ensure_2d
        self.data_dir = data_dir
        self.persistent_workers = persistent_workers
        self.dataset = dataset

    def setup(self, stage=None):
        # Initialize dataset attributes with dynamic frame counts
        self.train_dataset = self.dataset(
            split="train",
            num_input_frames=self.num_input_frames,
            num_target_frames=self.num_target_frames,
            include_datetimes=False,
            ensure_2d=self.ensure_2d,
            data_dir=self.data_dir,
        )
        self.val_dataset = self.dataset(
            split="validation",
            num_input_frames=self.num_input_frames,
            num_target_frames=self.num_target_frames,
            include_datetimes=False,
            ensure_2d=self.ensure_2d,
            data_dir=self.data_dir,
        )
        self.test_dataset = self.dataset(
            split="test",
            num_input_frames=self.num_input_frames,
            num_target_frames=self.num_target_frames,
            include_datetimes=False,
            ensure_2d=self.ensure_2d,
            data_dir=self.data_dir,
        )

    def train_dataloader(self):
        # Use RandomSampler to select only a limited number of sequences per epoch
        sampler = RandomSampler(self.train_dataset, replacement=False, num_samples=self.num_sequences_per_epoch)
        # Use the already initialized dataset
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,  # Recommended by warning
        )
        return train_loader

    def val_dataloader(self):
        # Use the already initialized dataset
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,  # Data is sequential
            persistent_workers=self.persistent_workers,  # Recommended by warning
        )
        return val_loader

    def test_dataloader(self):
        # Use the already initialized dataset
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,  # Data is sequential
            persistent_workers=self.persistent_workers,  # Recommended by warning
        )
        return test_loader
