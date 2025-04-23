# The Functional run_casa.py

import os
from pathlib import Path
import time
from datetime import datetime

import numpy as np
from netCDF4 import Dataset
import wandb

from dgmr import DGMR
from sprite_core.config import Config

import torch.utils.data.dataset
from torch.utils.data import DataLoader, RandomSampler
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only


NUM_INPUT_FRAMES = 4
NUM_TARGET_FRAMES = 18
TOTAL_FRAMES = NUM_INPUT_FRAMES + NUM_TARGET_FRAMES


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    raise Exception("You are using wandb related callback, but WandbLogger was not found for some reason...")


class WatchModel(Callback):
    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log = log
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq, log_graph=True)


class UploadCheckpointsAsArtifact(Callback):
    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    @rank_zero_only
    def on_keyboard_interrupt(self, trainer, pl_module):
        self.on_train_end(trainer, pl_module)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in Path(self.ckpt_dir).rglob("*.ckpt"):
                ckpts.add_file(str(path))

        experiment.log_artifact(ckpts)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in Path(self.ckpt_dir).rglob("*.ckpt"):
                ckpts.add_file(str(path))

        experiment.log_artifact(ckpts)

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in Path(self.ckpt_dir).rglob("*.ckpt"):
                ckpts.add_file(str(path))

        experiment.log_artifact(ckpts)


class EpochTimer(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):  # Note the unused argument
        end_time = time.time()
        duration = end_time - self.start_time
        print(f"\nTraining: Epoch {trainer.current_epoch} completed in {duration:.2f} seconds\n")

    def on_validation_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        end_time = time.time()
        duration = end_time - self.start_time
        print(f"\nValidation: Epoch {trainer.current_epoch} completed in {duration:.2f} seconds\n")


class NetCDFDataset(torch.utils.data.dataset.Dataset):
    """
    Typically, dataset returns an individual item from the dataset in __getitem__ method.
    Also, it should return the number of items in the dataset in __len__ method.

    Here, dataset returns a batch of frames in __getitem__ method, therefore __len__ method returns the number of
    batches.
    Also, DGMRDataModule should have batch_size=1, since we are returning a batch of frames in __getitem__ method.
    """

    def __init__(self, split):
        super().__init__()
        self.split = split
        self.local_folder_path = os.path.join(Config.DATA_DIR, split)
        self.all_sequences = self._get_all_sequences()

    def __len__(self):
        """
        Return size of the data set for DataLoader, but if Dataset gives complete batches
        and not individual items from the dataset, then it should return total number of batches
        """
        return len(self.all_sequences)

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

    def __getitem__(self, idx):
        """Returns one sequence(batch) of frames"""
        seq_path = self.all_sequences[idx]
        frame_paths = sorted([os.path.join(seq_path, f) for f in os.listdir(seq_path)])

        frames = [self._load_frame(fp) for fp in frame_paths]
        input_frames = np.stack(frames[:NUM_INPUT_FRAMES])
        target_frames = np.stack(frames[NUM_INPUT_FRAMES:TOTAL_FRAMES])

        # Checks for the entire batch
        self._check_batch(input_frames, idx, "input")
        self._check_batch(target_frames, idx, "target")

        return input_frames, target_frames


class DGMRDataModule(LightningDataModule):
    """
    Example of LightningDataModule for NETCDF dataset.
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html

    pin_memory: use pinned memory and enables faster data transfer to CUDA-enabled GPUs.
    num_workers: setting it to 0 means that the main process will do the data loading.
    batch_size: 1 batch is 1 sequence of radar frames which is 22 frames in total.
    """

    def __init__(
        self,
        num_workers=0,
        pin_memory=True,
        batch_size=16,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.num_sequences_per_epoch = 100

    def setup(self, stage=None):
        # Initialize dataset attributes
        self.train_dataset = NetCDFDataset(split="train")
        self.val_dataset = NetCDFDataset(split="validation")
        self.test_dataset = NetCDFDataset(split="test")

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
        )
        return train_loader

    def val_dataloader(self):
        # Use the already initialized dataset
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,  # Data is sequential
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
        )
        return test_loader


def start_training(modelname="DGMR-CASA"):
    os.makedirs(Config.WANDB_DIR / "dgmr", exist_ok=True)
    wandb.init(project="dgmr", dir=Config.WANDB_DIR / "dgmr", name=modelname)

    wandb_logger = WandbLogger(logger="dgmr")
    model_checkpoint = ModelCheckpoint(
        monitor="train/g_loss",
        dirpath="/work/pi_mzink_umass_edu/SPRITE/skillful_nowcasting/output/models",
        filename=modelname,
        verbose=True,
    )

    epoch_timer = EpochTimer()
    datamodule = DGMRDataModule()
    model = DGMR(generation_steps=2)

    # check how many gpus are available
    print("Number of GPUs:", torch.cuda.device_count())

    trainer = Trainer(
        max_time={"days": 7},  # Train for a week
        # max_epochs=1,
        logger=wandb_logger,
        callbacks=[model_checkpoint, epoch_timer],
        num_nodes=1,  # num_nodes = machines
        devices=-1,  # GPUs on the node
        accelerator="gpu",
        precision=32,
    )

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    ### Specify the model name
    modelname = f"DGMR-CASA-Test-{timestamp}"
    start_training(modelname)
