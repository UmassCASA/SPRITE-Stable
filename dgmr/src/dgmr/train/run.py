import os
from pathlib import Path
from datetime import datetime
import numpy as np
from numpy.random import default_rng

import wandb
from datasets import load_dataset

from sprite_core.config import Config
from dgmr import DGMR

import torch.utils.data.dataset
from torch.utils.data import DataLoader
from pytorch_lightning import Callback, Trainer, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only


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


NUM_INPUT_FRAMES = 4
NUM_TARGET_FRAMES = 18


def extract_input_and_target_frames(radar_frames):
    """Extract input and target frames from a dataset row's radar_frames."""
    # We align our targets to the end of the window, and inputs precede targets.
    input_frames = radar_frames[-NUM_TARGET_FRAMES - NUM_INPUT_FRAMES : -NUM_TARGET_FRAMES]
    target_frames = radar_frames[-NUM_TARGET_FRAMES:]
    return input_frames, target_frames


class TFDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, split):
        super().__init__()
        self.reader = load_dataset(
            "openclimatefix/nimrod-uk-1km",
            "sample",
            split=split,
            streaming=True,
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
    """

    def __init__(
        self,
        num_workers: int = 1,
        pin_memory: bool = True,
        batch_size: int = 16,
    ):
        """
        fake_data: random data is created and used instead. This is useful for testing
        pin_memory: use pinned memory and enables faster data transfer to CUDA-enabled GPUs.
        num_workers: setting it to 0 means that the main process will do the data loading.
        """
        super().__init__()

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.batch_size = batch_size

        self.dataloader_config = {
            "pin_memory": self.pin_memory,
            "num_workers": self.num_workers,
            "prefetch_factor": 4,  # each worker will prefetch 8 batches ahead of time.
            "persistent_workers": True,
            # Disable automatic batching because dataset
            # returns complete batches.
            "batch_size": None,
        }

    def train_dataloader(self):
        dataloader = DataLoader(TFDataset(split="train"), batch_size=self.batch_size, num_workers=0)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(TFDataset(split="validation"), batch_size=self.batch_size, num_workers=0)
        return dataloader


def start_training(modelname="DGMR-Nimrod"):
    os.makedirs(Config.WANDB_DIR / "dgmr", exist_ok=True)
    wandb.init(project="dgmr", dir=Config.WANDB_DIR / "dgmr", name=modelname)

    wandb_logger = WandbLogger(logger="dgmr")
    model_checkpoint = ModelCheckpoint(
        monitor="train_g_loss",
        dirpath="/work/pi_mzink_umass_edu/SPRITE/skillful_nowcasting/output/models",
        filename=modelname,
    )

    trainer = Trainer(
        # max_time={"days": 7},  # Train for a week
        max_time={"days": 1},
        logger=wandb_logger,
        callbacks=[model_checkpoint],
        num_nodes=1,
        devices=1,
        accelerator="gpu",
        precision=32,
    )

    model = DGMR(generation_steps=2)
    datamodule = DGMRDataModule(batch_size=8)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    ### Specify the model name
    modelname = f"DGMR-Nirmod-32_2_16-{timestamp}"
    start_training(modelname)
