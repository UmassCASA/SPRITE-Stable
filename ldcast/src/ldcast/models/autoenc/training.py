import lightning.pytorch as pl
import torch
import os
from . import autoenc

from lightning.pytorch.loggers import WandbLogger
import wandb

from sprite_core.config import Config
from datetime import datetime
from lightning.pytorch.callbacks import Callback
import shutil
from lightning.pytorch.utilities import rank_zero_only
import time


class CleanUpLogsCallback(Callback):
    def __init__(self, log_dir):
        self.log_dir = log_dir

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        shutil.rmtree(self.log_dir, ignore_errors=True)
        print(f"Removed log directory: {self.log_dir}")


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


def setup_autoenc_training(encoder, decoder, model_dir):
    autoencoder = autoenc.AutoencoderKL(encoder, decoder)

    num_gpus = torch.cuda.device_count()
    accelerator = "gpu" if (num_gpus > 0) else "cpu"
    devices = torch.cuda.device_count() if (accelerator == "gpu") else 1

    early_stopping = pl.callbacks.EarlyStopping("val_rec_loss", patience=100, verbose=True)
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=model_dir, filename="{epoch}-{val_rec_loss:.4f}", monitor="val_rec_loss", every_n_epochs=1, save_top_k=3
    )

    # Initialize WandB
    group_name = "ldcast"
    os.makedirs(Config.WANDB_DIR / "ldcast", exist_ok=True)
    wandb.init(project="ldcast", dir=Config.WANDB_DIR / "ldcast", name="ldcast_autoenc", group=group_name)
    wandb_logger = WandbLogger()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"lightning_logs_{timestamp}"

    callbacks = [
        EpochTimer(),
        CleanUpLogsCallback(log_dir),
        early_stopping,
        checkpoint,
    ]

    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator=accelerator,
        devices=devices,
        max_epochs=1000,
        strategy="ddp" if (num_gpus > 1) else "auto",
        callbacks=callbacks,
    )

    return (autoencoder, trainer)
