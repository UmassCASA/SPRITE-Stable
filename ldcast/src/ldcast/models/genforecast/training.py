import lightning.pytorch as pl
import torch
import os
from ..diffusion import diffusion

import wandb
from datetime import datetime
from sprite_core.config import Config
from lightning.pytorch.loggers import WandbLogger

import shutil

import time
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only


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


class CleanUpLogsCallback(Callback):
    def __init__(self, log_dir):
        self.log_dir = log_dir

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        shutil.rmtree(self.log_dir, ignore_errors=True)
        print(f"Removed log directory: {self.log_dir}")


def setup_genforecast_training(
    model,
    autoencoder,
    context_encoder,
    config,
):
    # Initialize WandB
    # group_name = f"{config.model_name}_{config.job_id}"
    group_name = "ldcast"
    os.makedirs(Config.WANDB_DIR / "ldcast", exist_ok=True)
    wandb.init(project="ldcast", dir=Config.WANDB_DIR / "ldcast", name=config.model_name, group=group_name)
    wandb_logger = WandbLogger()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"lightning_logs_{config.job_id}_{timestamp}"

    # Define callbacks for saving the best models
    checkpoint_callback_train = pl.callbacks.ModelCheckpoint(
        monitor="train_eval_score",
        mode="max",
        dirpath=config.model_dir,
        filename="best_train_eval_score-{epoch:02d}-{train_eval_score:.2f}",
        save_top_k=1,
    )

    checkpoint_callback_final = pl.callbacks.ModelCheckpoint(
        dirpath=config.model_dir,
        filename="last",
        save_last=True,
    )

    early_stopping = pl.callbacks.EarlyStopping(
        f"val_{config.loss_type}_loss_ema", patience=200, verbose=True, check_finite=False
    )
    checkpoint_callback_val_ema = pl.callbacks.ModelCheckpoint(
        dirpath=config.model_dir,
        filename="{epoch}-{val_loss_ema:.4f}",
        monitor=f"val_{config.loss_type}_loss_ema",
        every_n_epochs=1,
        save_top_k=1,
    )

    callbacks = [
        EpochTimer(),
        CleanUpLogsCallback(log_dir),
        # checkpoint_callback_train,
        early_stopping,
        checkpoint_callback_val_ema,
        checkpoint_callback_final,
        # WatchModel(log='gradients', log_freq=10),
    ]

    ldm = diffusion.LatentDiffusion(
        model=model,
        autoencoder=autoencoder,
        timesteps=config.diffusion_steps,
        context_encoder=context_encoder,
        lr=config.lr,
        loss_type=config.loss_type,
        parameterization=config.parameterization,
    )

    num_gpus = torch.cuda.device_count()

    trainer_args = {
        "logger": wandb_logger,
        "callbacks": callbacks,
        "num_nodes": config.num_nodes,
        "devices": config.gpus,
        "accelerator": "gpu" if (num_gpus > 0) else "cpu",
        "precision": config.precision,
        "strategy": config.strategy,
        "num_sanity_val_steps": 2,
        "fast_dev_run": config.fast_dev_run,
    }
    if config.epochs:
        trainer_args["max_epochs"] = config.epochs
    else:
        trainer_args["max_time"] = {"days": 7}

    trainer = pl.Trainer(**trainer_args)

    return (ldm, trainer)
