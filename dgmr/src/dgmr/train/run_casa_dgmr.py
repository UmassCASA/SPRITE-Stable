import os
import argparse
from pathlib import Path
from datetime import datetime
import wandb
import torch

import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.callbacks import DeviceStatsMonitor

from dgmr.models.dgmr import DGMR
from sprite_core.config import Config
from casa_datatools.CASADataModule import CASADataModule
from dgmr.train.checkpoints import (
    get_train_g_loss_checkpoint,
    get_val_g_loss_checkpoint,
    get_checkpoint_callback_nth_epochs,
    get_last_epoch_checkpoint,
)
from dgmr.train.callbacks import UploadCheckpointsAsArtifact, WatchModel, get_wandb_logger


def aggregate_starting_info(group_name, trainer=None, **kwargs):
    """
    Aggregate and print starting info for training.
    This includes system information, arguments, and configurations.
    """
    if trainer is None or trainer.global_rank == 0:
        starting_info = {
            "GPU Count": torch.cuda.device_count(),
            "CPU Count": os.cpu_count(),
            "Current DateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "GPU Name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU available",
            "WandB Group Name": group_name,
            "PyTorch Version": torch.__version__,
            "Lightning Version": L.__version__,
            "Arguments": {k: v for k, v in kwargs.items() if v is not None},
        }

        # Print the starting information in a structured format
        print("\n=== Starting Training Configuration ===")
        for key, value in starting_info.items():
            if key == "Arguments":
                print(f"{key}:")
                for arg_key, arg_value in value.items():
                    print(f"  - {arg_key}: {arg_value}")
            else:
                print(f"{key}: {value}")
        print("=======================================\n")

        # Log configuration details to WandB
        wandb.config.update(starting_info)


def start_training(params):
    # Initialize WandB
    group_name = params.model_name + ("_" + params.job_id if params.job_id else "")
    os.makedirs(Config.WANDB_DIR / "dgmr", exist_ok=True)

    wandb.init(
        project=params.project,
        dir=Config.WANDB_DIR / "dgmr",
        name=params.model_name,
        group=group_name,
    )
    wandb_logger = WandbLogger()

    # Checkpoint callbacks
    train_g_loss_model_checkpoint = get_train_g_loss_checkpoint(params.output_dir, params.model_name)
    val_g_loss_model_checkpoint = get_val_g_loss_checkpoint(params.output_dir, params.model_name)
    save_every_n_epochs = get_checkpoint_callback_nth_epochs(
        params.output_dir, params.model_name, every_n_epochs=20, save_top_k=0
    )
    save_last_epoch = get_last_epoch_checkpoint(params.output_dir, params.model_name)

    datamodule = CASADataModule(
        batch_size=params.batch_size,
        val_batch_size=4,
        num_workers=1,
        num_input_frames=4,
        num_target_frames=18,
        ensure_2d=False,
        data_dir=Config.DATA_DIR,
    )

    print(f"Generator Learning Rate: {params.gen_lr}")
    print(f"Discriminator Learning Rate: {params.disc_lr}")
    print(f"Precipitation Weight Cap: {params.precip_weight_cap}")

    model = DGMR(
        generation_steps=params.gen_steps,
        metrics_path=os.path.join(params.output_dir, "metrics"),
        job_id=params.job_id,
        gen_lr=params.gen_lr,
        disc_lr=params.disc_lr,
        precip_weight_cap=params.precip_weight_cap,
        regularizer_type=params.regularizer_type,
        regularizer_lambda=params.regularizer_lambda,
    )

    # Decide the training duration (epochs or time)
    trainer_args = {
        "logger": wandb_logger,
        "callbacks": [
            save_every_n_epochs,
            save_last_epoch,
            train_g_loss_model_checkpoint,
            val_g_loss_model_checkpoint,
            DeviceStatsMonitor(),
        ],
        "num_nodes": params.num_nodes,
        "devices": params.gpus,
        "accelerator": "gpu",
        "precision": params.precision,
        "strategy": params.strategy,
        "num_sanity_val_steps": 2,
        "fast_dev_run": params.fast_dev_run,
    }

    if params.epochs:
        trainer_args["max_epochs"] = params.epochs
    else:
        trainer_args["max_time"] = {"days": 7}

    trainer = Trainer(**trainer_args)

    aggregate_starting_info(group_name=group_name, trainer=trainer, **vars(params))

    trainer.fit(model, datamodule, ckpt_path=params.checkpoint_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--project", type=str, default="dgmr", help="Project name")
    parser.add_argument("--precision", type=str, default="32", help="Precision for training")
    parser.add_argument("--gen_steps", type=int, default=2, help="Number of generation steps")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--job_id", type=str, help="Job ID")
    parser.add_argument("--partition", type=str, help="Partition used for SLURM job")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Checkpoint path for resuming training")
    parser.add_argument("--strategy", type=str, default=None, help="Training strategy")
    parser.add_argument("--precip_weight_cap", type=float, default=24.0, help="Cap for precipitation weights")
    parser.add_argument("--epochs", type=int, help="Number of epochs to train")
    parser.add_argument("--gen_lr", type=float, default=5e-5, help="Generator learning rate")
    parser.add_argument("--disc_lr", type=float, default=2e-4, help="Discriminator learning rate")
    parser.add_argument("--fast_dev_run", action="store_true", help="Fast dev run for testing")
    parser.add_argument(
        "--regularizer_type",
        type=str,
        default="grid_cell",
        choices=["grid_cell", "rec"],
        help="Type of regularization loss to use",
    )
    parser.add_argument(
        "--regularizer_lambda", type=float, default=20.0, help="Lambda coefficient for the regularization loss"
    )
    return parser.parse_args()


if __name__ == "__main__":
    params = parse_args()
    start_training(params)
