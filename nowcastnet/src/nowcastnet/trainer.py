# Standard library imports
import os
import argparse
import shutil
import time
from datetime import datetime
from pathlib import Path

# Third party imports
import torch
import wandb
import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only

# Local imports
from casa_datatools.CASADataModule import CASADataModule
from nowcastnet.models.nowcastnet_pytorch_lightning import Net
from nowcastnet.layers.entities.net_config import Configs
from sprite_core.config import Config

# modelname = "NowcastNet-CASA"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


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


class CleanUpLogsCallback(Callback):
    def __init__(self, log_dir):
        self.log_dir = log_dir

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        shutil.rmtree(self.log_dir, ignore_errors=True)
        print(f"Removed log directory: {self.log_dir}")


def start_training(params):
    # Setup logging directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"lightning_logs_{params.job_id}_{timestamp}"

    # Initialize WandB logging
    group_name = params.model_name + (f"_{params.job_id}" if params.job_id else "")

    os.makedirs(Config.WANDB_DIR / "nowcastnet", exist_ok=True)
    wandb.init(project="NowcastNet", dir=Config.WANDB_DIR / "nowcastnet", name=params.model_name, group=group_name)
    wandb_logger = WandbLogger()

    # Define callbacks for saving the best models
    checkpoint_callback_train = ModelCheckpoint(
        monitor="train_eval_score",
        mode="max",
        dirpath=params.output_dir,
        filename="best_train_eval_score-{epoch:02d}-{train_eval_score:.2f}",
        save_top_k=1,
    )
    checkpoint_callback_val_d_loss = ModelCheckpoint(
        monitor="val_d_loss",
        mode="min",
        dirpath=params.output_dir,
        filename="best_val_d_loss-{epoch:02d}-{val_d_loss:.2f}",
        save_top_k=1,
    )
    checkpoint_callback_val_g_loss = ModelCheckpoint(
        monitor="val_g_loss",
        mode="min",
        dirpath=params.output_dir,
        filename="best_val_g_loss-{epoch:02d}-{val_g_loss:.2f}",
        save_top_k=1,
    )

    checkpoint_last_epoch = ModelCheckpoint(
        save_top_k=1,
        dirpath=params.output_dir,
        filename=params.model_name + "-last-{epoch:03d}",
        verbose=True,
    )

    callbacks = [
        EpochTimer(),
        CleanUpLogsCallback(log_dir),
        # checkpoint_callback_train,
        checkpoint_callback_val_d_loss,
        checkpoint_callback_val_g_loss,
        checkpoint_last_epoch,
        # WatchModel(log='gradients', log_freq=10),
    ]

    trainer_args = {
        "logger": wandb_logger,
        "callbacks": callbacks,
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

    net_config = Configs()
    model = Net(net_config)

    datamodule = CASADataModule(
        batch_size=params.batch_size,
        val_batch_size=4,
        num_workers=1,
        num_input_frames=4,
        num_target_frames=18,
        ensure_2d=False,
        data_dir=Config.DATA_DIR,
        persistent_workers=False,
    )
    datamodule.setup()

    aggregate_starting_info(
        group_name=group_name,
        trainer=trainer,
        **vars(params),  # Pass parsed arguments as a dictionary
    )

    trainer.fit(model, datamodule)

    # Save final model after all epochs are complete
    final_model_path = os.path.join(params.output_dir, f"{params.model_name}_final_{timestamp}.ckpt")
    trainer.save_checkpoint(final_model_path)
    print(f"Final model saved at {final_model_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--precision", type=str, default="32", help="Precision for training")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--job_id", type=str, help="Job ID")
    parser.add_argument("--partition", type=str, help="Partition used for SLURM job")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Checkpoint path for resuming training")
    parser.add_argument("--strategy", type=str, default=None, help="Training strategy")
    parser.add_argument("--epochs", type=int, help="Number of epochs to train")
    parser.add_argument("--fast_dev_run", action="store_true", help="Fast dev run for testing")
    return parser.parse_args()


if __name__ == "__main__":
    params = parse_args()
    start_training(params)
