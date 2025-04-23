import argparse
from collections import OrderedDict

from pathlib import Path
from sprite_core.config import Config
from datetime import datetime
import os
import shutil

import torch
import time
from pytorch_lightning import Trainer
from flwr.client import NumPyClient, start_client

# import model and dataloader
from dgmr.dgmr_es_fl_gradient_checkpoint import DGMR
from train.flower.dataloader.dgmr_fl_dataloader import DGMRDataModule
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

import wandb


def convert_metrics(metrics: dict) -> dict:
    processed_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            processed_metrics[key] = value.item()
        else:
            processed_metrics[key] = value
    return processed_metrics


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger
    else:
        raise Exception("WandbLogger not found in trainer.")


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


class UploadCheckpointsAsArtifact(Callback):
    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False, client_id: int = 0):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only
        self.client_id = client_id

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        artifact_name = f"client_{self.client_id}_experiment-ckpts"
        ckpts = wandb.Artifact(artifact_name, type="checkpoints")

        # The rest remains the same as in the original method
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


class CleanUpLogsCallback(Callback):
    def __init__(self, log_dir):
        self.log_dir = log_dir

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        shutil.rmtree(self.log_dir, ignore_errors=True)
        print(f"Removed log directory: {self.log_dir}")


class FlowerClient(NumPyClient):
    def __init__(self, client_id, log: str = "gradients", log_freq: int = 100):
        self.client_id = client_id  # Unique client ID
        self.model = None  # Model will be initialized during training
        self.datamodule = None  # Data module will be initialized during training
        # Initialize wandb
        if log not in ["gradients", "parameters", "all", None]:
            raise ValueError("log must be one of 'gradients', 'parameters', 'all', or None")
        self.log_option = log  # Renamed to avoid conflict
        self.log_freq = log_freq
        os.makedirs(Config.WANDB_DIR / "dgmr_fl", exist_ok=True)
        wandb.init(project="DGMR_FL_dataSplit", name=f"client_{self.client_id}", dir=Config.WANDB_DIR / "dgmr_fl")
        process_id = os.getpid()  # Get process ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"lightning_logs_{process_id}_{timestamp}"
        self.logger = WandbLogger(save_dir=self.log_dir)
        print(f"Debug: self.log_option = {self.log_option}")

    def get_parameters(self):
        if self.model is None:
            return []
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        if self.model is None:
            # Initialize model
            self.model = DGMR(generation_steps=6)
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        task = config.get("task", "train")
        data_shard = config.get("data_shard", 0)
        total_shards = config.get("total_shards", 1)
        num_clients = config.get("num_client", 1)
        model_version = config.get("model_version", 0)

        # Set model parameters
        if parameters:
            self.set_parameters(parameters)
        else:
            # For first training, randomly initialize the model
            self.model = DGMR(generation_steps=6)
            # self.logger.watch(model=self.model, log=self.log_option, log_freq=self.log_freq)

        # Initialize data module
        batch_size = 16
        self.datamodule = DGMRDataModule(
            client_id=self.client_id,
            num_clients=num_clients,
            batch_size=batch_size,
            data_shard=data_shard,
            total_shards=total_shards,
        )
        self.datamodule.setup()

        modelname = "FL_DGMR-CASA"
        dirpath = "/work/pi_mzink_umass_edu/SPRITE/skillful_nowcasting/output/models"

        g_loss_model_checkpoint = ModelCheckpoint(
            monitor="train_g_loss",
            dirpath=dirpath,
            filename=modelname + "-{epoch}-{train_g_loss}",
            verbose=True,
            save_top_k=1,
        )

        es_model_checkpoint = ModelCheckpoint(
            monitor="val_eval_score",
            mode="max",
            dirpath=dirpath,
            filename=modelname + "-{epoch}-{val_eval_score}",
            save_top_k=1,
            verbose=True,
        )

        checkpoint_callback_300_epochs = ModelCheckpoint(
            every_n_epochs=300,
            dirpath=dirpath,
            filename=modelname + "{epoch}",
            verbose=True,
        )

        # Define callbacks
        callbacks = [
            # Pause saving checkpoints on the client; they will be saved after aggregation on the server
            # UploadCheckpointsAsArtifact(ckpt_dir="checkpoints/", upload_best_only=False),
            EpochTimer(),
            # g_loss_model_checkpoint,
            # es_model_checkpoint,
            # checkpoint_callback_300_epochs,
            CleanUpLogsCallback(self.log_dir),
        ]

        # Configure Trainer
        trainer = Trainer(
            default_root_dir=self.log_dir,
            max_epochs=1,  # Train for only one epoch at a time
            logger=self.logger,
            callbacks=callbacks,
            # enable_checkpointing=False,
            accelerator="gpu",
            devices="auto",
            # Other configurations
        )

        # Perform operations based on the task
        if task == "train":
            # Execute backpropagation to update the model
            train_metrics = self.train_one_epoch(trainer)
            # Return the updated model parameters
            parameters = self.get_parameters()
            metrics = {"task": "update", "data_shard": data_shard}
            metrics.update(convert_metrics(train_metrics))
        else:
            # Unknown task
            parameters = []
            metrics = {"task": task, "data_shard": data_shard}

        return parameters, len(self.datamodule.train_dataloader().dataset), metrics

    def train_one_epoch(self, trainer):
        # Execute one epoch of training, including forward and backward propagation
        trainer.fit(self.model, self.datamodule)
        return trainer.callback_metrics

    # def evaluate(self, parameters, config):
    #     # Set model parameters
    #     self.set_parameters(parameters)
    #
    #     # Execute validation
    #     trainer = Trainer(logger=self.logger, accelerator="gpu", devices="auto")
    #     val_metrics = trainer.validate(self.model, self.datamodule, verbose=True)
    #
    #     # Return validation metrics
    #     return val_metrics["val_g_loss"], len(self.datamodule.train_dataloader().dataset), val_metrics

    def __del__(self):
        # wandb.finish()
        server_domain_path = "FL_Server_GPU.txt"
        if os.path.exists(server_domain_path):
            os.remove(server_domain_path)
            print("Server domain record deleted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--client_id", type=int, required=True, help="Client ID")
    parser.add_argument("--server_address", type=str, required=True, help="Server address")
    args = parser.parse_args()

    # Create a client instance
    client = FlowerClient(client_id=args.client_id)

    try:
        # Start the client
        start_client(
            server_address=args.server_address,
            client=client,
            grpc_max_message_length=1024 * 1024 * 1024,
        )
    except Exception as e:
        print(f"Client {args.client_id} encountered exception: {e}")
        print("Client has stopped.")
