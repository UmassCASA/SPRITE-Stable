import os
import argparse
from collections import OrderedDict

from pathlib import Path
from sprite_core.config import Config

import torch
import time
from pytorch_lightning import Trainer
from flwr.client import NumPyClient, start_client

# import model and dataloader
from dgmr.dgmr_es import DGMR
from train.flower.dataloader.dgmr_fl_dataloader import DGMRDataModule
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

import wandb


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


class WatchModel(Callback):
    def __init__(self, log: str = "gradients", log_freq: int = 100):
        if log not in ["gradients", "parameters", "all", None]:
            raise ValueError("log must be one of 'gradients', 'parameters', 'all', or None")
        self.log_option = log  # Renamed to avoid conflict
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        print(f"Debug: self.log_option = {self.log_option}")
        logger.watch(model=trainer.model, log=self.log_option, log_freq=self.log_freq)


class FlowerClient(NumPyClient):
    def __init__(self, model, datamodule, max_epochs, client_id):
        self.model = model
        self.datamodule = datamodule
        self.max_epochs = max_epochs
        self.client_id = client_id  # 客户端的唯一 ID
        self.logger = None  # 将在训练过程中初始化

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # 初始化 wandb
        os.makedirs(Config.WANDB_DIR / "dgmr_fl", exist_ok=True)
        wandb.init(project="dgmr_fl", name=f"client_{self.client_id}", dir=Config.WANDB_DIR / "dgmr_fl")
        self.logger = WandbLogger()

        # TODO: Put modelname and dirpath into parameter/config
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
            save_top_k=3,
            verbose=True,
        )

        checkpoint_callback_300_epochs = ModelCheckpoint(
            every_n_epochs=300,
            dirpath=dirpath,
            filename=modelname + "{epoch}",
            verbose=True,
        )

        # 定义回调
        callbacks = [
            WatchModel(log="gradients"),
            # Pause saving checkpoints in client, will save it after aggregation in the server
            # UploadCheckpointsAsArtifact(ckpt_dir="checkpoints/", upload_best_only=False),
            EpochTimer(),
            g_loss_model_checkpoint,
            es_model_checkpoint,
            # checkpoint_callback_300_epochs,
        ]

        # 配置 Trainer
        trainer = Trainer(
            max_epochs=self.max_epochs,
            logger=self.logger,
            callbacks=callbacks,
            enable_checkpointing=True,
            accelerator="gpu",
            devices=1,
            # 其他配置
        )

        # 执行训练
        trainer.fit(self.model, self.datamodule)

        # 结束 wandb 运行
        wandb.finish()

        return self.get_parameters(), len(self.datamodule.train_dataloader().dataset), {}

    # def evaluate(self, parameters, config):
    #     self.set_parameters(parameters)
    #
    #     # 初始化 wandb
    #     os.makedirs(Config.WANDB_DIR / "dgmr_fl", exist_ok=True)
    #     wandb.init(project="dgmr_fl", name=f"client_{self.client_id}_eval", dir=Config.WANDB_DIR / "dgmr_fl")
    #     self.logger = WandbLogger()
    #
    #     # 配置 Trainer
    #     trainer = Trainer(
    #         logger=self.logger,
    #         accelerator="gpu",
    #         devices=1,
    #         # 其他配置
    #     )
    #
    #     # 执行评估
    #     results = trainer.test(self.model, self.datamodule)
    #     loss = results[0]["test_loss"]
    #
    #     # 结束 wandb 运行
    #     wandb.finish()
    #
    #     return float(loss), len(self.datamodule.test_dataloader().dataset), {}


# implementing client_fn
def client_fn(client_id: int):
    batch_size = 16
    generation_steps = 2
    max_epochs = 100  # epoch number for each round of local training TODO:Adjust
    num_clients = 2  # number of clients, keep same with server

    datamodule = DGMRDataModule(client_id=client_id, num_clients=num_clients, batch_size=batch_size)
    datamodule.setup()
    model = DGMR(generation_steps=generation_steps)

    client = FlowerClient(model, datamodule, max_epochs, client_id)
    return client


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--client_id", type=int, required=True, help="Client ID")
    parser.add_argument("--server_address", type=str, required=True, help="Server address")
    args = parser.parse_args()

    # start client
    start_client(
        server_address=args.server_address,
        client=client_fn(args.client_id),
        grpc_max_message_length=1024 * 1024 * 1024,
    )
