import argparse
import os
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch import loggers
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.utilities import rank_zero_only

from casa_datatools.CASADataModule import CASADataModule
from sprite_core.config import Config
from smaat_unet.models import unet_casa_regression_lightning as unet_regr
from smaat_unet.root import ROOT_DIR


class CustomEarlyStopping(EarlyStopping):
    def on_train_end(self, trainer, pl_module):
        if self.stopped_epoch != 0:
            trainer.logger.log_metrics({"early_stopping_reached": 1}, step=self.stopped_epoch)
            print(
                f"Early stopping reached at epoch {self.stopped_epoch}. "
                f"Best monitored metric value: {self.best_score:.6f}"
            )
        super().on_train_end(trainer, pl_module)


@rank_zero_only
def print_training_start(model_name):
    print(f"\n\n######################### Start training model: {model_name} #########################\n")


def train_regression(hparams, find_batch_size_automatically: bool = False):
    if hparams.model == "UNetDS_Attention":
        net = unet_regr.UNetDS_Attention(hparams=hparams)
    elif hparams.model == "UNet_Attention":
        net = unet_regr.UNet_Attention(hparams=hparams)
    elif hparams.model == "UNet":
        net = unet_regr.UNet(hparams=hparams)
    elif hparams.model == "UNetDS":
        net = unet_regr.UNetDS(hparams=hparams)
    else:
        raise NotImplementedError(f"Model '{hparams.model}' not implemented")

    default_save_path = ROOT_DIR / "lightning" / "precip_regression"

    # Use the provided output path or fall back to default
    save_path = (
        Path(hparams.output_path) if hasattr(hparams, "output_path") and hparams.output_path else default_save_path
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=default_save_path / net.__class__.__name__,
        filename=net.__class__.__name__ + "_rain_threshold_50_{epoch}-{val_loss:.6f}",
        save_top_k=1,
        verbose=False,
        monitor="val_loss",
        mode="min",
    )

    last_checkpoint_callback = ModelCheckpoint(
        dirpath=default_save_path / net.__class__.__name__,
        filename=net.__class__.__name__ + "_rain_threshold_50_{epoch}-{val_loss:.6f}_last",
        save_top_k=1,
        verbose=False,
    )

    earlystopping_callback = CustomEarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=hparams.es_patience,
    )

    lr_monitor = LearningRateMonitor()
    tb_logger = loggers.TensorBoardLogger(save_dir=save_path, name=net.__class__.__name__)

    datamodule = CASADataModule(
        batch_size=hparams.batch_size,
        val_batch_size=hparams.batch_size,
        num_workers=1,
        num_input_frames=hparams.num_input_images,
        num_target_frames=hparams.num_output_images,
        ensure_2d=True,
        data_dir=Config.DATA_DIR,
    )

    trainer = pl.Trainer(
        accelerator=hparams.accelerator,
        num_nodes=hparams.num_nodes,
        devices=hparams.gpus,
        fast_dev_run=hparams.fast_dev_run,
        max_epochs=hparams.epochs,
        default_root_dir=default_save_path,
        logger=tb_logger,
        callbacks=[checkpoint_callback, last_checkpoint_callback, earlystopping_callback, lr_monitor],
        val_check_interval=hparams.val_check_interval,
        strategy=hparams.strategy,
    )

    if find_batch_size_automatically:
        tuner = Tuner(trainer)

        # Auto-scale batch size by growing it exponentially (default)
        tuner.scale_batch_size(net, mode="binsearch")

    # This can be used to speed up training with newer GPUs:
    # https://lightning.ai/docs/pytorch/stable/advanced/speed.html#low-precision-matrix-multiplication
    # torch.set_float32_matmul_precision('medium')

    trainer.fit(model=net, datamodule=datamodule, ckpt_path=hparams.resume_from_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CASA UNet Regression Model")

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--fast_dev_run", action="store_true", help="Run a fast development test")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--val_check_interval", type=float, default=0.25, help="Validation check interval")

    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator")
    parser.add_argument("--strategy", type=str, default="auto", help="Training strategy")
    parser.add_argument("--output_path", type=str, default=None, help="Output path for checkpoints and logs")

    # Add arguments from the model-specific method without overwriting the parser
    parser = unet_regr.CASARegressionBase.add_model_specific_args(parser)
    args = parser.parse_args()

    # Number of input images - must match num_input_images in datamodule
    args.n_channels = 4
    # Number of predicted images - must match num_output_images in datamodule for comparing predictions to ground truth
    args.n_classes = 10
    args.num_input_images = 4  # Must match n_channels
    args.num_output_images = 10  # Must match n_classes

    args.model = "UNetDS_Attention"
    args.lr_patience = 4
    args.es_patience = 15
    # args.val_check_interval = 0.25
    args.kernels_per_layer = 2
    args.use_oversampled_dataset = True
    args.metrics_path = os.path.join(args.output_path, "metrics")

    # args.resume_from_checkpoint = f"lightning/precip_regression/{args.model}/UNetDS_Attention.ckpt"

    # train_regression(args, find_batch_size_automatically=False)

    # All the models below will be trained
    for m in ["UNet", "UNetDS", "UNet_Attention", "UNetDS_Attention"]:
        args.model = m
        print_training_start(m)
        train_regression(args, find_batch_size_automatically=False)

# Showing the metrics in the logs
# AdamW
# Learning Rate
# Patience
