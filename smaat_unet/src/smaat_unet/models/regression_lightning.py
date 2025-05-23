import argparse
import csv
import lightning.pytorch as pl
import numpy as np
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from smaat_unet.metric.precipitation_metrics import PrecipitationMetrics
from smaat_unet.utils.formatting import make_metrics_str
from smaat_unet.utils import dataset_precip


def record_metrics_to_csv(filepath, epoch, metrics):
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            headers = ["Epoch"] + list(metrics.keys())
            writer.writerow(headers)
        row = [epoch] + [metrics[key].item() for key in metrics]
        writer.writerow(row)


class UNetBase(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--model",
            type=str,
            default="UNet",
            choices=["UNet", "UNetDS", "UNetAttention", "UNetDSAttention", "PersistenceModel"],
        )
        parser.add_argument("--n_channels", type=int, default=12)
        parser.add_argument("--n_classes", type=int, default=1)
        parser.add_argument("--kernels_per_layer", type=int, default=1)
        parser.add_argument("--bilinear", type=bool, default=True)
        parser.add_argument("--reduction_ratio", type=int, default=16)
        parser.add_argument("--lr_patience", type=int, default=5)
        parser.add_argument("--threshold", type=float, default=0.5)
        parser.add_argument("--metrics_path", type=str, default="./metrics", help="Path to save metrics")
        parser.add_argument("--job_id", type=str, default=None, help="Optional job ID for metrics files")
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.train_metrics = PrecipitationMetrics(
            threshold=self.hparams.threshold if hasattr(self.hparams, "threshold") else 0.5
        )
        self.val_metrics = PrecipitationMetrics(
            threshold=self.hparams.threshold if hasattr(self.hparams, "threshold") else 0.5
        )
        self.test_metrics = PrecipitationMetrics(
            threshold=self.hparams.threshold if hasattr(self.hparams, "threshold") else 0.5
        )
        self.metrics_path = self.hparams.metrics_path
        self.job_id = self.hparams.job_id
        os.makedirs(self.metrics_path, exist_ok=True)
        self.training_outputs = []
        self.validation_outputs = []
        self.model = self.hparams.model

    def forward(self, x):
        pass

    def configure_optimizers(self):
        # 0.001 LR and 1e-4 WD are the standard
        opt = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4)
        scheduler = {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.1, patience=self.hparams.lr_patience
            ),
            "monitor": "val_loss",  # Default: val_loss
        }
        return [opt], [scheduler]

    def loss_func(self, y_pred, y_true):
        # Ensure consistent shapes before computing loss
        if y_pred.dim() > y_true.dim():
            y_pred = y_pred.squeeze(1)
        elif y_true.dim() > y_pred.dim():
            y_pred = y_pred.unsqueeze(1)

        # reduction="mean" is average of every pixel, but I want average of image
        return nn.functional.mse_loss(y_pred, y_true, reduction="sum") / y_true.size(0)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred, y)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            rank_zero_only=True,
        )
        self.train_metrics.update(y_pred.detach(), y.detach())
        metrics = {"train_loss": loss.detach().cpu()}
        self.training_outputs.append(metrics)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred.squeeze(), y)
        # Add sync_dist=True for distributed training
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, rank_zero_only=True)

        self.val_metrics.update(y_pred.detach(), y.detach())
        metrics = {"val_loss": loss.detach().cpu()}
        self.validation_outputs.append(metrics)

    def test_step(self, batch, batch_idx):
        """Calculate the loss (MSE per default) and other metrics on the test set normalized and denormalized."""
        x, y = batch
        y_pred = self(x)

        # Update the test metrics
        self.test_metrics.update(y_pred.detach(), y.detach())

    def on_test_epoch_end(self):
        """Compute and log all metrics at the end of the test epoch."""
        test_metrics_dict = self.test_metrics.compute()

        # Print all metrics in one line
        print(f"\n\nEpoch {self.current_epoch} - Test Metrics: {make_metrics_str(test_metrics_dict)}")

        # Reset the metrics for the next test epoch
        self.test_metrics.reset()

    def on_train_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        train_metrics_dict = self.train_metrics.compute()
        print(f"\n\nEpoch {self.current_epoch} - Train Metrics: {make_metrics_str(train_metrics_dict)}")

        self.train_metrics.reset()

        # Gather and average metrics across all processes
        keys = self.training_outputs[0].keys()
        metrics = {key: torch.stack([output[key] for output in self.training_outputs]).mean() for key in keys}
        metrics = {key: self.all_gather(metrics[key]).mean() for key in metrics.keys()}
        self.training_outputs.clear()

        # Save metrics only on rank zero
        if self.global_rank == 0:
            filename = (
                f"training_metrics_{self.job_id}_{self.model}.csv"
                if self.job_id
                else f"training_metrics_{self.model}.csv"
            )
            record_metrics_to_csv(os.path.join(self.metrics_path, filename), self.current_epoch, metrics)
            for key, value in metrics.items():
                self.log(f"avg_{key}", value)

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        val_metrics_dict = self.val_metrics.compute()

        print(f"\n\nEpoch {self.current_epoch} - Validation Metrics: {make_metrics_str(val_metrics_dict)}")
        self.val_metrics.reset()

        # Gather and average metrics across all processes
        keys = self.validation_outputs[0].keys()
        metrics = {key: torch.stack([output[key] for output in self.validation_outputs]).mean() for key in keys}
        metrics = {key: self.all_gather(metrics[key]).mean() for key in metrics.keys()}
        self.validation_outputs.clear()

        # Save metrics only on rank zero
        if self.global_rank == 0:
            filename = (
                f"validation_metrics_{self.job_id}_{self.model}.csv"
                if self.job_id
                else f"validation_metrics_{self.model}.csv"
            )
            record_metrics_to_csv(os.path.join(self.metrics_path, filename), self.current_epoch, metrics)
            for key, value in metrics.items():
                self.log(f"avg_{key}", value)


class PrecipRegressionBase(UNetBase):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = UNetBase.add_model_specific_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_input_images", type=int, default=12)
        parser.add_argument("--num_output_images", type=int, default=6)
        parser.add_argument("--valid_size", type=float, default=0.1)
        parser.add_argument("--use_oversampled_dataset", type=bool, default=True)
        parser.n_channels = parser.parse_args().num_input_images
        parser.n_classes = 1
        return parser

    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        self.train_dataset = None
        self.valid_dataset = None
        self.train_sampler = None
        self.valid_sampler = None

    def prepare_data(self):
        # train_transform = transforms.Compose([
        #     transforms.RandomHorizontalFlip()]
        # )
        train_transform = None
        valid_transform = None
        precip_dataset = (
            dataset_precip.precipitation_maps_oversampled_h5
            if self.hparams.use_oversampled_dataset
            else dataset_precip.precipitation_maps_h5
        )
        self.train_dataset = precip_dataset(
            in_file=self.hparams.dataset_folder,
            num_input_images=self.hparams.num_input_images,
            num_output_images=self.hparams.num_output_images,
            train=True,
            transform=train_transform,
        )
        self.valid_dataset = precip_dataset(
            in_file=self.hparams.dataset_folder,
            num_input_images=self.hparams.num_input_images,
            num_output_images=self.hparams.num_output_images,
            train=True,
            transform=valid_transform,
        )

        num_train = len(self.train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.hparams.valid_size * num_train))

        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.valid_sampler = SubsetRandomSampler(valid_idx)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            sampler=self.train_sampler,
            pin_memory=True,
            # The following can/should be tweaked depending on the number of CPU cores
            num_workers=1,
            persistent_workers=True,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.hparams.batch_size,
            sampler=self.valid_sampler,
            pin_memory=True,
            # The following can/should be tweaked depending on the number of CPU cores
            num_workers=1,
            persistent_workers=True,
        )
        return valid_loader


class PersistenceModel(UNetBase):
    def forward(self, x):
        return x[:, -1:, :, :]


class CASARegressionBase(UNetBase):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = UNetBase.add_model_specific_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_input_images", type=int, default=12, help="Number of input images")
        parser.add_argument("--num_output_images", type=int, default=6, help="Number of output images")
        parser.add_argument("--valid_size", type=float, default=0.1, help="Validation split size")
        parser.add_argument("--use_oversampled_dataset", type=bool, default=True, help="Use oversampled dataset")
        parser.n_channels = parser.parse_args().num_input_images
        parser.n_classes = 1
        return parser

    def __init__(self, hparams):
        super().__init__(hparams=hparams)
