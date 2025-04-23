import os
import csv
import torch
import torchvision
from torch.utils.checkpoint import checkpoint
import pytorch_lightning as pl


from dgmr.common import ContextConditioningStack, LatentConditioningStack
from dgmr.discriminators import Discriminator
from dgmr.generators import Generator, Sampler
from dgmr.hub import NowcastingModelHubMixin
from sprite_metrics.losses import (
    GridCellLoss,
    NowcastingLoss,
    grid_cell_regularizer,
    loss_hinge_disc,
    loss_hinge_gen,
    PrecipitationCRPS,
    compute_csi,
    compute_psd,
)


def record_metric(filepath, metric_name, value, job_id=None):
    """Record a single metric value to a file with job_id in the filename."""
    filename = f"{metric_name}_{job_id}.txt" if job_id else f"{metric_name}.txt"
    with open(os.path.join(filepath, filename), "a") as file:
        file.write(f"{value.item()}\n")


def record_metrics_to_csv(filepath, epoch, metrics):
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            headers = ["Epoch"] + list(metrics.keys())
            writer.writerow(headers)
        row = [epoch] + [metrics[key].item() for key in metrics]
        writer.writerow(row)


class DGMR(pl.LightningModule, NowcastingModelHubMixin):
    """Deep Generative Model of Radar"""

    def __init__(
        self,
        forecast_steps: int = 18,
        input_channels: int = 1,
        output_shape: int = 256,
        gen_lr: float = 5e-5,
        disc_lr: float = 2e-4,
        visualize: bool = False,
        conv_type: str = "standard",
        num_samples: int = 6,
        grid_lambda: float = 20.0,
        beta1: float = 0.0,
        beta2: float = 0.999,
        latent_channels: int = 768,
        context_channels: int = 384,
        generation_steps: int = 2,  # 6 in paper
        metrics_path: str = "./metrics",
        job_id: str = None,
        **kwargs,
    ):
        """
        Nowcasting GAN is an attempt to recreate DeepMind's Skillful Nowcasting GAN from https://arxiv.org/abs/2104.00954
        but slightly modified for multiple satellite channels

        Args:
            forecast_steps: Number of steps to predict in the future
            input_channels: Number of input channels per image
            visualize: Whether to visualize output during training
            gen_lr: Learning rate for the generator
            disc_lr: Learning rate for the discriminators, shared for both temporal and spatial discriminator
            conv_type: Type of 2d convolution to use, see satflow/models/utils.py for options
            beta1: Beta1 for Adam optimizer
            beta2: Beta2 for Adam optimizer
            num_samples: Number of samples of the latent space to sample for training/validation
            grid_lambda: Lambda for the grid regularization loss
            output_shape: Shape of the output predictions, generally should be same as the input shape
            generation_steps: Number of generation steps to use in forward pass, in paper is 6 and the best is
                chosen for the loss this results in huge amounts of GPU memory though, so less might work better
                for training.
            latent_channels: Number of channels that the latent space should be reshaped to,
                input dimension into ConvGRU, also affects the number of channels for other linked inputs/outputs
            pretrained:
        """
        super().__init__()
        config = locals()
        config.pop("__class__")
        config.pop("self")
        self.config = kwargs.get("config", config)
        input_channels = self.config["input_channels"]
        forecast_steps = self.config["forecast_steps"]
        output_shape = self.config["output_shape"]
        gen_lr = self.config["gen_lr"]
        disc_lr = self.config["disc_lr"]
        conv_type = self.config["conv_type"]
        num_samples = self.config["num_samples"]
        grid_lambda = self.config["grid_lambda"]
        beta1 = self.config["beta1"]
        beta2 = self.config["beta2"]
        latent_channels = self.config["latent_channels"]
        context_channels = self.config["context_channels"]
        visualize = self.config["visualize"]
        self.gen_lr = gen_lr
        self.disc_lr = disc_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.discriminator_loss = NowcastingLoss()
        self.grid_regularizer = GridCellLoss()
        self.grid_lambda = grid_lambda
        self.num_samples = num_samples
        self.visualize = visualize
        self.latent_channels = latent_channels
        self.context_channels = context_channels
        self.input_channels = input_channels
        self.generation_steps = generation_steps
        self.conditioning_stack = ContextConditioningStack(
            input_channels=input_channels,
            conv_type=conv_type,
            output_channels=self.context_channels,
        )
        self.latent_stack = LatentConditioningStack(
            shape=(8 * self.input_channels, output_shape // 32, output_shape // 32),
            output_channels=self.latent_channels,
        )
        self.sampler = Sampler(
            forecast_steps=forecast_steps,
            latent_channels=self.latent_channels,
            context_channels=self.context_channels,
        )
        self.generator = Generator(self.conditioning_stack, self.latent_stack, self.sampler)
        self.crps_logistic = PrecipitationCRPS(method="logistic", sigma=3.0)
        self.discriminator = Discriminator(input_channels)
        self.save_hyperparameters()

        self.global_iteration = 0
        self.training_outputs = []
        self.validation_outputs = []

        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        torch.autograd.set_detect_anomaly(True)

        self.metrics_path = metrics_path
        os.makedirs(self.metrics_path, exist_ok=True)

        self.training_outputs = []
        self.validation_outputs = []

        self.job_id = job_id

    def forward(self, x):
        x = self.generator(x)
        return x

    def training_step(self, batch, batch_idx):
        images, future_images = batch

        ### Images must be of type float32. Our data is already float32
        # images = images.float()
        # future_images = future_images.float()

        self.global_iteration += 1
        g_opt, d_opt = self.optimizers()

        ##########################
        # Optimize Discriminator #
        ##########################

        # Two discriminator steps per generator step
        for _ in range(2):
            d_opt.zero_grad()
            # self is instance of DGMR
            # predictions = self(images)
            predictions = checkpoint(self.forward, images, use_reentrant=False)
            # Cat along time dimension [B, T, C, H, W]
            generated_sequence = torch.cat([images, predictions], dim=1)
            real_sequence = torch.cat([images, future_images], dim=1)
            # Cat long batch for the real+generated
            concatenated_inputs = torch.cat([real_sequence, generated_sequence], dim=0)

            concatenated_outputs = self.discriminator(concatenated_inputs)
            score_real, score_generated = torch.split(
                concatenated_outputs,
                [real_sequence.shape[0], generated_sequence.shape[0]],
                dim=0,
            )

            score_real_spatial, score_real_temporal = torch.split(score_real, 1, dim=1)
            score_generated_spatial, score_generated_temporal = torch.split(score_generated, 1, dim=1)
            discriminator_loss = loss_hinge_disc(score_generated_spatial, score_real_spatial) + loss_hinge_disc(
                score_generated_temporal, score_real_temporal
            )

            self.manual_backward(discriminator_loss)

            d_opt.step()

        ######################
        # Optimize Generator #
        ######################

        # self is instance of DGMR
        # predictions = [self(images) for _ in range(self.generation_steps)]
        predictions = [checkpoint(self.forward, images, use_reentrant=False) for _ in range(self.generation_steps)]

        grid_cell_reg = grid_cell_regularizer(torch.stack(predictions, dim=0), future_images)

        # Concat along time dimension
        generated_sequence = [torch.cat([images, x], dim=1) for x in predictions]
        real_sequence = torch.cat([images, future_images], dim=1)

        # Cat long batch for the real+generated, for each example in the range
        # For each of the 6 examples
        generated_scores = []
        for g_seq in generated_sequence:
            concatenated_inputs = torch.cat([real_sequence, g_seq], dim=0)
            concatenated_outputs = self.discriminator(concatenated_inputs)
            # Split along the concatenated dimension, as discrimnator concatenates along dim=1
            score_real, score_generated = torch.split(
                concatenated_outputs, [real_sequence.shape[0], g_seq.shape[0]], dim=0
            )
            generated_scores.append(score_generated)
        generator_disc_loss = loss_hinge_gen(torch.cat(generated_scores, dim=0))
        generator_loss = generator_disc_loss + self.grid_lambda * grid_cell_reg
        g_opt.zero_grad()
        self.manual_backward(generator_loss)
        g_opt.step()

        crps_score_local = torch.tensor(
            [self.crps_logistic(generated_image, real_sequence) for generated_image in generated_sequence]
        ).mean()

        csi = torch.tensor(
            [compute_csi(generated_image, real_sequence, 0.9) for generated_image in generated_sequence]
        ).mean()

        psd = torch.tensor(
            [compute_psd(generated_image, real_sequence, 0.9) for generated_image in generated_sequence]
        ).mean()

        metrics = {
            "train_d_loss": discriminator_loss.detach().cpu(),
            "train_g_loss": generator_loss.detach().cpu(),
            "train_grid_loss": grid_cell_reg.detach().cpu(),
            "train_csi": csi.detach().cpu(),
            "train_psd": psd.detach().cpu(),
            "train_crps": crps_score_local.detach().cpu(),
        }

        self.log_dict(metrics, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.training_outputs.append(metrics)

        # log sampled images
        if self.visualize:
            generated_images = self(images)
            self.visualize_step(
                images,
                future_images,
                generated_images,
                self.global_iteration,
                step="train",
            )

        return metrics

    def validation_step(self, batch, batch_idx):
        images, future_images = batch
        images = images
        future_images = future_images
        ##########################
        # Optimize Discriminator #
        ##########################
        with torch.no_grad():  # OOM Addition
            # Two discriminator steps per generator step
            for _ in range(2):
                predictions = self(images)
                # Cat along time dimension [B, T, C, H, W]
                generated_sequence = torch.cat([images, predictions], dim=1)
                real_sequence = torch.cat([images, future_images], dim=1)

                # Cat long batch for the real+generated
                concatenated_inputs = torch.cat([real_sequence, generated_sequence], dim=0)
                concatenated_outputs = self.discriminator(concatenated_inputs)
                score_real, score_generated = torch.split(
                    concatenated_outputs,
                    [real_sequence.shape[0], generated_sequence.shape[0]],
                    dim=0,
                )
                score_real_spatial, score_real_temporal = torch.split(score_real, 1, dim=1)
                score_generated_spatial, score_generated_temporal = torch.split(score_generated, 1, dim=1)
                discriminator_loss = loss_hinge_disc(score_generated_spatial, score_real_spatial) + loss_hinge_disc(
                    score_generated_temporal, score_real_temporal
                )

            ######################
            # Optimize Generator #
            ######################
            predictions = [self(images) for _ in range(self.generation_steps)]
            grid_cell_reg = grid_cell_regularizer(torch.stack(predictions, dim=0), future_images)
            # Concat along time dimension
            generated_sequence = [torch.cat([images, x], dim=1) for x in predictions]
            real_sequence = torch.cat([images, future_images], dim=1)
            # Cat long batch for the real+generated, for each example in the range
            # For each of the 6 examples
            generated_scores = []
            for g_seq in generated_sequence:
                concatenated_inputs = torch.cat([real_sequence, g_seq], dim=0)
                concatenated_outputs = self.discriminator(concatenated_inputs)
                # Split along the concatenated dimension, as discrimnator concatenates along dim=1
                score_real, score_generated = torch.split(
                    concatenated_outputs, [real_sequence.shape[0], g_seq.shape[0]], dim=0
                )
                generated_scores.append(score_generated)
            generator_disc_loss = loss_hinge_gen(torch.cat(generated_scores, dim=0))
            generator_loss = generator_disc_loss + self.grid_lambda * grid_cell_reg

            crps_score_local = torch.tensor(
                [self.crps_logistic(generated_image, real_sequence) for generated_image in generated_sequence]
            ).mean()

            csi = torch.tensor(
                [compute_csi(generated_image, real_sequence, 0.9) for generated_image in generated_sequence]
            ).mean()

            psd = torch.tensor(
                [compute_psd(generated_image, real_sequence, 0.9) for generated_image in generated_sequence]
            ).mean()

        metrics = {
            "val_d_loss": discriminator_loss.detach().cpu(),
            "val_g_loss": generator_loss.detach().cpu(),
            "val_grid_loss": grid_cell_reg.detach().cpu(),
            "val_csi": csi.detach().cpu(),
            "val_psd": psd.detach().cpu(),
            "val_crps": crps_score_local.detach().cpu(),
        }

        self.log_dict(metrics, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.validation_outputs.append(metrics)

        # log sampled images
        if self.visualize:
            # generate images
            generated_images = self(images)
            self.visualize_step(
                images,
                future_images,
                generated_images,
                self.global_iteration,
                step="val",
            )

        return metrics

    def configure_optimizers(self):
        b1 = self.beta1
        b2 = self.beta2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.gen_lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.disc_lr, betas=(b1, b2))

        return [opt_g, opt_d], []

    def visualize_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        batch_idx: int,
        step: str,
    ) -> None:
        # the logger you used (in this case tensorboard)
        tensorboard = self.logger.experiment[0]
        # Timesteps per channel
        images = x[0].cpu().detach()
        future_images = y[0].cpu().detach()
        generated_images = y_hat[0].cpu().detach()
        for i, t in enumerate(images):  # Now would be (C, H, W)
            t = [torch.unsqueeze(img, dim=0) for img in t]
            image_grid = torchvision.utils.make_grid(t, nrow=self.input_channels)
            tensorboard.add_image(f"{step}/Input_Image_Stack_Frame_{i}", image_grid, global_step=batch_idx)
            t = [torch.unsqueeze(img, dim=0) for img in future_images[i]]
            image_grid = torchvision.utils.make_grid(t, nrow=self.input_channels)
            tensorboard.add_image(f"{step}/Target_Image_Frame_{i}", image_grid, global_step=batch_idx)
            t = [torch.unsqueeze(img, dim=0) for img in generated_images[i]]
            image_grid = torchvision.utils.make_grid(t, nrow=self.input_channels)
            tensorboard.add_image(f"{step}/Generated_Image_Frame_{i}", image_grid, global_step=batch_idx)

    def on_train_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        # Gather and average metrics across all processes
        keys = self.training_outputs[0].keys()
        metrics = {key: torch.stack([output[key] for output in self.training_outputs]).mean() for key in keys}
        metrics = {key: self.all_gather(metrics[key]).mean() for key in metrics.keys()}
        self.training_outputs.clear()

        # Now, we can save the metrics only on rank zero
        if self.global_rank == 0:
            filename = f"training_metrics_{self.job_id}.csv" if self.job_id else "training_metrics.csv"
            record_metrics_to_csv(os.path.join(self.metrics_path, filename), self.current_epoch, metrics)
            for key, value in metrics.items():
                self.log(f"avg_{key}", value)

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        # Gather and average metrics across all processes
        keys = self.validation_outputs[0].keys()
        metrics = {key: torch.stack([output[key] for output in self.validation_outputs]).mean() for key in keys}
        metrics = {key: self.all_gather(metrics[key]).mean() for key in metrics.keys()}
        self.validation_outputs.clear()

        # Now, we can save the metrics only on rank zero
        if self.global_rank == 0:
            filename = f"validation_metrics_{self.job_id}.csv" if self.job_id else "validation_metrics.csv"
            record_metrics_to_csv(os.path.join(self.metrics_path, filename), self.current_epoch, metrics)
            for key, value in metrics.items():
                self.log(f"avg_{key}", value)
