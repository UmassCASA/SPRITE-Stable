import os
import csv
import pytorch_lightning as pl
import torch
import torchvision

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
    compute_evaluation_score,
)


def record_metric(filepath, metric_name, value):
    with open(os.path.join(filepath, f"{metric_name}.txt"), "a") as file:
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
            generation_steps: Number of generation steps to use in forward pass, in paper is 6 and
                the best is chosen for the loss this results in huge amounts of GPU memory though,
                so less might work better for training.
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

        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        torch.autograd.set_detect_anomaly(True)

        self.metrics_path = metrics_path
        os.makedirs(self.metrics_path, exist_ok=True)

        self.training_outputs = []
        self.validation_outputs = []

        self.is_forwarding = True
        self.is_backprop = not self.is_forwarding

        self.generator_loss = None
        self.discriminator_loss_val = None
        self.grid_cell_reg = None
        self.images = None
        self.future_images = None

        self.g_opt = None
        self.d_opt = None

    def forward(self, x):
        x = self.generator(x)
        return x

    def set_forwarding(self):
        self.is_forwarding = True
        self.is_backprop = not self.is_forwarding

    def set_backprop(self):
        self.is_forwarding = False
        self.is_backprop = not self.is_forwarding

    def fl_forward(self):
        # Optimize Discriminator
        # self.discriminator_loss_val = self.optimize_discriminator(self.images, self.future_images)
        self.discriminator_loss_val = torch.tensor(1000)

        # Optimize Generator
        # self.generator_loss, self.grid_cell_reg = self.optimize_generator(self.images, self.future_images)
        self.generator_loss, self.grid_cell_reg = (torch.tensor(1000), torch.tensor(1000))

    def set_generator_loss(self, generator_loss):
        self.generator_loss = generator_loss

    def get_generator_loss(self):
        return self.generator_loss

    def training_step(self, batch, batch_idx):
        if self.d_opt is None or self.g_opt is None:
            self.g_opt, self.d_opt = self.optimizers()

        if self.is_forwarding and not self.is_backprop:
            self.generator_loss = None
            self.discriminator_loss_val = None
            self.grid_cell_reg = None

            self.global_iteration += 1
            self.images, self.future_images = batch
            self.fl_forward()
        elif self.is_backprop and not self.is_forwarding:
            self.backward_step(self.generator_loss, self.g_opt)
            self.backward_step(self.discriminator_loss_val, self.d_opt)
            # Calculate Metrics
            metrics = self.calculate_metrics(
                self.images, self.future_images, self.discriminator_loss_val, self.generator_loss, self.grid_cell_reg
            )

            self.log_dict(metrics, prog_bar=True, sync_dist=True, rank_zero_only=True)
            self.training_outputs.append(metrics)

        # Generate and log images
        generated_images = self(self.images)
        if self.visualize:
            self.visualize_step(
                self.images,
                self.future_images,
                generated_images,
                self.global_iteration,
                step="train",
            )

        return None

    def optimize_discriminator(self, images, future_images):
        discriminator_loss = 0.0

        # Two discriminator steps per generator step
        for _ in range(2):
            predictions = self(images)
            generated_sequence = torch.cat([images, predictions], dim=1)
            real_sequence = torch.cat([images, future_images], dim=1)
            concatenated_inputs = torch.cat([real_sequence, generated_sequence], dim=0)

            concatenated_outputs = self.discriminator(concatenated_inputs)
            score_real, score_generated = torch.split(
                concatenated_outputs,
                [real_sequence.shape[0], generated_sequence.shape[0]],
                dim=0,
            )

            score_real_spatial, score_real_temporal = torch.split(score_real, 1, dim=1)
            score_generated_spatial, score_generated_temporal = torch.split(score_generated, 1, dim=1)
            step_discriminator_loss = loss_hinge_disc(score_generated_spatial, score_real_spatial) + loss_hinge_disc(
                score_generated_temporal, score_real_temporal
            )

            discriminator_loss += step_discriminator_loss

        del predictions, generated_sequence, concatenated_inputs
        torch.cuda.empty_cache()

        return discriminator_loss

    def optimize_generator(self, images, future_images):
        predictions = [self(images) for _ in range(self.generation_steps)]
        grid_cell_reg = grid_cell_regularizer(torch.stack(predictions, dim=0), future_images)

        generated_sequence = [torch.cat([images, x], dim=1) for x in predictions]
        real_sequence = torch.cat([images, future_images], dim=1)

        generated_scores = []
        for g_seq in generated_sequence:
            concatenated_inputs = torch.cat([real_sequence, g_seq], dim=0)
            concatenated_outputs = self.discriminator(concatenated_inputs)
            score_real, score_generated = torch.split(
                concatenated_outputs, [real_sequence.shape[0], g_seq.shape[0]], dim=0
            )
            generated_scores.append(score_generated)

        generator_disc_loss = loss_hinge_gen(torch.cat(generated_scores, dim=0))
        generator_loss = generator_disc_loss + self.grid_lambda * grid_cell_reg

        return generator_loss, grid_cell_reg

    def backward_step(self, loss, optimizer):
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

    def calculate_metrics(self, images, future_images, discriminator_loss, generator_loss, grid_cell_reg):
        predictions = [self(images) for _ in range(self.generation_steps)]
        real_sequence = torch.cat([images, future_images], dim=1)

        crps_score_local = torch.tensor(
            [self.crps_logistic(generated_image, real_sequence) for generated_image in predictions]
        ).mean()

        csi = torch.tensor([compute_csi(generated_image, real_sequence, 0.9) for generated_image in predictions]).mean()

        psd = torch.tensor([compute_psd(generated_image, real_sequence, 0.9) for generated_image in predictions]).mean()

        eval_score = compute_evaluation_score(
            csi, crps_score_local, generator_loss, discriminator_loss, weights=[1, 1, 1]
        )

        metrics = {
            "train_eval_score": eval_score.to("cuda"),
            "train_d_loss": discriminator_loss.to("cuda"),
            "train_g_loss": generator_loss.to("cuda"),
            "train_grid_loss": grid_cell_reg.to("cuda"),
            "train_csi": csi.to("cuda"),
            "train_psd": psd.to("cuda"),
            "train_crps": crps_score_local.to("cuda"),
        }

        return metrics

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            images, future_images = batch
            ##########################
            # Optimize Discriminator #
            ##########################
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

            eval_score = compute_evaluation_score(
                csi, crps_score_local, generator_loss, discriminator_loss, weights=[1, 1, 1]
            )

            metrics = {
                "val_eval_score": eval_score.to("cuda"),
                "val_d_loss": discriminator_loss.to("cuda"),
                "val_g_loss": generator_loss.to("cuda"),
                "val_grid_loss": grid_cell_reg.to("cuda"),
                "val_csi": csi.to("cuda"),
                "val_psd": psd.to("cuda"),
                "val_crps": crps_score_local.to("cuda"),
            }

            self.log_dict(metrics, prog_bar=True, sync_dist=True, rank_zero_only=True)
            self.validation_outputs.append(metrics)

            print()

            # generate images
            generated_images = self(images)
            # log sampled images
            if self.visualize:
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
        # Ensure aggregation happens only on rank zero
        if self.global_rank != 0 or self.trainer.sanity_checking or len(self.training_outputs) == 0:
            return

        # Gather and average metrics across all processes
        keys = self.training_outputs[0].keys()
        metrics = {key: torch.stack([output[key] for output in self.training_outputs]).mean() for key in keys}
        metrics = {key: self.all_gather(metrics[key]).mean() for key in metrics.keys()}
        self.training_outputs.clear()

        record_metrics_to_csv(os.path.join(self.metrics_path, "training_metrics.csv"), self.current_epoch, metrics)
        for key, value in metrics.items():
            self.log(f"avg_{key}", value)

    def on_validation_epoch_end(self):
        # Ensure aggregation happens only on rank zero
        if self.global_rank != 0 or self.trainer.sanity_checking:
            return

        # Gather and average metrics across all processes
        keys = self.validation_outputs[0].keys()
        metrics = {key: torch.stack([output[key] for output in self.validation_outputs]).mean() for key in keys}
        metrics = {key: self.all_gather(metrics[key]).mean() for key in metrics.keys()}
        self.validation_outputs.clear()

        record_metrics_to_csv(os.path.join(self.metrics_path, "validation_metrics.csv"), self.current_epoch, metrics)
        for key, value in metrics.items():
            self.log(f"avg_{key}", value)
