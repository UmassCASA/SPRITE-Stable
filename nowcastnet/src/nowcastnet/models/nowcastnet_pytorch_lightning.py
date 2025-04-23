# Standard library imports

# Third party imports
import lightning as pl
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

# Local imports
from nowcastnet.layers.evolution.evolution_network import Evolution_Network
from nowcastnet.layers.generation.discriminator_3Dconv import (
    Temporal_Discriminator as Discriminator_3Dconv,
)
from nowcastnet.layers.generation.generative_network import (
    Generative_Decoder,
    Generative_Encoder,
)
from nowcastnet.layers.generation.noise_projector import Noise_Projector
from nowcastnet.layers.utils import make_grid, warp
from sprite_metrics.losses import (
    PrecipitationCRPS,
    compute_csi,
    compute_evaluation_score,
    compute_psd,
    loss_hinge_disc,
    loss_hinge_gen,
)


def weight_fn(y, precip_weight_cap=24.0):
    """
    Weight function for the grid cell loss.
    w(y) = max(y + 1, ceil)

    Args:
        y: Tensor of rainfall intensities.
        ceil: Custom ceiling for the weight function.

    Returns:
        Weights for each grid cell.
    """
    return torch.max(y + 1, torch.tensor(precip_weight_cap, device=y.device))


class Net(pl.LightningModule):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.pred_length = self.configs.total_length - self.configs.input_length

        self.evo_net = Evolution_Network(self.configs.input_length, self.pred_length, base_c=32)
        self.gen_enc = Generative_Encoder(self.configs.total_length, base_c=self.configs.ngf)
        self.gen_dec = Generative_Decoder(self.configs)
        self.proj = Noise_Projector(self.configs.ngf, configs)

        sample_tensor = torch.zeros(1, 1, self.configs.img_height, self.configs.img_width)
        self.grid = make_grid(sample_tensor)
        self.grid_lambda: float = 20.0

        self.automatic_optimization = False
        torch.autograd.set_detect_anomaly(True)
        self.crps_logistic = PrecipitationCRPS(method="logistic", sigma=3.0)

        if self.configs.discriminator_type == "conv3d":
            self.discriminator = Discriminator_3Dconv(
                num_frames_input=self.configs.input_length,
                num_frames_predict=self.pred_length,
                T=self.configs.total_length,
            )

    def forward(self, all_frames):
        all_frames.requires_grad_()

        batch = all_frames.shape[0]
        height = all_frames.shape[2]
        width = all_frames.shape[3]

        # Input Frames
        input_frames = all_frames[:, : self.configs.input_length].requires_grad_()
        input_frames = input_frames.reshape(batch, self.configs.input_length, height, width).requires_grad_()

        # Evolution Network
        intensity, motion = self.evo_net(input_frames)
        motion_ = motion.reshape(batch, self.pred_length, 2, height, width)
        intensity_ = intensity.reshape(batch, self.pred_length, 1, height, width)
        series = []
        last_frames = all_frames[:, (self.configs.input_length - 1) : self.configs.input_length, :, :]
        grid = self.grid.repeat(batch, 1, 1, 1)
        for i in range(self.pred_length):
            last_frames = warp(last_frames, motion_[:, i], grid.cuda(), mode="nearest", padding_mode="border")
            last_frames = last_frames + intensity_[:, i]
            series.append(last_frames)
        evo_result = torch.cat(series, dim=1)

        evo_result = evo_result / 128

        # Generative Network
        evo_feature = self.gen_enc(torch.cat([input_frames, evo_result], dim=1))

        noise = torch.randn(batch, self.configs.ngf, height // 32, width // 32).cuda()
        noise_feature = (
            self.proj(noise)
            .reshape(batch, -1, 4, 4, 8, 8)
            .permute(0, 1, 4, 5, 2, 3)
            .reshape(batch, -1, height // 8, width // 8)
        )

        feature = torch.cat([evo_feature, noise_feature], dim=1)
        gen_result = self.gen_dec(feature, evo_result)

        return gen_result.unsqueeze(-1)  # shape=(batch, T, H, W, Channel)

    def configure_optimizers(self):
        gen_optimizer = torch.optim.Adam(
            list(self.evo_net.parameters())
            + list(self.gen_enc.parameters())
            + list(self.gen_dec.parameters())
            + list(self.proj.parameters()),
            lr=self.configs.g_lr,
            betas=(0.5, 0.999),
        )
        dis_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.configs.d_lr, betas=(0.5, 0.999))
        return [gen_optimizer, dis_optimizer], []

    def training_step(self, batch, batch_idx):
        images, future_images = batch
        input_images = torch.cat((images, future_images), dim=1).squeeze(dim=2)
        input_images.requires_grad_(True)

        # Get optimizers
        gen_opt, dis_opt = self.optimizers()

        # ========== Generator Forward Pass ==========
        # Generate fake image sequences
        generated_images = self.forward(input_images)  # [B, T, H, W, 1]
        generated_images = generated_images.squeeze(-1)  # [B, T, H, W]
        future_images_no1Dim = future_images.squeeze(2)  # [B, T, H, W]

        # ========== Train Discriminator ==========
        # Input real and generated image sequences into the discriminator
        # Convert image sequences to shape [B, C, T, H, W], where C=1
        real_seq = future_images_no1Dim.unsqueeze(1)  # [B, 1, T, H, W]
        gen_seq = generated_images.detach().unsqueeze(1)  # [B, 1, T, H, W]

        if self.configs.discriminator_type == "conv3d":
            dis_loss = self.train_discriminator_conv3d(images, dis_opt, real_seq, gen_seq)
        else:
            dis_loss = self.train_discriminator_dgmr(dis_opt, real_seq, gen_seq, images)

        # ========== Train Generator ==========
        gen_opt.zero_grad()

        # Adversarial loss for generator (wants discriminator to think generated images are real)
        if self.configs.discriminator_type == "conv3d":
            conv3d_input = images.squeeze(2).unsqueeze(1)
            adv_loss = self.forward_discriminator_conv3d(
                torch.cat([conv3d_input, real_seq], dim=2), torch.cat([conv3d_input, gen_seq], dim=2)
            )

            # Pooling regularization loss
            # Average over spatial dimensions, resulting in [B, T]
            gen_pooled = generated_images.mean(dim=[2, 3])  # [B, T]
            real_pooled = future_images_no1Dim.mean(dim=[2, 3])
            pool_reg_loss = F.mse_loss(gen_pooled, real_pooled)

            # Total Generator Loss
            lambda_pool = self.configs.pool_reg_weight  # Pool regularization weight, needs to be defined in configs
            gen_loss = adv_loss + lambda_pool * pool_reg_loss
        else:
            predictions = [
                checkpoint(self.forward, input_images, use_reentrant=False)
                for _ in range(self.configs.generation_steps)
            ]
            print("Computing Grid Cell Loss")

            future_images_dgmr = future_images_no1Dim.unsqueeze(2)

            gen_mean = torch.stack(predictions, dim=0).mean(dim=0)  # Mean over samples
            grid_cell_reg = self.grid_regularizer(gen_mean, future_images_dgmr)

            # Concat along time dimension
            generated_sequence = [torch.cat([images, x.permute(0, 1, 4, 2, 3)], dim=1) for x in predictions]
            real_sequence = torch.cat([images, future_images_dgmr], dim=1)

            generated_scores = []
            for g_seq in generated_sequence:
                concatenated_inputs = torch.cat([real_sequence, g_seq], dim=0)
                concatenated_outputs = checkpoint(self.discriminator, concatenated_inputs, use_reentrant=False)
                # Split along the concatenated dimension, as discrimnator concatenates along dim=1
                score_real, score_generated = torch.split(
                    concatenated_outputs, [real_sequence.shape[0], g_seq.shape[0]], dim=0
                )
                generated_scores.append(score_generated)
            generator_disc_loss = loss_hinge_gen(torch.cat(generated_scores, dim=0))
            gen_loss = generator_disc_loss + self.grid_lambda * grid_cell_reg

        # Backpropagation and Generator Optimization
        self.manual_backward(gen_loss)
        gen_opt.step()

        crps_score_local = torch.tensor(self.crps_logistic(gen_seq, real_seq))

        csi = torch.tensor(compute_csi(gen_seq, real_seq, 0.9))

        psd = torch.tensor(compute_psd(gen_seq, real_seq, 0.9))

        # eval_score = compute_evaluation_score(csi, crps_score_local, gen_loss, dis_loss, [1, 1, 1])

        metrics = {
            "train_d_loss": dis_loss.to("cuda"),
            "train_g_loss": gen_loss.to("cuda"),
            "train_csi": csi.to("cuda"),
            "train_psd": psd.to("cuda"),
            "train_crps": crps_score_local.to("cuda"),
            # "train_eval_score": eval_score,
        }

        if self.configs.discriminator_type == "dgmr":
            metrics.update({"train_grid_cell_reg": grid_cell_reg.to("cuda")})

        self.log_dict(metrics, prog_bar=True, sync_dist=True, rank_zero_only=True)

        return metrics

    def train_discriminator_conv3d(self, images, dis_opt, real_seq, gen_seq):
        # Discriminator Forward Pass
        dis_opt.zero_grad()

        conv3d_input = images.squeeze(2).unsqueeze(1)
        dis_loss = self.forward_discriminator_conv3d(
            torch.cat([conv3d_input, real_seq], dim=2), torch.cat([conv3d_input, gen_seq], dim=2)
        )

        # Backpropagation and Discriminator Optimization
        self.manual_backward(dis_loss)
        dis_opt.step()
        return dis_loss

    def forward_discriminator_conv3d(self, real_seq, gen_seq):
        # Compute Discriminator Loss
        real_pred = self.discriminator(real_seq)
        fake_pred = self.discriminator(gen_seq)

        real_labels = torch.ones_like(real_pred)
        fake_labels = torch.zeros_like(fake_pred)
        dis_loss_real = F.binary_cross_entropy(real_pred, real_labels)
        dis_loss_fake = F.binary_cross_entropy(fake_pred, fake_labels)
        dis_loss = (dis_loss_real + dis_loss_fake) / 2

        return dis_loss

    def train_discriminator_dgmr(self, dis_opt, real_seq, gen_seq, images):
        # Discriminator Forward Pass
        dis_opt.zero_grad()

        discriminator_loss = self.forward_discriminator_dgmr(real_seq, gen_seq, images)

        # Backpropagation and Discriminator Optimization
        self.manual_backward(discriminator_loss)
        dis_opt.step()
        return discriminator_loss

    def forward_discriminator_dgmr(self, real_seq, gen_seq, images):
        real_seq = real_seq.permute(0, 2, 1, 3, 4)
        gen_seq = gen_seq.permute(0, 2, 1, 3, 4)

        generated_sequence = torch.cat([images, gen_seq], dim=1)
        real_sequence = torch.cat([images, real_seq], dim=1)

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

        return discriminator_loss

    def validation_step(self, batch, batch_idx):
        images, future_images = batch
        images.requires_grad_(False)
        future_images.requires_grad_(False)

        with torch.no_grad():
            images, future_images = batch
            input_images = torch.cat((images, future_images), dim=1).squeeze(dim=2)
            input_images.requires_grad_(True)

            # ========== Generator Forward Pass ==========
            # Generate fake image sequences
            generated_images = self.forward(input_images)  # [B, T, H, W, 1]
            generated_images = generated_images.squeeze(-1)  # [B, T, H, W]
            future_images_no1Dim = future_images.squeeze(2)  # [B, T, H, W]

            # ========== Train Discriminator ==========
            # Input real and generated image sequences into the discriminator
            # Convert image sequences to shape [B, C, T, H, W], where C=1
            real_seq = future_images_no1Dim.unsqueeze(1)  # [B, 1, T, H, W]
            gen_seq = generated_images.detach().unsqueeze(1)  # [B, 1, T, H, W]

            if self.configs.discriminator_type == "conv3d":
                conv3d_input = images.squeeze(2).unsqueeze(1)
                dis_loss = self.forward_discriminator_conv3d(
                    torch.cat([conv3d_input, real_seq], dim=2), torch.cat([conv3d_input, gen_seq], dim=2)
                )
            else:
                dis_loss = self.forward_discriminator_dgmr(real_seq, gen_seq, images)

            # ========== Train Generator ==========

            # Adversarial loss for generator (wants discriminator to think generated images are real)
            if self.configs.discriminator_type == "conv3d":
                conv3d_input = images.squeeze(2).unsqueeze(1)
                adv_loss = self.forward_discriminator_conv3d(
                    torch.cat([conv3d_input, real_seq], dim=2), torch.cat([conv3d_input, gen_seq], dim=2)
                )
                # Pooling regularization loss
                # Average over spatial dimensions, resulting in [B, T]
                gen_pooled = generated_images.mean(dim=[2, 3])  # [B, T]
                real_pooled = future_images_no1Dim.mean(dim=[2, 3])
                pool_reg_loss = F.mse_loss(gen_pooled, real_pooled)

                # Total Generator Loss
                lambda_pool = self.configs.pool_reg_weight  # Pool regularization weight, needs to be defined in configs
                gen_loss = adv_loss + lambda_pool * pool_reg_loss
            else:
                predictions = [
                    checkpoint(self.forward, input_images, use_reentrant=False)
                    for _ in range(self.configs.generation_steps)
                ]
                print("Computing Grid Cell Loss")

                future_images_dgmr = future_images_no1Dim.unsqueeze(2)

                gen_mean = torch.stack(predictions, dim=0).mean(dim=0)  # Mean over samples
                grid_cell_reg = self.grid_regularizer(gen_mean, future_images_dgmr)

                # Concat along time dimension
                generated_sequence = [torch.cat([images, x.permute(0, 1, 4, 2, 3)], dim=1) for x in predictions]
                real_sequence = torch.cat([images, future_images_dgmr], dim=1)

                generated_scores = []
                for g_seq in generated_sequence:
                    concatenated_inputs = torch.cat([real_sequence, g_seq], dim=0)
                    concatenated_outputs = checkpoint(self.discriminator, concatenated_inputs, use_reentrant=False)
                    # Split along the concatenated dimension, as discrimnator concatenates along dim=1
                    score_real, score_generated = torch.split(
                        concatenated_outputs, [real_sequence.shape[0], g_seq.shape[0]], dim=0
                    )
                    generated_scores.append(score_generated)
                generator_disc_loss = loss_hinge_gen(torch.cat(generated_scores, dim=0))
                gen_loss = generator_disc_loss + self.grid_lambda * grid_cell_reg

            crps_score_local = torch.tensor(self.crps_logistic(gen_seq, real_seq))

            csi = torch.tensor(compute_csi(gen_seq, real_seq, 0.9))

            psd = torch.tensor(compute_psd(gen_seq, real_seq, 0.9))

            # eval_score = compute_evaluation_score(csi, crps_score_local, gen_loss, dis_loss, [1, 1, 1])

            metrics = {
                "val_d_loss": dis_loss.to("cuda"),
                "val_g_loss": gen_loss.to("cuda"),
                "val_csi": csi.to("cuda"),
                "val_psd": psd.to("cuda"),
                "val_crps": crps_score_local.to("cuda"),
                # "val_eval_score": eval_score,
            }

            if self.configs.discriminator_type == "dgmr":
                metrics.update({"val_grid_cell_reg": grid_cell_reg.to("cuda")})

            self.log_dict(metrics, prog_bar=True, sync_dist=True, rank_zero_only=True)

            # plot_precipitation_radar(generated_images[0])
            # plot_precipitation_radar(future_images_no1Dim[0])

            return metrics

        ####################
        # Visualize Output #
        ####################

        # if self.visualize:
        #     generated_images = self(images)
        #     self.visualize_step(
        #         images,
        #         future_images,
        #         generated_images,
        #         self.global_iteration,
        #         step="val",
        #     )

    def load_checkpoint(self, checkpoint_path):
        """
        Load a model checkpoint from the specified path.

        Args:
            checkpoint_path (str): Path to the checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            RuntimeError: If there is an error loading the checkpoint.
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.load_state_dict(checkpoint["state_dict"])
            print(f"Checkpoint loaded successfully from {checkpoint_path}")
        except FileNotFoundError:
            print(f"Error: Checkpoint file not found at {checkpoint_path}")
            raise
        except RuntimeError as e:
            print(f"Error loading checkpoint: {e}")
            raise

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, outputs):
        pass

    def on_epoch_end(self):
        pass

    def on_train_epoch_end(self):
        pass


def plot_precipitation_radar(tensor):
    """
    Plot the last four timesteps of a precipitation radar image.

    Parameters:
        tensor (torch.Tensor): Input tensor of shape [1, 1, T, W, H]
    """
    # Check the dimensions of the input tensor

    # Extract the last four timesteps, shape [4, W, H]
    last_four = tensor[-4:, :, :]

    # Convert tensor to NumPy array
    images = last_four.cpu().numpy()

    # Create subplots
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i in range(4):
        ax = axes[i]
        img = images[i]
        im = ax.imshow(img, cmap="jet")
        ax.set_title(f"Timestep {i}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
