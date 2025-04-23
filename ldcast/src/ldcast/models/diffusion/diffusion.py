"""
From https://github.com/CompVis/latent-diffusion/main/ldm/models/diffusion/ddpm.py
Pared down to simplify code.

The original file acknowledges:
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
"""

import torch
import numpy as np
import lightning.pytorch as pl
from contextlib import contextmanager
from functools import partial

from matplotlib import pyplot as plt

from .utils import make_beta_schedule, extract_into_tensor
from .ema import LitEma
from ldcast.features.transform import Antialiasing

from sprite_metrics.losses import (
    PrecipitationCRPS,
    compute_csi,
    compute_psd,
)

antialiasing = Antialiasing()


def transform_precip(R):
    R_min_value = 0.1
    R_zero_value = 0.02
    log_R_mean = 1.643
    log_R_std = 6.405

    x = R.detach().cpu().numpy()
    x[~(x >= R_min_value)] = R_zero_value
    x = np.log10(x)
    x -= log_R_mean
    x /= log_R_std
    x = antialiasing(x)
    return torch.Tensor(x).to(device="cuda")


def inv_transform_precip(x):
    R_min_output = 0.1
    R_max_output = 128
    log_R_mean = 1.643
    log_R_std = 6.405

    x = x.detach().cpu()
    x *= log_R_std
    x += log_R_mean
    R = torch.pow(10, x)
    # R = x
    if R_min_output:
        R[R < R_min_output] = 0.0
    if R_max_output is not None:
        R[R > R_max_output] = R_max_output
    # return R.to(device="cpu").numpy()
    return R.to(device="cuda")


class LatentDiffusion(pl.LightningModule):
    def __init__(
        self,
        model,
        autoencoder,
        context_encoder=None,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        use_ema=True,
        lr=1e-4,
        lr_warmup=0,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        parameterization="eps",  # all assuming fixed variance schedules
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.autoencoder = autoencoder.requires_grad_(False)
        self.conditional = context_encoder is not None
        self.context_encoder = context_encoder
        self.lr = lr
        self.lr_warmup = lr_warmup

        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)

        self.register_schedule(
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

        self.loss_type = loss_type
        self.process_name = "val"

    def register_schedule(
        self, beta_schedule="linear", timesteps=1000, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
    ):
        betas = make_beta_schedule(
            beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s
        )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, "alphas have to be defined for each timestep"

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def apply_model(self, x_noisy, t, cond=None, return_ids=False):
        if self.conditional:
            cond = self.context_encoder(cond)
        with self.ema_scope():
            return self.model(x_noisy, t, context=cond)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == "l2":
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction="none")
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None, context=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t, context=context)

        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Parameterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean()
        loss_ema = self.get_loss(model_out, x_start, mean=False).mean()

        loss_dict = {f"{self.process_name}_{self.loss_type}_loss_ema": loss_ema} | {"loss": loss}

        # write only at self.global_step % 10 == 0
        if (self.global_step % 10) == 0:
            # in Lightning, self.global_step could be obtained by trainer.global_step
            step = self.global_step

            csv_file = "/home/zhexu_umass_edu/PycharmProjects/SPRITE/ldcast/debug/tensor_dist.csv"

            def log_tensor_stats(tensor, name):
                mean_ = tensor.mean().item()
                std_ = tensor.std().item()
                min_ = tensor.min().item()
                max_ = tensor.max().item()
                # wirte to csv, format: step,name_mean,value
                with open(csv_file, "a") as f:
                    f.write(f"{step},{name}_mean,{mean_}\n")
                    f.write(f"{step},{name}_std,{std_}\n")
                    f.write(f"{step},{name}_min,{min_}\n")
                    f.write(f"{step},{name}_max,{max_}\n")

            with open(csv_file, "w") as f:
                f.write("step,metric_name,value\n")

            log_tensor_stats(model_out, "model_out")
            log_tensor_stats(target, "target")
            log_tensor_stats(x_noisy, "x_noisy")

            #  TODO: Remove here
            R_pred = self.autoencoder.decode(model_out)[0].detach().squeeze().cpu().numpy()

            for t in range(2):
                plt.figure()
                plt.imshow(R_pred[t], cmap="jet", origin="lower")
                plt.title(f"prediction: {t + 1}")
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.colorbar(label="intensity")
                plt.tight_layout()
                plt.show()

        metrics = self.get_metrics(model_out, x_start, process_name=self.process_name, loss_dict=loss_dict)

        return metrics

    def forward(self, x, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def shared_step(self, batch):
        (x, y) = batch
        x[0][0] = transform_precip(x[0][0])
        y = transform_precip(y)
        y = self.autoencoder.encode(y)[0]

        context = self.context_encoder(x) if self.conditional else None
        return self(y, context=context)

    def training_step(self, batch, batch_idx):
        self.process_name = "train"

        # TODO: Remove here
        # R_pred = batch[0][0][0].squeeze().cpu().numpy()[0]
        #
        # for t in range(2):
        #     plt.figure()
        #     plt.imshow(R_pred[t], cmap='jet', origin='lower')
        #     plt.title(f"precipitation_filed {t + 1}")
        #     plt.xlabel("X")
        #     plt.ylabel("Y")
        #     plt.colorbar(label='intensity')
        #     plt.tight_layout()
        #     plt.show()

        metrics = self.shared_step(batch)
        self.log_dict(metrics, prog_bar=True, sync_dist=True, rank_zero_only=True)

        return metrics

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.process_name = "val"

        metrics = self.shared_step(batch)
        with self.ema_scope():
            metrics_ema = self.shared_step(batch)
            self.log_dict(metrics_ema, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log_dict(metrics, prog_bar=True, sync_dist=True, rank_zero_only=True)

        return metrics

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,
    #         betas=(0.5, 0.9), weight_decay=1e-3)
    #     reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer, patience=3, factor=0.25, verbose=True
    #     )
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": reduce_lr,
    #             "monitor": "val_loss_ema",
    #             "frequency": 1,
    #         },
    #     }
    #
    # def optimizer_step(
    #     self,
    #     epoch,
    #     batch_idx,
    #     optimizer,
    #     optimizer_idx,
    #     optimizer_closure,
    #     **kwargs
    # ):
    #     if self.trainer.global_step < self.lr_warmup:
    #         lr_scale = (self.trainer.global_step+1) / self.lr_warmup
    #         for pg in optimizer.param_groups:
    #             pg['lr'] = lr_scale * self.lr
    #
    #     super().optimizer_step(
    #         epoch, batch_idx, optimizer,
    #         optimizer_idx, optimizer_closure,
    #         **kwargs
    #     )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, *args, **kwargs):
        # Adjust warmup lr before executing optimizing steps
        if self.trainer.global_step < self.hparams.lr_warmup:
            # Calculating current lr scale(linear increasing from 0 to 1)
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.hparams.lr_warmup)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr

        # Call optimizer to executing further weight updates, and call closure inside(forward + backward)
        optimizer.step(closure=optimizer_closure)

        # (Lightning auto optimization will handle loss.backward() and gradient cleaning through closure)

    def get_metrics(self, gen_seq, real_seq, process_name="val", loss_dict=None):
        crps_logistic = PrecipitationCRPS(method="logistic", sigma=3.0)

        crps_score_local = torch.tensor(crps_logistic(gen_seq, real_seq))

        csi = torch.tensor(compute_csi(gen_seq, real_seq, 0.9))

        psd = torch.tensor(compute_psd(gen_seq, real_seq, 0.9))

        # eval_score = compute_evaluation_score(csi, crps_score_local, loss_dict["loss"], loss_dict["loss"],
        #                                       weights=[0.5, 0.5, 1])

        metrics = {
            f"{process_name}_csi": csi.to("cuda"),
            f"{process_name}_psd": psd.to("cuda"),
            f"{process_name}_crps": crps_score_local.to("cuda"),
            # f"{process_name}_eval_score": eval_score.to("cuda"),
        } | loss_dict

        return metrics
