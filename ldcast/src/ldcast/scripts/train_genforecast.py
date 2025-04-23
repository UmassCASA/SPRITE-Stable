import gc

from fire import Fire
import torch
from omegaconf import OmegaConf

from ldcast.models.autoenc import autoenc
from ldcast.models.autoenc import encoder
from ldcast.models.genforecast import training
from ldcast.models.genforecast import analysis, unet

from train_nowcaster import setup_data_casa


def setup_model(
    config,
    num_timesteps=5,
):
    enc = encoder.SimpleConvEncoder()
    dec = encoder.SimpleConvDecoder()
    autoencoder_obs = autoenc.AutoencoderKL(enc, dec)

    checkpoint = torch.load(config.autoenc_weights_fn)
    if "state_dict" in checkpoint:
        autoencoder_obs.load_state_dict(checkpoint["state_dict"])
    else:
        autoencoder_obs.load_state_dict(checkpoint)

    autoencoders = []
    input_patches = []
    input_size_ratios = []
    embed_dim = []
    analysis_depth = []
    if config.use_obs:
        autoencoders.append(autoencoder_obs)
        input_patches.append(1)
        input_size_ratios.append(1)
        embed_dim.append(128)
        analysis_depth.append(4)
    if config.use_nwp:
        autoencoder_nwp = autoenc.DummyAutoencoder(width=config.num_nwp_vars)
        autoencoders.append(autoencoder_nwp)
        input_patches.append(config.nwp_input_patches)
        input_size_ratios.append(2)
        embed_dim.append(32)
        analysis_depth.append(2)

    analysis_net = analysis.AFNONowcastNetCascade(
        autoencoders,
        input_patches=input_patches,
        input_size_ratios=input_size_ratios,
        train_autoenc=False,
        output_patches=num_timesteps,
        cascade_depth=3,
        embed_dim=embed_dim,
        analysis_depth=analysis_depth,
    )

    model = unet.UNetModel(
        in_channels=autoencoder_obs.hidden_width,
        model_channels=256,
        out_channels=autoencoder_obs.hidden_width,
        num_res_blocks=2,
        attention_resolutions=(1, 2),
        dims=3,
        channel_mult=(1, 2, 4),
        num_heads=8,
        num_timesteps=num_timesteps,
        context_ch=analysis_net.cascade_dims,
    )

    (ldm, trainer) = training.setup_genforecast_training(
        model,
        autoencoder_obs,
        config=config,
        context_encoder=analysis_net,
    )
    gc.collect()
    return (ldm, trainer)


def train(config):
    print("Loading data...")
    datamodule = setup_data_casa(for_ae=False, batch_size=config.batch_size)

    print("Setting up model...")
    (model, trainer) = setup_model(
        num_timesteps=config.future_timesteps // 4,
        config=config,
    )
    if config.initial_weights is not None:
        print(f"Loading weights from {config.initial_weights}...")
        model.load_state_dict(
            torch.load(config.initial_weights, map_location=model.device), strict=config.strict_weights
        )

    print("Starting training...")
    trainer.fit(model, datamodule=datamodule, ckpt_path=config.checkpoint_path)


def main(config=None, **kwargs):
    config = OmegaConf.load(config) if (config is not None) else {}
    config = OmegaConf.merge(config, kwargs)
    train(config)


if __name__ == "__main__":
    Fire(main)
