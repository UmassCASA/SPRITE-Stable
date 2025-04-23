from datetime import datetime, timedelta

from fire import Fire
from matplotlib import pyplot as plt
import numpy as np

from ldcast import forecast
from ldcast.visualization import plots

from model_compare.data_readers.datasets.NetCDFDataset import NetCDFDataset
from torch.utils.data import DataLoader


def read_data(
    data_dir="../data/demo/20210622",
    t0=datetime(2021, 6, 22, 18, 35),
    interval=timedelta(minutes=5),
    past_timesteps=4,
    crop_box=((128, 480), (160, 608)),
):
    # cb = crop_box
    # R_past = []
    # t = t0 - (past_timesteps-1) * interval
    # for i in range(past_timesteps):
    #     timestamp = t.strftime("%y%j%H%M")
    #     fn = f"RZC{timestamp}VL.[80]01.h5"
    #     fn = os.path.join(data_dir, fn)
    #     found_files = glob.glob(fn)
    #     if found_files:
    #         fn = found_files[0]
    #     else:
    #         raise FileNotFoundError(f"Unable to find data file {fn}.")
    #     with h5py.File(fn, 'r') as f:
    #         R = f["dataset1"]["data1"]["data"][:]
    #     R = R[cb[0][0]:cb[0][1], cb[1][0]:cb[1][1]]
    #     R_past.append(R)
    #     t += interval
    #
    # R_past = np.stack(R_past, axis=0)
    # return R_past
    dataset = NetCDFDataset(
        split="test",
        num_prev_files=3,
        num_next_files=20,
        include_datetimes=True,
    )

    rainrate_field, refobs_field, metadata, data_date_frame = DataLoader(dataset, batch_size=1, shuffle=False).dataset[
        160
    ]

    return rainrate_field[:past_timesteps]


def plot_border(ax, crop_box=((128, 480), (160, 608))):
    import shapefile

    border = shapefile.Reader("../data/Border_CH.shp")
    shapes = list(border.shapeRecords())
    for shape in shapes:
        x = np.array([i[0] / 1000.0 for i in shape.shape.points[:]])
        y = np.array([i[1] / 1000.0 for i in shape.shape.points[:]])
        ax.plot(x - crop_box[1][0] - 255, 480 - y - crop_box[0][0], "k", linewidth=1.0)


def plot_frame(R, fn, draw_border=True, t=None, label=None):
    fig = plt.figure(dpi=150)
    ax = fig.add_subplot()
    plots.plot_precip_image(ax, R)
    if draw_border:
        plot_border(ax)
    if t is not None:
        timestamp = "%Y-%m-%d %H:%M UTC"
        if label is not None:
            timestamp += f" ({label})"
        ax.text(
            0.02,
            0.98,
            t.strftime(timestamp),
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
        )

    fig.savefig(fn, bbox_inches="tight")
    plt.close(fig)


def forecast_demo(
    ldm_weights_fn="/work/pi_mzink_umass_edu/SPRITE/outputs/ldcast/diffusion/VTPM/last.ckpt",
    # ldm_weights_fn="/home/zhexu_umass_edu/PycharmProjects/ldcast/models/"
    #                "genforecast/genforecast-radaronly-256x256-20step.pt",
    # autoenc_weights_fn="/work/pi_mzink_umass_edu/SPRITE/outputs/ldcast/autoenc/epoch=841-val_rec_loss=0.3662.ckpt",
    autoenc_weights_fn="/home/zhexu_umass_edu/PycharmProjects/ldcast/models/autoenc/autoenc-32-0.01.pt",
    num_diffusion_iters=50,
    out_dir="../figures/demo/",
    data_dir="../data/demo/20210622",
    t0=datetime(2021, 6, 22, 18, 35),
    interval=timedelta(minutes=5),
    past_timesteps=4,
    crop_box=((128, 480), (160, 608)),
    draw_border=True,
    ensemble_members=1,
):
    R_past = read_data(data_dir=data_dir, t0=t0, interval=interval, past_timesteps=past_timesteps, crop_box=crop_box)
    if ensemble_members == 1:
        fc = forecast.Forecast(
            ldm_weights_fn=ldm_weights_fn, autoenc_weights_fn=autoenc_weights_fn, log_R_std=6.405, log_R_mean=1.643
        )
        R_pred = fc(R_past, num_diffusion_iters=num_diffusion_iters)
    elif ensemble_members > 1:
        fc = forecast.ForecastDistributed(
            ldm_weights_fn=ldm_weights_fn,
            autoenc_weights_fn=autoenc_weights_fn,
        )
        R_past = R_past.reshape((1,) + R_past.shape)
        R_pred = fc(R_past, num_diffusion_iters=num_diffusion_iters, ensemble_members=ensemble_members)
        R_past = R_past[0, ...]
        R_pred = R_pred[0, ...].mean(axis=-1)  # compute ensemble mean
    else:
        raise ValueError("ensemble_members must be > 0")

    T, H, W = R_pred.shape

    for t in range(T):
        plt.figure()
        plt.imshow(R_pred[t], cmap="jet", origin="lower")
        plt.title(f"precipitation_filed {t + 1}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.colorbar(label="intensity")
        plt.tight_layout()
        plt.show()

        # plt.figure()
        # plt.imshow(R_past[t], cmap='jet', origin='lower')
        # plt.title(f"precipitation_filed {t + 1}")
        # plt.xlabel("X ")
        # plt.ylabel("Y ")
        # plt.colorbar(label='intensity')
        # plt.tight_layout()
        # plt.show()

    # os.makedirs(out_dir, exist_ok=True)
    # for k in range(R_past.shape[0]):
    #     fn = os.path.join(out_dir, f"R_past-{k:02d}.png")
    #     t = t0 - (R_past.shape[0]-k-1) * interval
    #     plot_frame(R_past[k,:,:], fn, draw_border=draw_border,
    #         t=t, label="Real")
    # for k in range(R_pred.shape[0]):
    #     fn = os.path.join(out_dir, f"R_pred-{k:02d}.png")
    #     t = t0 + (k+1)*interval
    #     plot_frame(R_pred[k,:,:], fn, draw_border=draw_border,
    #         t=t, label="Predicted")


if __name__ == "__main__":
    Fire(forecast_demo)
