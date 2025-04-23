from model_compare.experiments.forecast_methods.Interface.ForecastMethodBase import InferenceModelMethod
from pysteps.nowcasts import steps
from pysteps.utils import transformation


class PystepsSTEPSForecast(InferenceModelMethod):
    def __init__(self, dataset, forecast_steps, num_prev_files, sample_idx=0, seed=24):
        super().__init__(dataset, forecast_steps, num_prev_files, sample_idx)
        self.seed = seed
        self.is_probabilistic = True

    def generate(self):
        rainrate_field_db, _ = transformation.dB_transform(
            self.rainrate_field, self.metadata, threshold=0.1, zerovalue=-15.0
        )

        forecast_steps_result = steps.forecast(
            rainrate_field_db[: -self.forecast_steps],
            self.velocity,
            self.forecast_steps,
            20,
            n_cascade_levels=6,
            R_thr=-10.0,
            kmperpixel=2.0,
            timestep=5,
            noise_method="nonparametric",
            vel_pert_method="bps",
            mask_method="incremental",
            seed=self.seed,
        )
        forecast_steps_result = transformation.dB_transform(forecast_steps_result, threshold=-10.0, inverse=True)[0]
        return forecast_steps_result
