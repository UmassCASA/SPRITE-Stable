from model_compare.experiments.forecast_methods.Interface.ForecastMethodBase import InferenceModelMethod
from pysteps.nowcasts import sprog
from pysteps.utils import transformation
import numpy as np


class PystepsSPROGForecast(InferenceModelMethod):
    def __init__(self, dataset, forecast_steps, num_prev_files, sample_idx=0):
        super().__init__(dataset, forecast_steps, num_prev_files, sample_idx)

    def generate(self):
        rainrate_field_db, _ = transformation.dB_transform(
            self.rainrate_field, self.metadata, threshold=0.1, zerovalue=-15.0
        )
        rainrate_thr, _ = transformation.dB_transform(np.array([0.5]), self.metadata, threshold=0.1, zerovalue=-15.0)

        rainrate_field_db[~np.isfinite(rainrate_field_db)] = -15.0
        rainrate_field_db = np.nan_to_num(rainrate_field_db, nan=-15.0).astype(np.float64)

        forecast_sprog = sprog.forecast(
            rainrate_field_db[: -self.forecast_steps],
            self.velocity,
            self.forecast_steps,
            n_cascade_levels=6,
            R_thr=rainrate_thr[0],
        )
        forecast_sprog, _ = transformation.dB_transform(forecast_sprog, threshold=-10.0, inverse=True)
        forecast_sprog[forecast_sprog < 0.5] = 0.0
        return forecast_sprog
