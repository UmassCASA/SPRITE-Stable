from model_compare.experiments.forecast_methods.Interface.ForecastMethodBase import InferenceModelMethod
from pysteps.nowcasts import anvil
from pysteps.utils import transformation
import numpy as np


class PystepsANVILForecast(InferenceModelMethod):
    def __init__(self, dataset, forecast_steps, num_prev_files, sample_idx=0):
        super().__init__(dataset, forecast_steps, num_prev_files, sample_idx)

    def generate(self) -> np.ndarray:
        rainrate_field_db, _ = transformation.dB_transform(
            self.rainrate_field, self.metadata, threshold=0.1, zerovalue=-15.0
        )
        forecast_anvil = anvil.forecast(
            rainrate_field_db[: -self.forecast_steps],
            self.velocity,
            self.forecast_steps,
            ar_window_radius=25,
            ar_order=self.num_prev_files - 1,
        )
        # forecast_anvil, _ = transformation.dB_transform(forecast_anvil, threshold=-10.0, inverse=True)
        forecast_anvil[forecast_anvil < 0.5] = 0.0
        return forecast_anvil
