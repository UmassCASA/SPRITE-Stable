from model_compare.experiments.forecast_methods.Interface.ForecastMethodBase import InferenceModelMethod
from pysteps.nowcasts import linda
from pysteps.utils import transformation
import numpy as np


class PystepsLINDAdeterForecast(InferenceModelMethod):
    def __init__(self, dataset, forecast_steps, num_prev_files, sample_idx=0):
        super().__init__(dataset, forecast_steps, num_prev_files, sample_idx)

    def generate(self) -> np.ndarray:
        rainrate_field_db, _ = transformation.dB_transform(
            self.rainrate_field, self.metadata, threshold=0.1, zerovalue=-15.0
        )
        forecast_linda = linda.forecast(
            rainrate_field_db[: -self.forecast_steps],
            self.advection,
            self.forecast_steps,
            max_num_features=15,
            add_perturbations=False,
            num_workers=8,
            measure_time=True,
        )[0]
        return forecast_linda
