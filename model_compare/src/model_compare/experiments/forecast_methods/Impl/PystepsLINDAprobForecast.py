from model_compare.experiments.forecast_methods.Interface.ForecastMethodBase import InferenceModelMethod
from pysteps.nowcasts import linda
from pysteps.utils import transformation
import numpy as np


class PystepsLINDAprobForecast(InferenceModelMethod):
    def __init__(self, dataset, forecast_steps, num_prev_files, sample_idx=0):
        super().__init__(dataset, forecast_steps, num_prev_files, sample_idx)
        self.is_probabilistic = True

    def generate(self, val_pert_method=None):
        rainrate_field = np.nan_to_num(self.rainrate_field, nan=0.1, posinf=0.1, neginf=0.1)
        rainrate_field[rainrate_field == 0.0] = 0.1

        rainrate_field_db, _ = transformation.dB_transform(
            rainrate_field, self.metadata, threshold=0.1, zerovalue=-15.0
        )
        forecast_linda = linda.forecast(
            rainrate_field,
            self.advection,
            self.forecast_steps,
            max_num_features=15,
            add_perturbations=True,
            vel_pert_method=val_pert_method,  # No perturbation of the velocity field by default
            n_ens_members=40,
            num_workers=8,
            measure_time=True,
        )[0]

        return forecast_linda
