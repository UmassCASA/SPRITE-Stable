from model_compare.experiments.forecast_methods.Interface.ForecastMethodBase import InferenceModelMethod
from pysteps.nowcasts import extrapolation


class PystepsExtrapForecast(InferenceModelMethod):
    def __init__(self, dataset, forecast_steps, num_prev_files, sample_idx=0):
        super().__init__(dataset, forecast_steps, num_prev_files, sample_idx)

    def generate(self):
        forecast_extrap = extrapolation.forecast(
            self.rainrate_field[-self.forecast_steps],
            self.velocity,
            self.forecast_steps,
            extrap_kwargs={"allow_nonfinite_values": True},
        )
        forecast_extrap[forecast_extrap < 0.5] = 0.0
        return forecast_extrap
