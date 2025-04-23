from model_compare.experiments.forecast_methods.Interface.ForecastMethodBase import ParametricModelMethod
from model_compare.experiments.results_providers.Impl.unet_predicting import UNetPredictor


class UNetDSForecast(ParametricModelMethod):
    def __init__(self, dataset, forecast_steps, num_prev_files, model_name, path_to_dir, sample_idx=0):
        super().__init__(dataset, forecast_steps, num_prev_files, model_name, path_to_dir, sample_idx)

        self.unet_pre = UNetPredictor(
            self.dataset, self.model_name, self.path_to_dir, self.forecast_steps, self.num_prev_files, sample_idx
        )

    def load_model(self):
        return self.unet_pre.load_model()

    def predict(self):
        return self.unet_pre.get_prediction()
