import logging
from abc import abstractmethod
from torch.utils.data import Dataset, DataLoader
from model_compare.experiments.forecast_methods.Interface.ForecastMethodInterface import ForecastMethodInterface
import numpy as np
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps import motion


class ParametricModelMethod(ForecastMethodInterface):
    def __init__(
        self,
        dataset: Dataset,
        forecast_steps: int,
        num_prev_files: int,
        model_name: str,
        path_to_dir: str,
        sample_idx: int = 0,
    ) -> None:
        """
        model_name: Name of the model or checkpoint to use.
        path_to_dir: Directory containing the model checkpoint.
        """

        super().__init__(dataset, forecast_steps, num_prev_files, sample_idx)
        self.model_name = model_name
        self.path_to_dir = path_to_dir

    @abstractmethod
    def load_model(self):
        """
        Abstract method to load model.

        """

    @abstractmethod
    def predict(self) -> np.ndarray:
        """
        Abstract method to generate predictions.

        Returns:
            np.ndarray: Predictions in the desired format.
            output_shape: T x H x W
        """

    def generate(self) -> np.ndarray:
        logging.debug(">>> calling generate from ParametricModelMethod <<<")
        self.load_model()
        return self.predict()


class InferenceModelMethod(ForecastMethodInterface):
    def __init__(self, dataset: Dataset, forecast_steps: int, num_prev_files: int, sample_idx: int = 0) -> None:
        super().__init__(dataset, forecast_steps, num_prev_files, sample_idx)
        self.rainrate_field, self.refobs_field, self.metadata, self.data_date_frame = DataLoader(
            self.dataset, batch_size=1, shuffle=False
        ).dataset[sample_idx]

        self.rainrate_field = np.nan_to_num(self.rainrate_field, nan=0.0)
        self.refobs_field = np.nan_to_num(self.refobs_field, nan=0.0)

        self.velocity = self._calculate_velocity_field()
        self.advection = dense_lucaskanade(self.rainrate_field, verbose=True)

    def _calculate_velocity_field(self):
        oflow_method = motion.get_method("LK")
        return oflow_method(self.rainrate_field)

    @abstractmethod
    def generate(self) -> np.ndarray:
        """
        Abstract method to generate predictions.

        Returns:
            np.ndarray: Predictions in the desired format.
            output_shape: T x H x W
        """
