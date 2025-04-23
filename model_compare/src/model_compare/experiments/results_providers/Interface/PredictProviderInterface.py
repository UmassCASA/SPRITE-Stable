from abc import ABC, abstractmethod
import numpy as np


class PredictProviderInterface(ABC):
    def __init__(
        self,
        dataset,
        model_name: str,
        path_to_dir: str,
        forecast_steps: int,
        num_prev_files: int,
        sample_idx: int = 0,
        model=None,
    ) -> None:
        """
        Abstract initializer for predictors.

        Args:
            dataset: The dataset to use for prediction.
            model_name: Name of the model or checkpoint to use.
            path_to_dir: Directory containing the model checkpoint.
            forecast_steps: Number of forecast steps for prediction.
            num_prev_files: Number of previous files to use as input.
            sample_idx: Index of the sample to predict.
        """
        self.dataset = dataset
        self.model_name = model_name
        self.path_to_dir = path_to_dir
        self.forecast_steps = forecast_steps
        self.num_prev_files = num_prev_files
        self.sample_idx = sample_idx
        self.model = model

    @abstractmethod
    def load_model(self):
        """
        Abstract method to load model.

        """

    @abstractmethod
    def get_prediction(self) -> np.ndarray:
        """
        Abstract method to generate predictions.

        Returns:
            np.ndarray: Predictions in the desired format.
            output_shape: T x H x W
        """
