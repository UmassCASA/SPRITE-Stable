import logging
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import numpy as np

from model_compare.entities.constants.Constants import Constants
from model_compare.utils.decorators.Timing_decorator import timing_decorator


class ForecastMethodInterface(ABC):
    def __init__(
        self,
        dataset: Dataset,
        forecast_steps: int,
        num_prev_files: int,
        sample_idx: int = 0,
        is_probabilistic: bool = False,
    ) -> None:
        """
        Abstract initializer for predictors.

        Args:
            dataset: The dataset to use for prediction.
            forecast_steps: Number of forecast steps for prediction.
            num_prev_files: Number of previous files to use as input.
            sample_idx: Index of the sample to predict.
            is_probabilistic: indicator if the forecast method is probabilistic
        """
        self.dataset = dataset
        self.forecast_steps = forecast_steps
        self.num_prev_files = num_prev_files
        self.sample_idx = sample_idx
        self.is_probabilistic = is_probabilistic
        self.forecast_result = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for method_name, attr_name in Constants.TIMING_CONFIG.value.items():
            if method_name in cls.__dict__:
                original = cls.__dict__[method_name]
                decorated = timing_decorator(attr_name)(original)
                setattr(cls, method_name, decorated)

    def method_is_probabilistic(self):
        return self.is_probabilistic

    def get_forecast_result(self):
        logging.debug(">>> calling generate from ForecastMethodInterface <<<")
        if self.forecast_result is None:
            self.forecast_result = self.generate()
        return self.forecast_result

    def get_forecast_plot_result(self):
        if self.is_probabilistic:
            return np.median(self.get_forecast_result(), axis=0)
        return self.get_forecast_result()

    def get_forecast_metric_result(self):
        if not self.is_probabilistic:
            return self.get_forecast_result()[np.newaxis, :, :, :]
        return self.get_forecast_result()

    @abstractmethod
    def generate(self) -> np.ndarray:
        """
        Abstract method to generate predictions.

        Returns:
            np.ndarray: Predictions in the desired format.
            output_shape: T x H x W
        """
