import datetime
import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
import pickle

import numpy as np

from src.model_compare.entities.constants.Constants import Constants


@dataclass
class ForecastResultsMonitor:
    forecasts: Dict[str, Dict[str, Tuple[np.ndarray, Dict]]] = field(default_factory=dict)
    data_date_frame: Dict[str, List[datetime.datetime]] = field(default_factory=dict)
    forecast_time_consuming: Dict[str, List[float]] = field(default_factory=dict)

    def __post_init__(self):
        self.forecasts[Constants.FORECAST_DICT_PLOT_KEY.value] = {}
        self.forecasts[Constants.FORECAST_DICT_METRIC_KEY.value] = {}
        self.forecasts[Constants.REF_OBS_FIELD_NAME.value] = {}
        self.forecasts[Constants.FORECAST_INPUT_FIELD_NAME.value] = {}

    def read_from_file(self, forecast_data_path: str):
        with open(forecast_data_path, "rb") as f:
            loaded_data = pickle.load(f)

        self.__dict__.update(loaded_data)

        logging.info(f"Loaded existing forecast data: {forecast_data_path}")

    def save_to_file(self, forecast_data_path: str):
        with open(forecast_data_path, "wb") as f:
            pickle.dump(self.__dict__, f)
        logging.info(f"Saved forecast data to: {forecast_data_path}")

    def add_forecast_plot(self, forecast_name: str, forecast: np.ndarray, metadata: Dict = None):
        self.forecasts[Constants.FORECAST_DICT_PLOT_KEY.value][forecast_name] = (forecast, metadata)

    def add_forecast_metric(self, forecast_name: str, forecast: np.ndarray, metadata: Dict = None):
        self.forecasts[Constants.FORECAST_DICT_METRIC_KEY.value][forecast_name] = (forecast, metadata)

    def add_forecast_refobs(self, forecast_name: str, forecast: np.ndarray, metadata: Dict = None):
        self.forecasts[Constants.REF_OBS_FIELD_NAME.value][forecast_name] = (forecast, metadata)

    def add_forecast_input(self, forecast_name: str, forecast: np.ndarray, metadata: Dict = None):
        self.forecasts[Constants.FORECAST_INPUT_FIELD_NAME.value][forecast_name] = (forecast, metadata)

    def add_data_date_frame(self, forecast_name: str, date_frame: List[datetime.datetime]):
        self.data_date_frame[forecast_name] = date_frame

    def add_forecast_time_consuming(self, forecast_name: str, time_consuming: float):
        if self.exists_forecast_time_consuming(forecast_name):
            self.forecast_time_consuming[forecast_name].append(time_consuming)
        else:
            self.forecast_time_consuming[forecast_name] = [time_consuming]

    def get_forecast_plot(self, forecast_name: str, need_metadata: bool = False):
        if need_metadata:
            return self.forecasts[Constants.FORECAST_DICT_PLOT_KEY.value][forecast_name]
        else:
            return self.forecasts[Constants.FORECAST_DICT_PLOT_KEY.value][forecast_name][0]

    def get_forecast_metric(self, forecast_name: str, need_metadata: bool = False):
        if need_metadata:
            return self.forecasts[Constants.FORECAST_DICT_METRIC_KEY.value][forecast_name]
        else:
            return self.forecasts[Constants.FORECAST_DICT_METRIC_KEY.value][forecast_name][0]

    def get_forecast_refobs(self, forecast_name: str, need_metadata: bool = False):
        if need_metadata:
            return self.forecasts[Constants.REF_OBS_FIELD_NAME.value][forecast_name]
        else:
            return self.forecasts[Constants.REF_OBS_FIELD_NAME.value][forecast_name][0]

    def get_forecast_input(self, forecast_name: str, need_metadata: bool = False):
        if need_metadata:
            return self.forecasts[Constants.FORECAST_INPUT_FIELD_NAME.value][forecast_name]
        else:
            return self.forecasts[Constants.FORECAST_INPUT_FIELD_NAME.value][forecast_name][0]

    def get_data_date_frame(self, forecast_name: str):
        return self.data_date_frame[forecast_name]

    def get_forecast_time_consuming(self, forecast_name: str):
        return self.forecast_time_consuming[forecast_name]

    def exists_forecast_plot(self, forecast_name: str):
        return forecast_name in self.forecasts[Constants.FORECAST_DICT_PLOT_KEY.value]

    def exists_forecast_metric(self, forecast_name: str):
        return forecast_name in self.forecasts[Constants.FORECAST_DICT_METRIC_KEY.value]

    def exists_forecast_refobs(self, forecast_name: str):
        return forecast_name in self.forecasts[Constants.REF_OBS_FIELD_NAME.value]

    def exists_forecast_input(self, forecast_name: str):
        return forecast_name in self.forecasts[Constants.FORECAST_INPUT_FIELD_NAME.value]

    def exists_data_date_frame(self, forecast_name: str):
        return forecast_name in self.data_date_frame

    def exists_forecast_time_consuming(self, forecast_name: str):
        return forecast_name in self.forecast_time_consuming

    def remove_forecast_plot(self, forecast_name: str):
        return self.forecasts[Constants.FORECAST_DICT_PLOT_KEY.value].pop(forecast_name, None)

    def remove_forecast_metric(self, forecast_name: str):
        return self.forecasts[Constants.FORECAST_DICT_METRIC_KEY.value].pop(forecast_name, None)

    def remove_forecast_refobs(self, forecast_name: str):
        return self.forecasts[Constants.REF_OBS_FIELD_NAME.value].pop(forecast_name, None)

    def remove_forecast_input(self, forecast_name: str):
        return self.forecasts[Constants.FORECAST_INPUT_FIELD_NAME.value].pop(forecast_name, None)

    def remove_data_date_frame(self, forecast_name: str):
        return self.data_date_frame.pop(forecast_name, None)

    def remove_forecast_time_consuming(self, forecast_name: str):
        return self.forecast_time_consuming.pop(forecast_name, None)

    def print_shape(self):
        for k, v in self.forecasts.items():
            logging.info(f"{k}: {v.shape}")

    def assemble_time_consumed_dict(self):
        time_consumed_dict = {}
        for k, v in self.forecast_time_consuming.items():
            time_consumed_dict[k] = np.mean(v)
        return time_consumed_dict
