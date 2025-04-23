import logging
import os.path
import pickle

from torch.utils.data import DataLoader
import numpy as np

from model_compare.data_readers.datasets.NetCDFDataset import NetCDFDataset
from model_compare.entities.constants.Constants import Constants

from model_compare.entities.forecast.forecast_result_monitor import ForecastResultsMonitor
from model_compare.experiments.evaluating_methods.MetricsEvaluator import MetricsEvaluator
from model_compare.experiments.presenting_results.ForecastPlotter import ForecastPlotter
from model_compare.experiments.presenting_results.MetricsPlotter import MetricsPlotter
from model_compare.utils.methods_scanner.AbstractClassScanner import AbstractClassScanner

import model_compare.experiments.forecast_methods.Impl
from model_compare.experiments.forecast_methods.Interface.ForecastMethodInterface import ForecastMethodInterface
from model_compare.utils.plotting.ColormapConfig import ColormapConfig
from model_compare.utils.plotting.ForecastPlottingTool import PlotTool


class NowcastingExperiment:
    def __init__(
        self,
        model_path,
        storage_path,
        date_value,
        num_prev_files,
        num_next_files,
        prediction_step,
        model_base_path,
        sample_idx=0,
        incremental=False,
        base_plot_saving_path="experiment_results",
        scanner_filter=None,
        date_key=None,
        individually_plotting=False,
    ):
        self.model_path = model_path
        self.storage_path = storage_path
        self.date_value = date_value
        self.num_prev_files = num_prev_files
        self.num_next_files = num_next_files
        self.prediction_step = prediction_step
        self.model_base_path = model_base_path
        self.sample_idx = sample_idx
        self.incremental = incremental
        self.base_plot_saving_path = base_plot_saving_path
        self.date_key = date_key
        self.individually_plotting = individually_plotting

        self.date_split_name = self.date_key is not None and self.date_key or f"idx:{self.sample_idx}"

        self.forecast_results_monitor = ForecastResultsMonitor()
        self.forecast_data_path = os.path.join(self.base_plot_saving_path, Constants.FORECAST_PKL_FILE_NAME.value)

        if self.incremental and os.path.exists(self.forecast_data_path):
            self.forecast_results_monitor.read_from_file(self.forecast_data_path)

        self.dataset = NetCDFDataset(
            split="test",
            num_prev_files=num_prev_files,
            num_next_files=num_next_files,
            include_datetimes=True,
            date=self.date_value,
        )

        self.scanner = AbstractClassScanner(model_compare.experiments.forecast_methods.Impl, ForecastMethodInterface)
        self.scanner_filter = scanner_filter  # example: {'contains': ['Pysteps', 'DGMR']}

        rainrate_field, self.refobs_field, self.metadata, self.data_date_frame = DataLoader(
            self.dataset, batch_size=1, shuffle=False
        ).dataset[sample_idx]

        rainrate_field = np.nan_to_num(rainrate_field, nan=0.0)
        self.refobs_field = np.nan_to_num(self.refobs_field, nan=0.0)

        self.input_field = rainrate_field[: -self.prediction_step]

        self.metrics_evaluator = MetricsEvaluator(refobs_field=self.refobs_field)

        self.forecast_plot_data = {}
        self.forecast_metrics_data = {}

        self.crps_scores_3d = {}
        self.det_cat_scores_3d = {}
        self.crps_scores_2d = {}
        self.det_cat_scores_2d = {}
        self.all_scores_2d = {}

    def run_forecasts(self):
        """Get forecast results for each forecast method"""

        forecast_methods = self.scanner.filter_allow_then_contains(**self.scanner_filter)
        for name, cls in forecast_methods.items():
            replaced_name = name.replace(Constants.FORECAST_CLASS_NAME_SUFFIX.value, "")
            replaced_name = replaced_name.replace(Constants.PYSTEPS_METHODS_NAME_KEYWORD.value, "")
            model_name = f"{replaced_name}"
            execution_time_field_name = Constants.DECORATOR_TIMING_FILE_NAME_EXECUTIOON.value
            # load_model_time_field_name = Constants.DECORATOR_TIMING_FILE_NAME_LOAD_MODEL.value

            if cls.__bases__[0].__name__ == Constants.INFERENCE_MODEL_METHODS_KEY.value:
                instance = cls(
                    dataset=self.dataset,
                    forecast_steps=self.prediction_step,
                    num_prev_files=self.num_prev_files,
                    sample_idx=self.sample_idx,
                )
            else:
                execution_time_field_name = Constants.DECORATOR_TIMING_FILE_NAME_PREDICT.value
                model_name = f"{replaced_name}_{self.model_path}"
                instance = cls(
                    dataset=self.dataset,
                    forecast_steps=self.prediction_step,
                    num_prev_files=self.num_prev_files,
                    model_name=self.model_path,
                    path_to_dir=self.model_base_path,
                    sample_idx=self.sample_idx,
                )

            predict_name = f"{self.date_split_name}_forecast_{model_name}"

            if not self.forecast_results_monitor.exists_forecast_metric(predict_name):
                logging.debug(">>> Running forecast method: " + name + " <<<")
                forecast_metric_result = instance.get_forecast_metric_result()
                self.forecast_results_monitor.add_forecast_metric(predict_name, forecast_metric_result)
                self.forecast_results_monitor.add_forecast_time_consuming(
                    replaced_name, getattr(instance, execution_time_field_name)
                )
                # if hasattr(instance, load_model_time_field_name):
                #     self.forecast_results_monitor.add_forecast_time_consuming(
                # f"{replaced_name}{load_model_time_field_name}",
                # getattr(instance, load_model_time_field_name))

            if not self.forecast_results_monitor.exists_forecast_plot(predict_name):
                forecast_plot_data = instance.get_forecast_plot_result()
                self.forecast_results_monitor.add_forecast_plot(predict_name, forecast_plot_data, self.metadata)

            self.forecast_plot_data[replaced_name] = self.forecast_results_monitor.get_forecast_plot(predict_name)
            self.forecast_metrics_data[replaced_name] = self.forecast_results_monitor.get_forecast_metric(predict_name)

        if not self.forecast_results_monitor.exists_forecast_refobs(
            f"{self.date_split_name}_forecast_{Constants.REF_OBS_FIELD_NAME.value}"
        ):
            self.forecast_results_monitor.add_forecast_refobs(
                f"{self.date_split_name}_forecast_{Constants.REF_OBS_FIELD_NAME.value}", self.refobs_field
            )
        if not self.forecast_results_monitor.exists_forecast_input(
            f"{self.date_split_name}_forecast_{Constants.FORECAST_INPUT_FIELD_NAME.value}"
        ):
            self.forecast_results_monitor.add_forecast_input(
                f"{self.date_split_name}_forecast_{Constants.FORECAST_INPUT_FIELD_NAME.value}", self.input_field
            )
        if not self.forecast_results_monitor.exists_data_date_frame(
            f"{self.date_split_name}_forecast_{Constants.DATA_DATE_FRAME_NAME.value}"
        ):
            self.forecast_results_monitor.add_data_date_frame(
                f"{self.date_split_name}_forecast_{Constants.DATA_DATE_FRAME_NAME.value}", self.data_date_frame
            )

        self.forecast_results_monitor.save_to_file(self.forecast_data_path)

    def evaluate_forecasts(self):
        """Calculating CRPS and deterministic categorical"""
        self.crps_scores_3d = self.metrics_evaluator.get_crps_score_3d(self.forecast_metrics_data)
        self.det_cat_scores_3d = self.metrics_evaluator.get_det_cat_scores_3d(self.forecast_plot_data, threshold=0.1)

        self.crps_scores_2d = self.metrics_evaluator.get_crps_score_2d(self.forecast_metrics_data)
        self.det_cat_scores_2d = self.metrics_evaluator.get_det_cat_scores_2d(
            self.forecast_plot_data,
            threshold=0.1,
            scores=[Constants.CRPS_METRIC_NAME.value, Constants.CSI_METRIC_NAME.value, Constants.POD_METRIC_NAME.value],
        )

        # Save metrics
        metrics_save_path = os.path.join(self.storage_path, Constants.METRICS_PKL_FILE_NAME.value)
        self.all_scores_2d = {Constants.CRPS_METRIC_NAME.value: self.crps_scores_2d} | self.det_cat_scores_2d
        with open(metrics_save_path, "wb") as f:
            pickle.dump(self.all_scores_2d, f)
        logging.info(f"Saved metrics data for {self.model_path} on {self.date_split_name}")

        crps_result = {f"{self.date_split_name}": np.array(list(self.crps_scores_3d.values()))}
        det_cat_result = {
            k: {f"{self.date_split_name}": np.array(list(inner_dict.values()))}
            for k, inner_dict in self.det_cat_scores_3d.items()
        }

        return crps_result, det_cat_result

    def visualize(self):
        """Plotting the forecasts and metrics"""
        forecast_plotter = ForecastPlotter(
            plotter=PlotTool(
                cmap_config=ColormapConfig(),
                individually_plotting=self.individually_plotting,
                time_interval=5,
                independent_metrics=self.all_scores_2d,
            ),
            prediction_step=self.prediction_step,
            input_field=self.input_field,
            forecast=self.forecast_plot_data,
            refobs_field=self.refobs_field,
            metadata=self.metadata,
            data_date_frame=self.data_date_frame,
            title_name=f"Forecast_Plotting_{self.date_split_name}",
            storage_path=self.storage_path,
        )
        forecast_plotter.plot_forecast()

        metrics_plotter = MetricsPlotter(
            refobs_field=self.refobs_field,
            prediction_step=self.prediction_step,
            model_path=self.model_path,
            storage_path=self.storage_path,
            data_date_frame=self.data_date_frame,
            detcat_metric_score_dict=self.det_cat_scores_3d,
            crps_scores=self.crps_scores_3d,
        )
        metrics_plotter.plot_and_record_verification_metrics(self.forecast_plot_data)
        metrics_plotter.plot_scores()
