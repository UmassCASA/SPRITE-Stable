import logging
import os
import pickle
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from src.model_compare.experiments.nowcasting_experiment import NowcastingExperiment
from model_compare.entities.constants.Constants import Constants
from model_compare.utils.file_operators.SymbolicLinkCreator import SymbolicLinkCreator
from model_compare.utils.methods_scanner.AbstractClassScanner import AbstractClassScanner
from model_compare.utils.plotting.plot_bar import plot_bars_dict
from model_compare.entities.config.model_compare_experiment import RunExperimentConfig

import model_compare.experiments.forecast_methods.Impl
from model_compare.experiments.forecast_methods.Interface.ForecastMethodInterface import ForecastMethodInterface

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

config = RunExperimentConfig.from_yaml("run_experiment.yaml")


def process_experiment(model_path, model_base_path, date_key, date_value, incremental=False, scanner_filter=None):
    logging.info(f"Processing model {model_path} for date {date_key}...")
    storage_dir = os.path.join(config.BASE_RESULTS_DIR, model_path, date_key)
    os.makedirs(storage_dir, exist_ok=True)

    date_time_value = isinstance(date_value, datetime) and date_value or None
    sample_idx = isinstance(date_value, int) and date_value or 0

    experiment = NowcastingExperiment(
        model_path=model_path,
        storage_path=storage_dir,
        date_value=date_time_value,
        num_prev_files=3,
        num_next_files=10,
        prediction_step=10,
        model_base_path=model_base_path,
        sample_idx=sample_idx,
        incremental=incremental,
        base_plot_saving_path=config.BASE_RESULTS_DIR,
        scanner_filter=scanner_filter,
        date_key=date_key,
        individually_plotting=config.individually_plotting,
    )

    # run forecasting
    experiment.run_forecasts()
    # calculating average metrics
    crps_scores, det_cat_scores = experiment.evaluate_forecasts()
    # plotting forecasting + metrics + verifications
    experiment.visualize()

    return crps_scores, det_cat_scores


def start_running(model_name, model_base_path, dates_dict, incremental=False, scanner_filter=None):
    """
    Starts the nowcasting experiments for a specific model.

    Parameters:
    - model_name (str): The name of the model.
    - model_base_path(str): the base path of the model.
    - dates_dict (dict): Dict of date KV to process.
    - incremental (bool): Whether to run in incremental mode.
    - scanner_filter (dict): indicate scanner filter mode and key word list.
    """

    crps_dict = {}
    detcat_dict = {}
    models = []

    scanner = AbstractClassScanner(model_compare.experiments.forecast_methods.Impl, ForecastMethodInterface)
    scanner_filter = scanner_filter  # example: {'contains': ['Pysteps', 'DGMR']}
    forecast_methods = scanner.filter_allow_then_contains(**scanner_filter)
    for name, _cls in forecast_methods.items():
        replaced_name = name.replace("Forecast", "")
        models.append(replaced_name)

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {}
        for date_key, date_value in dates_dict.items():
            futures[
                executor.submit(
                    process_experiment, model_name, model_base_path, date_key, date_value, incremental, scanner_filter
                )
            ] = (model_name, date_key)
        for future in as_completed(futures):
            model, date_key = futures[future]
            try:
                crps, det_cat = future.result()
                crps_dict.update(crps)
                for metric_name, inner_dict in det_cat.items():
                    detcat_dict.setdefault(metric_name, {}).update(inner_dict)
            except Exception as e:
                logging.exception(f"Error processing model {model} date {date_key}: {e}")

    adjusted_data = crps_dict

    all_average_dict = {}
    # Extract and average data by models
    for metric_name, scores in {**detcat_dict, "CRPS": adjusted_data}.items():
        model_averages = {}
        for i, model in enumerate(models):
            model_scores = [data[i] for data in scores.values()]
            model_averages[model] = np.mean(model_scores)
        all_average_dict[metric_name] = model_averages
        plot_bars_dict(
            scores=model_averages,
            metric_name=metric_name,
            save_path=os.path.join(config.BASE_RESULTS_DIR, model_name, f"{metric_name}_average.png"),
            title_name="Model Averages by Calculation Method",
        )
        logging.info(f"Model {model_name} {metric_name} averages: {model_averages}")

    # Save the metrics to a pickle file
    metrics_path = os.path.join(config.BASE_RESULTS_DIR, model_name, Constants.METRICS_AVERAGE_PKL_FILE_NAME.value)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "wb") as dict_file:
        pickle.dump(all_average_dict, dict_file)
    logging.info(f"Saved average metrics for {model_name}")


def main():
    dates_dict = {}
    if config.have_dates():
        dates_dict = config.dates
        for key, value in dates_dict.items():
            if isinstance(value, str) and re.match(r"^\d{8}_\d{4}$", value):
                dates_dict[key] = datetime.strptime(value, "%Y%m%d_%H%M")

    models_to_process = []

    model_base_path_list = config.MODEL_BASE_PATH

    toplevels_list = {}
    for path in model_base_path_list:
        candidate_model_path = path
        if candidate_model_path not in toplevels_list:
            symbolic_link_creator = SymbolicLinkCreator(
                root_dir=os.path.abspath(candidate_model_path),
                target_suffix=".ckpt",
                num_links=config.experiment_repeat_time,
            )
            toplevels_list.update({candidate_model_path: symbolic_link_creator})
            symbolic_link_creator.delete_all_symlinks()
            symbolic_link_creator.scan_and_create_links()

        methods_filter_list = config.PYSTEPS_METHODS_LIST.copy()
        methods_filter_list.append(os.path.basename(path).split("_")[0])
        for _root, _dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".ckpt"):
                    models_to_process.append((file, methods_filter_list, path))

    logging.info(
        f"Running experiment with incremental mode: {config.incremental},"
        f"models: \n{models_to_process}\n,"
        f"dates: \n{dates_dict}"
    )

    for model_name_method_pair in models_to_process:
        model_name, methods_filter_list, model_base_path = model_name_method_pair
        start_running(
            model_name,
            model_base_path,
            dates_dict,
            config.incremental,
            scanner_filter={"class_name_list": methods_filter_list, "suffix_list": config.ABS_SCANNER_SUFFIX_LIST},
        )

    for _path, symbolic_link_creator in toplevels_list.items():
        symbolic_link_creator.delete_all_symlinks()


if __name__ == "__main__":
    main()
