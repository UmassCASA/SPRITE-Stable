import os
import pickle
from collections import defaultdict

from src.model_compare.entities.config.model_compare_experiment import RunExperimentConfig
from src.model_compare.entities.constants.Constants import Constants
from src.model_compare.entities.forecast.forecast_result_monitor import ForecastResultsMonitor
from src.model_compare.experiments.model_comparer import MetricModelComparer
from src.model_compare.experiments.presenting_results.ForecastPlotter import ForecastPlotter
from src.model_compare.utils.plotting.ColormapConfig import ColormapConfig
from src.model_compare.utils.plotting.ForecastPlottingTool import PlotTool
from src.model_compare.utils.plotting.plot_bar import plot_bars_dict

config = RunExperimentConfig.from_yaml("run_experiment.yaml")


def collect_method_scores(
    root_dir, model_name_filter=None, date_filter=None, pkl_file_name=Constants.METRICS_AVERAGE_PKL_FILE_NAME.value
):
    """
    Walk through root_dir, collect 'DGMR' scores from 'metrics_average.pkl' files.

    Parameters:
    - root_dir (str): The root directory to search.

    Returns:
    - forecast_scores (dict): A nested dictionary of the form {metric: {subdir_name: score}}
    """
    forecast_scores = defaultdict(dict)

    for subdir, _, files in os.walk(root_dir):
        if pkl_file_name in files:
            file_path = os.path.join(subdir, pkl_file_name)
            subdir_name = os.path.basename(subdir)
            if date_filter is not None and date_filter not in subdir_name:
                continue

            # Read the pickle file
            with open(file_path, "rb") as f:
                metrics_data = pickle.load(f)

            for metric, methods_scores in metrics_data.items():
                if metric not in forecast_scores:
                    forecast_scores[metric] = defaultdict(dict)
                if model_name_filter is not None and model_name_filter in methods_scores:
                    forecast_scores[metric][subdir_name] = methods_scores[model_name_filter]
                    continue
                else:
                    for methods_name, score in methods_scores.items():
                        forecast_scores[metric][methods_name] = score

    return forecast_scores


def adjust_scores(forecast_scores):
    """
    Adjust 'HK' and 'BIAS' scores to make them suitable for comparison.

    For 'HK' and 'BIAS', the closer to 1, the better, so we take the absolute difference from 1.
    """
    if "HK" in forecast_scores:
        forecast_scores["HK"] = {k: abs(v - 1) for k, v in forecast_scores["HK"].items()}

    if "BIAS" in forecast_scores:
        forecast_scores["BIAS"] = {k: abs(v - 1) for k, v in forecast_scores["BIAS"].items()}


def find_best_scores(method_scores, larger_better_metrics, lower_better_metrics):
    """
    Find the best (max or min) score and associated subdir_name for each metric.

    Parameters:
    - method_scores (dict): Nested dictionary of scores.
    - larger_better_metrics (list): List of metrics where a higher score is better.
    - lower_better_metrics (list): List of metrics where a lower score is better.

    Returns:
    - best_scores (dict): Dictionary with the best score and subdir for each metric.
    """
    best_scores = defaultdict(dict)

    for metric in method_scores:
        metric_scores = method_scores[metric]
        if metric in larger_better_metrics:
            # Find max score
            best_subdir, best_score = max(metric_scores.items(), key=lambda x: x[1])
        elif metric in lower_better_metrics:
            # Find min score
            best_subdir, best_score = min(metric_scores.items(), key=lambda x: x[1])
        else:
            print(f"Metric '{metric}' not in larger_better_metrics or lower_better_metrics. Skipping.")
            continue  # Skip metrics not in either list

        best_scores[metric] = {"subdir": best_subdir, "score": best_score}

    return best_scores


def plot_ml_models_horizontal_compare():
    forecast_results_monitor = ForecastResultsMonitor()
    forecast_data_path = os.path.join(config.BASE_RESULTS_DIR, Constants.FORECAST_PKL_FILE_NAME.value)

    if os.path.exists(forecast_data_path):
        forecast_results_monitor.read_from_file(forecast_data_path)

    plotting_data_with_metadata = defaultdict(dict)
    for predict_name in forecast_results_monitor.forecasts[Constants.FORECAST_DICT_PLOT_KEY.value].keys():
        split_name_list = predict_name.split("_")
        date_key = split_name_list[0]
        model_name = split_name_list[2]
        plotting_data_with_metadata[date_key][model_name] = forecast_results_monitor.get_forecast_plot(
            predict_name, True
        )

    for date_key, value in plotting_data_with_metadata.items():
        plotting_data = defaultdict(dict)
        metadata = {}
        for model_name, forecast_plot in value.items():
            plotting_data[model_name], metadata = forecast_plot
        unique_model_keys = list(set(value.keys()))

        metrics_data = collect_method_scores(
            root_dir=config.BASE_RESULTS_DIR, pkl_file_name=Constants.METRICS_PKL_FILE_NAME.value, date_filter=date_key
        )
        plotting_saving_path = os.path.join(config.BASE_RESULTS_DIR, f"{date_key}_ML_MODELS")
        input_data = forecast_results_monitor.get_forecast_input(
            f"{date_key}_forecast_{Constants.FORECAST_INPUT_FIELD_NAME.value}"
        )
        ref_obs = forecast_results_monitor.get_forecast_refobs(
            f"{date_key}_forecast_{Constants.REF_OBS_FIELD_NAME.value}"
        )
        data_date_frame = forecast_results_monitor.get_data_date_frame(
            f"{date_key}_forecast_{Constants.DATA_DATE_FRAME_NAME.value}"
        )

        forecast_data = {model: plotting_data[model] for model in unique_model_keys}

        forecast_plotter = ForecastPlotter(
            plotter=PlotTool(
                ColormapConfig(), config.individually_plotting, time_interval=5, independent_metrics=metrics_data
            ),
            prediction_step=ref_obs.shape[0],
            input_field=input_data,
            forecast=forecast_data,
            refobs_field=ref_obs,
            metadata=metadata,
            data_date_frame=data_date_frame,
            title_name=f"Forecast_Plotting_ML_Models_{date_key}",
            storage_path=plotting_saving_path,
        )
        forecast_plotter.plot_forecast()


def plot_time_consumed():
    forecast_results_monitor = ForecastResultsMonitor()
    forecast_data_path = os.path.join(config.BASE_RESULTS_DIR, Constants.FORECAST_PKL_FILE_NAME.value)

    if os.path.exists(forecast_data_path):
        forecast_results_monitor.read_from_file(forecast_data_path)

    time_consumed_dict = forecast_results_monitor.assemble_time_consumed_dict()
    plot_bars_dict(
        scores=time_consumed_dict,
        metric_name="time_consumed(in seconds)",
        save_path=os.path.join(config.BASE_RESULTS_DIR, "time_consumed_plot.png"),
        title_name="Time Consumed Average",
    )


def main():
    plot_time_consumed()
    plot_ml_models_horizontal_compare()

    root_directory = config.BASE_RESULTS_DIR
    method_scores = collect_method_scores(root_directory, model_name_filter="")

    # Adjust 'HK' and 'BIAS' scores
    adjust_scores(method_scores)

    # Define which metrics are larger is better and which are smaller is better
    larger_better_metrics = ["POD", "ACC", "CSI", "HSS", "GSS", "MCC", "F1", "Area_Under_ROC", "SEDI"]
    lower_better_metrics = ["FAR", "FA", "Distance_to_OBS", "CRPS", "HK", "BIAS"]

    accuracy_metrics = ["POD", "ACC", "CSI", "HSS", "GSS", "MCC", "F1", "Area_Under_ROC", "SEDI"]
    distribution_metrics = ["FAR", "FA", "Distance_to_OBS", "CRPS", "HK", "BIAS"]

    # Find best scores
    best_scores = find_best_scores(method_scores, larger_better_metrics, lower_better_metrics)

    # Count how many times each subdir appears as best
    score_rank = defaultdict(int)
    for metric, data in best_scores.items():
        subdir = data["subdir"]
        score_rank[subdir] += 1
        print(f"{metric}: Best score in '{subdir}' with score {data['score']}")

    # Print the rankings
    print("\nScore rankings:")
    for subdir, count in sorted(score_rank.items(), key=lambda x: x[1], reverse=True):
        print(f"{subdir}: {count}")

    # Create an instance of the comparer
    comparer = MetricModelComparer(
        data=method_scores,
        accuracy_metrics=accuracy_metrics,
        distribution_metrics=distribution_metrics,
        larger_better_metrics=larger_better_metrics,
        lower_better_metrics=lower_better_metrics,
        preserve_patterns=[
            r"epoch=\d+",
            r"(train|val|test)",
            r"(g_loss|eval_score|crps)",
            r"E\d+",
            r"^GridCellReg$",
            r"^LR$",
            r"^[DG]LR\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$",
            r"^BS\d+-PW\d+(?:\.\d+)?$",
        ],
        delimiters=r"[_=]",
        group_weight_accuracy=0.5,
        group_weight_distribution=0.5,
        plot_independently=config.individually_plotting,
    )

    # Run the comparer to process data and plot metrics
    output_dir = os.path.join(config.BASE_RESULTS_DIR, Constants.MODEL_SELECTING_RESULT_DIR_NAME.name)
    topsis_result = comparer.run(
        output_dir=output_dir,
    )

    print("TOPSIS aggregate score (High to low):")
    for model, score in topsis_result:
        print(f"  {model}: {score:.4f}")


if __name__ == "__main__":
    main()
