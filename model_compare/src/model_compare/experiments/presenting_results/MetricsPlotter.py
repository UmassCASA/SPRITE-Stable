import logging
import os
import numpy as np

import matplotlib.pyplot as plt
from pysteps.postprocessing import ensemblestats
from pysteps import verification
from pysteps.verification import probscores

from model_compare.utils.plotting.plot_bar import plot_bars_dict


class MetricsPlotter:
    def __init__(
        self,
        refobs_field,
        prediction_step,
        model_path,
        storage_path,
        data_date_frame,
        detcat_metric_score_dict,
        crps_scores,
    ):
        self.refobs_field = refobs_field
        self.prediction_step = prediction_step
        self.model_path = model_path
        self.storage_path = storage_path
        self.data_date_frame = data_date_frame
        self.detcat_metric_score_dict = detcat_metric_score_dict
        self.crps_scores = crps_scores

    def plot_and_record_verification_metrics(self, forecast_dict):
        ROC_AUC = {}
        distance_to_RD_diag = {}

        for title_name, forecast in forecast_dict.items():
            P_f = ensemblestats.excprob(forecast, 0.1, ignore_nan=True)
            roc = verification.ROC_curve_init(0.1, n_prob_thrs=10)
            verification.ROC_curve_accum(roc, P_f, self.refobs_field[-1, :, :])
            _, _, area = probscores.ROC_curve_compute(roc, compute_area=True)

            ROC_AUC[title_name] = area

            fig, ax = plt.subplots()
            verification.plot_ROC(roc, ax, opt_prob_thr=True)
            ax.set_title(f"ROC curve of {title_name} (+{self.prediction_step * 5} min)")
            if self.storage_path is not None:
                dir_name = os.path.join(self.storage_path, "verification_metrics/ROC", f"{title_name}_ROC.png")
                os.makedirs(os.path.dirname(dir_name), exist_ok=True)
                plt.savefig(dir_name)
            plt.show()

            reldiag = verification.reldiag_init(0.1)
            verification.reldiag_accum(reldiag, P_f, self.refobs_field[-1, :, :])

            p_observed = reldiag["X_sum"] / reldiag["num_idx"]
            p_forecast = reldiag["Y_sum"] / reldiag["num_idx"]
            mask = np.logical_and(np.isfinite(p_observed), np.isfinite(p_forecast))

            distances = p_observed - p_forecast
            fig, ax = plt.subplots()

            # filter invalid sample_size
            if np.any(reldiag["sample_size"] <= 0):
                logging.warning("Warning: sample_size contains non-positive values, filtering them.")
                reldiag["sample_size"] = np.where(reldiag["sample_size"] <= 0, 1e-10, reldiag["sample_size"])

            try:
                verification.plot_reldiag(reldiag, ax)
            except Exception as e:
                logging.error(
                    f"Error in plot_reldiag: {e}\nat dgmr model{self.model_path},\ndata frame: {self.data_date_frame}"
                )

            mean_distance = np.mean(np.abs(distances[mask]))
            distance_to_RD_diag[title_name] = mean_distance
            ax.set_title(
                f"Reliability diagram of {title_name} (+{self.prediction_step * 5} min), Distance:{mean_distance}"
            )
            for _, (x, y, d) in enumerate(zip(p_observed[mask], p_forecast[mask], distances[mask])):
                ax.plot([x, x], [x, y], color="red", linestyle="--", lw=0.5)
                ax.text(x, y, f"{d:.2f}", color="red", fontsize=8, ha="left")

            if self.storage_path is not None:
                dir_name = os.path.join(self.storage_path, "verification_metrics/RD", f"{title_name}_RD.png")
                os.makedirs(os.path.dirname(dir_name), exist_ok=True)
                plt.savefig(dir_name)
            plt.show()

        self.detcat_metric_score_dict |= {"Area_Under_ROC": ROC_AUC} | {"Distance_to_OBS": distance_to_RD_diag}

    def _plot_bar_chart_(self, scores, metric_name):
        """Plot a bar chart."""

        title = f"{metric_name} Scores for Different Forecast Methods"

        if self.storage_path is not None:
            dir_name = os.path.join(self.storage_path, "verification_metrics", f"{metric_name}.png")
            plot_bars_dict(scores=scores, metric_name=metric_name, save_path=dir_name, title_name=title)

    def plot_scores(self):
        """Plot deterministic categorical scores and CRPS as a bar chart."""
        for metric, scores in (self.detcat_metric_score_dict | {"CRPS": self.crps_scores}).items():
            self._plot_bar_chart_(scores, metric)
