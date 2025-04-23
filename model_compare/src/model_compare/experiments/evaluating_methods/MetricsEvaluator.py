import logging

import numpy as np
from pysteps.verification.probscores import CRPS
from pysteps.verification.detcatscores import det_cat_fct_init, det_cat_fct_accum, det_cat_fct_compute


class MetricsEvaluator:
    def __init__(self, refobs_field):
        self.refobs_field = refobs_field
        self.crps_scores_3d = None
        self.det_cat_metric_score_dict_3d = None
        self.crps_scores_2d = None
        self.det_cat_metric_score_dict_2d = None

    def _compute_crps_score(self, forecast, time_index=None):
        forecast[forecast > 128] = 128
        forecast[forecast < 0] = 0
        forecast = np.nan_to_num(forecast, nan=0.0, posinf=128, neginf=0.0)

        if time_index is None:
            return np.mean(CRPS(forecast, self.refobs_field))
        else:
            return np.mean(CRPS(forecast, self.refobs_field[time_index : time_index + 1, :, :]))

    def get_crps_score_3d(self, forecast_dict):
        if self.crps_scores_3d is None:
            crps_scores = {}

            for method_name, forecast in forecast_dict.items():
                crps_scores[method_name] = self._compute_crps_score(forecast)

            self.crps_scores_3d = crps_scores

        # Print CRPS scores
        for method, score in self.crps_scores_3d.items():
            logging.info(f"CRPS {method}: {score}")
        return self.crps_scores_3d

    def get_crps_score_2d(self, forecast_dict):
        if self.crps_scores_2d is None:
            crps_scores = {}

            for method_name, forecast in forecast_dict.items():
                score_list = []
                for i in range(forecast.shape[1]):
                    score_list.append(self._compute_crps_score(forecast[:, i : i + 1, :, :], time_index=i))
                crps_scores[method_name] = score_list

            self.crps_scores_2d = crps_scores

        # Print CRPS scores
        for method, score in self.crps_scores_2d.items():
            logging.info(f"Independent CRPS {method}: {score}")
        return self.crps_scores_2d

    def get_det_cat_scores_3d(self, forecast_dict, threshold, scores=""):
        if self.det_cat_metric_score_dict_3d is None:
            contabs = {}

            for method_name, forecast in forecast_dict.items():
                contab = det_cat_fct_init(threshold)
                det_cat_fct_accum(contab, forecast, self.refobs_field)
                contabs[method_name] = contab

            detcatscores = {}

            for method_name, contab in contabs.items():
                detcatscores[method_name] = det_cat_fct_compute(contab, scores=scores)

            self.det_cat_metric_score_dict_3d = {
                metric: {model: values[metric] for model, values in detcatscores.items()}
                for metric in next(iter(detcatscores.values())).keys()
            }

        # Print deterministic categorical scores
        logging.info(self.det_cat_metric_score_dict_3d)

        return self.det_cat_metric_score_dict_3d

    def get_det_cat_scores_2d(self, forecast_dict, threshold, scores=""):
        if self.det_cat_metric_score_dict_2d is None:
            contabs = {}

            for method_name, forecast in forecast_dict.items():
                contab_list = []
                for i in range(forecast.shape[0]):
                    contab = det_cat_fct_init(threshold)
                    det_cat_fct_accum(contab, forecast[i : i + 1, :, :], self.refobs_field[i : i + 1, :, :])
                    contab_list.append(contab)
                contabs[method_name] = contab_list

            detcatscores = {}

            for method_name, contab_list in contabs.items():
                detcatscores[method_name] = [det_cat_fct_compute(contab, scores=scores) for contab in contab_list]

            self.det_cat_metric_score_dict_2d = {}

            self.det_cat_metric_score_dict_2d = {
                metric: {model: [score_dict[metric] for score_dict in values] for model, values in detcatscores.items()}
                for metric in next(iter(detcatscores.values()))[0].keys()
            }

        # Print deterministic categorical scores
        logging.info(self.det_cat_metric_score_dict_2d)

        return self.det_cat_metric_score_dict_2d
