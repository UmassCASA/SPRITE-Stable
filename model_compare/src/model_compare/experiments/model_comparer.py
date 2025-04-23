import pandas as pd
import seaborn as sns
import re
import os
import numpy as np
from collections import Counter

from src.model_compare.utils.plotting.plot_bar import plot_bars_dict, plot_bars_dict_in_one


def dynamic_simplify_model_names(
    model_names,
    preserve_patterns,
    delimiters=r"[-_=\\.]",
):
    """
    Example function to dynamically simplify model names.

    Parameters
    ----------
    model_names : list of str
        List of original model names.
    delimiters : str, default r"[-_=\\.]"
        Specify delimiters in regex format. Default includes '-', '_', '=', '.'.
    preserve_patterns : list of str
        List of regex patterns to preserve. Tokens matching these patterns will not be removed even if they are common.

    Returns
    -------
    simplified_names : dict
        {Original model name: Simplified name}
    """

    # 1) Split each model name by the specified delimiters
    token_lists = []
    for name in model_names:
        tokens = re.split(delimiters, name)
        tokens = [t for t in tokens if t]  # Remove empty strings
        token_lists.append(tokens)

    # 2) Retain only tokens that match the preserve_patterns
    def is_preserved_token(token):
        return any(re.search(pat, token) for pat in preserve_patterns)

    filtered_token_lists = []
    for tokens in token_lists:
        filtered = [t for t in tokens if is_preserved_token(t)]
        filtered_token_lists.append(filtered)

    # 3) Join tokens matching the criteria with "-". If no tokens match, use placeholder "GenericModel"
    final_names = []
    for tokens in filtered_token_lists:
        if not tokens:
            final_names.append("GenericModel")
        else:
            final_names.append("-".join(tokens))

    # 4) Resolve conflicts (duplicates) by adding a sequential index
    name_counts = Counter(final_names)
    used_count = {}
    resolved_names = []
    for _raw_name, simp_name in zip(model_names, final_names):
        if name_counts[simp_name] > 1:
            used_count[simp_name] = used_count.get(simp_name, 0) + 1
            new_simp_name = f"{simp_name}({used_count[simp_name]})"
            resolved_names.append(new_simp_name)
        else:
            resolved_names.append(simp_name)

    # 5) Assemble {Original name: Simplified name} and return
    simplified_names = dict(zip(model_names, resolved_names))
    return simplified_names


class MetricModelComparer:
    def __init__(
        self,
        data,
        accuracy_metrics,
        distribution_metrics,
        larger_better_metrics,
        lower_better_metrics,
        preserve_patterns,
        delimiters,
        group_weight_accuracy=0.5,
        group_weight_distribution=0.5,
        plot_independently=True,
    ):
        """
        Initializes the comparer with the provided data.

        Parameters:
        - data: A nested dictionary in the format {metric_name: {model_name: value}}
        """
        self.original_data = data
        self.df = self._prepare_dataframe(data)
        self.preserve_patterns = preserve_patterns
        self.delimiters = delimiters
        self.model_name_mapping = self._create_model_name_mapping()

        self.accuracy_metrics = accuracy_metrics
        self.distribution_metrics = distribution_metrics
        self.group_weight_accuracy = group_weight_accuracy
        self.group_weight_distribution = group_weight_distribution
        self.larger_better_metrics = larger_better_metrics
        self.lower_better_metrics = lower_better_metrics
        self.plot_independently = plot_independently

    def _prepare_dataframe(self, data):
        """
        Converts the nested dictionary to a pandas DataFrame.
        """
        df = pd.DataFrame.from_dict(data, orient="index").T
        return df

    def _create_model_name_mapping(self):
        """
        Dynamically generates a mapping for simplified model names.
        """
        all_model_names = list(self.df.index)

        mapping = dynamic_simplify_model_names(
            model_names=all_model_names, delimiters=self.delimiters, preserve_patterns=self.preserve_patterns
        )
        return mapping

    def process_data(self):
        """
        Processes the data by simplifying model names and handling duplicates.
        """
        # Rename the index with simplified model names
        # self.df.rename(index=self.model_name_mapping, inplace=True)

        # Handle duplicate simplified names by averaging their values
        self.df = self.df.groupby(self.df.index).mean()

    def plot_metrics(
        self,
        save_figures=True,
        output_dir="figures",
    ):
        """
        Plots the metrics for each metric name.

        Parameters:
        - save_figures: Whether to save the figures to files.
        - output_dir: The directory where figures will be saved.
        """

        # Set the plotting style
        sns.set(style="whitegrid")

        if not self.plot_independently:
            save_path = os.path.join(output_dir, "model_comparison.png")
            plot_bars_dict_in_one(
                df=self.df,
                save_path=save_path,
            )
        else:
            # Iterate over each metric
            for metric in self.df.columns:
                # Get the data for the current metric
                s = self.df[metric].dropna()
                # Sort the data ascending
                s_sorted = s.sort_values(ascending=True)

                # Create the plot
                plot_bars_dict(
                    scores=s_sorted.to_dict(),
                    metric_name=metric,
                    save_path=os.path.join(output_dir, metric + "_model_comparison.png"),
                    title_name=f"Model Comparison for Metric: {metric}",
                )

    def compute_topsis_scores(self):
        """
        Computes the TOPSIS scores for models based on the data,
        considering the group weights of 'accuracy_metrics' and 'distribution_metrics'.
        """

        # 0) Collect all models and metrics (use only relevant metrics if needed)
        #    Here, assume we use accuracy_metrics + distribution_metrics
        used_metrics = list(
            set(self.accuracy_metrics + self.distribution_metrics) & set(self.original_data.keys())
        )  # Only metrics present in the data
        # Collect all models
        model_set = set()
        for m in used_metrics:
            model_set.update(self.original_data[m].keys())
        model_list = sorted(model_set)

        # 1) Assign equal weight to each metric within a group
        accuracy_in_use = [m for m in self.accuracy_metrics if m in used_metrics]
        distribution_in_use = [m for m in self.distribution_metrics if m in used_metrics]

        weight_acc_each = (self.group_weight_accuracy / len(accuracy_in_use)) if len(accuracy_in_use) > 0 else 0
        weight_dist_each = (
            (self.group_weight_distribution / len(distribution_in_use)) if len(distribution_in_use) > 0 else 0
        )

        metric_weights = {}
        for m in accuracy_in_use:
            metric_weights[m] = weight_acc_each
        for m in distribution_in_use:
            metric_weights[m] = weight_dist_each

        # 2) Construct the decision matrix: rows=models, columns=metrics
        matrix = np.zeros((len(model_list), len(used_metrics)), dtype=float)
        for j, metric in enumerate(used_metrics):
            for i, mdl in enumerate(model_list):
                matrix[i, j] = self.original_data[metric].get(mdl, np.nan)

        # 3) Normalize by vector length (Euclidean norm)
        for j in range(len(used_metrics)):
            col = matrix[:, j]
            norm = np.sqrt(np.nansum(col**2))
            if norm > 0:
                matrix[:, j] = col / norm

        # 4) Multiply by weights
        for j, metric in enumerate(used_metrics):
            w = metric_weights.get(metric, 0.0)
            matrix[:, j] = matrix[:, j] * w

        # 5) Determine positive and negative ideal solutions
        pos_ideal = np.zeros(len(used_metrics))
        neg_ideal = np.zeros(len(used_metrics))
        for j, metric in enumerate(used_metrics):
            col = matrix[:, j]
            if metric in (self.larger_better_metrics or []):
                pos_ideal[j] = np.nanmax(col)
                neg_ideal[j] = np.nanmin(col)
            elif metric in (self.lower_better_metrics or []):
                pos_ideal[j] = np.nanmin(col)
                neg_ideal[j] = np.nanmax(col)
            else:
                pos_ideal[j] = np.nanmax(col)
                neg_ideal[j] = np.nanmin(col)

        # 6) Calculate distances to positive/negative ideal solutions
        d_pos = np.zeros(len(model_list))
        d_neg = np.zeros(len(model_list))
        for i in range(len(model_list)):
            d_pos[i] = np.sqrt(np.nansum((matrix[i, :] - pos_ideal) ** 2))
            d_neg[i] = np.sqrt(np.nansum((matrix[i, :] - neg_ideal) ** 2))

        # 7) Compute overall scores
        scores = d_neg / (d_neg + d_pos)

        # 8) Return a sorted result of (model -> TOPSIS score)
        results = sorted(zip(model_list, scores), key=lambda x: x[1], reverse=True)
        return results

    def run(self, save_figures=True, output_dir="figures"):
        """
        Runs the full process of data processing and plotting.
        """
        self.process_data()
        self.plot_metrics(save_figures, output_dir)
        return self.compute_topsis_scores()
