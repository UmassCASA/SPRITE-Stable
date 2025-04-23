import io
import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib import cm

from sprite_core.config import Config


class PlotTool:
    def __init__(self, cmap_config, individually_plotting=False, time_interval=5, independent_metrics=None):
        self.cmap_config = cmap_config
        self.individually_plotting = individually_plotting
        self.time_interval = time_interval
        self.mapbox_token = Config.MAPBOX_TOKEN
        self.style_id = "mapbox/streets-v12"
        self.independent_metrics = independent_metrics

    def plot_data_and_save(
        self,
        inputs,
        targets,
        outputs,
        metadata,
        directory,
        moment_datetime="Unspecified",
        title=None,
        model_subset_idx=0,
    ):
        """Plot precipitation data and save the visualization to a specified directory.

        Args:
            inputs: Observational data slices.
            targets: Ground truth target data slices.
            outputs: Model output predictions.
            metadata: Geospatial metadata.
            directory: Output directory for the plot.
            moment_datetime (str, optional): Moment datetime string for labeling. Defaults to "Unspecified".
            title (str, optional): Title of the plot. Defaults to None.
            model_subset_idx (int, optional): Subset index for the models to plot. Defaults to 0.
        """

        input_slices = inputs
        # target_slices = [targets[i, :, :] for i in range(min(4, targets.shape[1]))]
        target_slices = targets

        if not self.individually_plotting:
            # width = 64
            # height = 25.4

            # fig = plt.figure(figsize=(width, height))

            if len(target_slices) == 10 and len(input_slices) == 4:
                width = 64
                height = 25.4
                fig = plt.figure(figsize=(width, height))

                gs1 = fig.add_gridspec(1, 4, bottom=0.7626, top=0.94, wspace=0.02, hspace=0.02, left=0.00, right=0.2820)
                gc2 = fig.add_gridspec(5, 10, bottom=0.04, top=0.94, wspace=0.02, hspace=0.02, left=0.2835, right=0.992)

            elif len(target_slices) == 8 and len(input_slices) == 4:
                width = 57
                height = 26
                fig = plt.figure(figsize=(width, height))

                gs1 = fig.add_gridspec(1, 4, bottom=0.7630, top=0.94, wspace=0.02, hspace=0.02, left=0.00, right=0.3305)
                gc2 = fig.add_gridspec(5, 8, bottom=0.04, top=0.94, wspace=0.02, hspace=0.02, left=0.3325, right=0.992)

            else:
                width = 64
                height = 25.4
                fig = plt.figure(figsize=(width, height))

                gs1 = fig.add_gridspec(
                    1, len(input_slices), bottom=0.7626, top=0.94, wspace=0.02, hspace=0.02, left=0.00, right=0.2820
                )
                gc2 = fig.add_gridspec(
                    5, len(target_slices), bottom=0.04, top=0.94, wspace=0.02, hspace=0.02, left=0.2835, right=0.992
                )

                warnings.warn(
                    f"Plotting may be broken. Formatting for {len(input_slices)} input slices"
                    f"and {len(target_slices)} target slices is not implemented.",
                    stacklevel=2,
                )

        else:
            fig = None
            gs1 = None
            gc2 = None

        model_items = list(outputs.items())
        start_idx = model_subset_idx * 4
        end_idx = min(start_idx + 4, len(model_items))
        current_outputs = dict(model_items[start_idx:end_idx])

        logging.info("Plotting observations")

        # Plot observations
        self._plot_observations(fig, gs1, input_slices, metadata, moment_datetime, directory)

        # Plot target slices
        self._plot_targets(fig, gc2, target_slices, metadata, directory)

        # Plot model outputs
        self._plot_model_outputs(fig, gc2, current_outputs, metadata, directory)

        if not self.individually_plotting:
            # Add colorbar
            self._add_colorbar(fig)

        # Ensure directory uniqueness without lock
        directory = self._ensure_unique_path(directory)

        if not self.individually_plotting:
            plt.savefig(directory, bbox_inches="tight")
            plt.show()

        plt.close()

        logging.info(f"Saved plot to {directory}")

    def _ensure_unique_path(self, directory):
        """
        Ensure the output directory has a unique name to avoid conflicts in a concurrent environment.
        """
        base, ext = os.path.splitext(directory)
        counter = 1
        unique_directory = directory

        while os.path.exists(unique_directory):
            unique_directory = f"{base}_{counter}{ext}"
            counter += 1

        return unique_directory

    def _plot_precip_field(self, precip, geodata, scores_dict=None, ax=None):
        x1, y1 = geodata["x1"], geodata["y1"]
        x2, y2 = geodata["x2"], geodata["y2"]
        x1 = x1.astype(float)
        x2 = x2.astype(float)
        y1 = y1.astype(float)
        y2 = y2.astype(float)

        url = (
            f"https://api.mapbox.com/styles/v1/{self.style_id}/static/"
            f"[{x1},{y2},{x2},{y1}]/"
            f"{256}x{256}"
            f"?access_token={self.mapbox_token}"
        )

        response = requests.get(url)
        response.raise_for_status()
        png_data = response.content

        map_img = plt.imread(io.BytesIO(png_data), format="png")

        ax.imshow(map_img)

        ax.imshow(precip, cmap=self.cmap_config.cmap, norm=self.cmap_config.norm, alpha=0.6)

        if scores_dict:
            text_content = "\n".join([f"{key}: {value}" for key, value in scores_dict.items()])

            # Calculate text box size
            bbox_props = {"boxstyle": "round,pad=0.3", "edgecolor": "black", "facecolor": "white", "alpha": 0.6}
            text_x, text_y = 0.98, 0.02  # Right bottom corner
            ax.text(
                text_x,
                text_y,
                text_content,
                transform=ax.transAxes,
                fontsize=15,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=bbox_props,
            )

    def _plot_individually(self, image_slice, metadata, directory, image_name, time_title, scores_dict=None):
        fig, ax = plt.subplots()
        ax.set_axis_off()

        self._plot_precip_field(precip=image_slice, geodata=metadata, scores_dict=scores_dict, ax=ax)
        # plot_precip_field(precip=image_slice, geodata=metadata, colorbar=False, ax=ax,
        #                   colormap_config=self.cmap_config)

        ax.set_axis_off()
        plt.tight_layout()
        saving_path = os.path.join(os.path.dirname(directory), image_name)
        file_name = os.path.join(str(saving_path), f"{time_title}.png")
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        plt.savefig(file_name, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    def _plot_observations(self, fig, gs1, input_slices, metadata, moment_datetime, directory):
        beginning_time_mark_pos = len(input_slices) - 1

        for i, input_slice in enumerate(input_slices):
            time_title = f"{(-beginning_time_mark_pos) * self.time_interval + (i * self.time_interval)} min"

            if i == beginning_time_mark_pos:
                # TODO: Convert to  last_input_datetime.strftime("%d %b, %y - %H:%M") +
                # TODO:" UTC" For that need to have moment_datetime as datetime object
                time_title = f"{moment_datetime}"

            if self.individually_plotting:
                self._plot_individually(input_slice, metadata, directory, "observation", time_title)
            else:
                ax = fig.add_subplot(gs1[0, i])
                self._plot_precip_field(precip=input_slice, geodata=metadata, ax=ax)

                if i == 0:
                    ax.text(
                        -0.07,
                        0.5,
                        "Observation",
                        fontsize=37,
                        ha="center",
                        va="center",
                        rotation=90,
                        transform=ax.transAxes,
                        clip_on=False,
                    )
                if i == beginning_time_mark_pos:
                    ax.text(
                        0.5,
                        1.07,
                        time_title,
                        fontsize=37,
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        clip_on=False,
                    )
                else:
                    ax.text(
                        0.5,
                        1.07,
                        time_title,
                        fontsize=37,
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        clip_on=False,
                    )
                self._clean_up_axes(ax)

    def _plot_targets(self, fig, gc2, target_slices, metadata, directory):
        for i, target_slice in enumerate(target_slices):
            time_title = f"{(i + 1) * self.time_interval} min"

            if self.individually_plotting:
                self._plot_individually(target_slice, metadata, directory, "observation", time_title)
            else:
                ax = fig.add_subplot(gc2[0, i])
                self._plot_precip_field(precip=target_slice, geodata=metadata, ax=ax)

                ax.text(
                    0.5, 1.07, time_title, fontsize=37, ha="center", va="center", transform=ax.transAxes, clip_on=False
                )
                self._clean_up_axes(ax)

    def _plot_model_outputs(self, fig, gc2, current_outputs, metadata, directory):
        for x, (model_name, output_slices) in enumerate(current_outputs.items(), start=1):
            logging.info(f"Model: {model_name}")
            logging.info(f"Output slices shape: {output_slices.shape}")

            if self.individually_plotting:
                num_slices = len(output_slices)
            else:
                num_slices = min(len(output_slices), gc2.ncols)

            for i in range(num_slices):
                time_title = f"{(i + 1) * self.time_interval} min"

                if self.independent_metrics is None:
                    scores_dict = None
                else:
                    scores_dict = {
                        metric: np.around(self.independent_metrics[metric][model_name][i], 2)
                        for metric in self.independent_metrics
                    }

                if self.individually_plotting:
                    self._plot_individually(
                        output_slices[i], metadata, directory, model_name, time_title, scores_dict=scores_dict
                    )
                else:
                    ax = fig.add_subplot(gc2[x, i])
                    self._plot_precip_field(
                        precip=np.squeeze(np.nan_to_num(output_slices[i], nan=0.0).astype(np.float64)),
                        geodata=metadata,
                        scores_dict=scores_dict,
                        ax=ax,
                    )

                    if i == 0:
                        ax.text(
                            -0.07,
                            0.5,
                            model_name,
                            fontsize=37,
                            ha="center",
                            va="center",
                            rotation=90,
                            transform=ax.transAxes,
                            clip_on=False,
                        )

                    self._clean_up_axes(ax)

    def _add_colorbar(self, fig):
        cmap = self.cmap_config.cmap
        norm = self.cmap_config.norm
        bounds = self.cmap_config.bounds
        cbaxes = fig.add_axes([0.3, 0.01, 0.4, 0.02])

        cbar = fig.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cbaxes,
            orientation="horizontal",
            fraction=5,
            shrink=0,
            anchor=(0.5, 1.0),
            panchor=(0.5, 0.0),
            ticks=bounds,
            extend="both",
            extendfrac="auto",
            spacing="uniform",
        )

        cbar.ax.set_xlabel(r"Rainfall intensity (mm h$^{-1}$)", fontsize=37)
        cbar.ax.tick_params(labelsize=37)

    def _clean_up_axes(self, ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
