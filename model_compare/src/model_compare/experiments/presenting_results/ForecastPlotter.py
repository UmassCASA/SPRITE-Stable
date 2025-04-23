import math
import os


class ForecastPlotter:
    def __init__(
        self,
        plotter,
        prediction_step,
        input_field,
        forecast,
        refobs_field,
        metadata,
        data_date_frame,
        title_name,
        storage_path,
    ):
        self.prediction_step = prediction_step
        self.data_date_frame = data_date_frame
        self.storage_path = storage_path
        self.plotter = plotter
        self.input_field = input_field
        self.forecast = forecast
        self.metadata = metadata
        self.title_name = title_name
        self.refobs_field = refobs_field

    def plot_forecast(self):
        # fig, axes = plt.subplots(1, self.prediction_step, figsize=(20, 5))
        # for i in range(self.prediction_step):
        #     ax = axes[i]
        #     ax.set_title(f'{title_name}: Min {(i + 1) * 5}')
        #     ax.axis('off')
        #     quiver(self.velocity, step=50, ax=ax)
        #     plot_precip_field(forecast[i, :, :], ax=ax)

        if self.storage_path is not None:
            num_groups = math.ceil(len(self.forecast) / 4)

            for i in range(num_groups):
                dir_name = os.path.join(self.storage_path, "forecast_plot", f"{self.title_name}-{i}.png")
                os.makedirs(os.path.dirname(dir_name), exist_ok=True)

                self.plotter.plot_data_and_save(
                    inputs=self.input_field,
                    targets=self.refobs_field,
                    outputs=self.forecast,
                    metadata=self.metadata,
                    directory=dir_name,
                    moment_datetime=[dt.strftime("%y-%m-%d_%H:%M:%S") for dt in self.data_date_frame][
                        self.prediction_step + 1
                    ],
                    title=self.title_name,
                    model_subset_idx=i,
                )

        #     plt.savefig(dir_name)
        #
        # plt.show()
