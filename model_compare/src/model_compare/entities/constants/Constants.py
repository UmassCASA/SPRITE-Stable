from enum import Enum, unique


@unique
class Constants(Enum):
    PYSTEPS_METHODS_NAME_KEYWORD = "Pysteps"
    DGMR_METHODS_NAME_KEYWORD = "DGMR"
    NOWCASTNET_METHODS_NAME_KEYWORD = "NowcastNet"
    UNet_METHODS_NAME_KEYWORD = "UNet"
    FORECAST_CLASS_NAME_SUFFIX = "Forecast"

    FORECAST_DICT_PLOT_KEY = "Forecast_Plot"
    FORECAST_DICT_METRIC_KEY = "Forecast_Metric"

    INFERENCE_MODEL_METHODS_KEY = "InferenceModelMethod"
    PARAMETRIC_MODEL_METHODS_KEY = "ParametricModelMethod"

    MODEL_SELECTING_RESULT_DIR_NAME = "Model_Selection_Results"

    REF_OBS_FIELD_NAME = "refobs"
    FORECAST_INPUT_FIELD_NAME = "input"
    DATA_DATE_FRAME_NAME = "data_date_frame"

    CRPS_METRIC_NAME = "CRPS"
    CSI_METRIC_NAME = "CSI"
    POD_METRIC_NAME = "POD"

    METRICS_PKL_FILE_NAME = "metrics.pkl"
    METRICS_AVERAGE_PKL_FILE_NAME = "metrics_average.pkl"
    FORECAST_PKL_FILE_NAME = "forecast_results.pkl"

    DECORATOR_TIMING_FILE_NAME_EXECUTIOON = "_execution_time"
    DECORATOR_TIMING_FILE_NAME_LOAD_MODEL = "_load_model_time"
    DECORATOR_TIMING_FILE_NAME_PREDICT = "_predict_time"
    FORECAST_METHOD_INTERFACE_GENERATE_FUNCTION_NAME = "generate"
    FORECAST_METHOD_INTERFACE_LOAD_MODEL_FUNCTION_NAME = "load_model"
    FORECAST_METHOD_INTERFACE_PREDICT_FUNCTION_NAME = "predict"
    TIMING_CONFIG = {
        FORECAST_METHOD_INTERFACE_PREDICT_FUNCTION_NAME: DECORATOR_TIMING_FILE_NAME_PREDICT,
        FORECAST_METHOD_INTERFACE_LOAD_MODEL_FUNCTION_NAME: DECORATOR_TIMING_FILE_NAME_LOAD_MODEL,
        FORECAST_METHOD_INTERFACE_GENERATE_FUNCTION_NAME: DECORATOR_TIMING_FILE_NAME_EXECUTIOON,
    }

    def __str__(self):
        """
        Returns a string representation of the enumeration value for easy output in logs or other scenarios.
        """
        return f"{self.name}: {self.value}"


# Examples
if __name__ == "__main__":
    print(Constants.PYSTEPS_METHODS_NAME_KEYWORD)

    for constant in Constants:
        print(constant)
