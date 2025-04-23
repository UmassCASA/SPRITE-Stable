# Nowcasting Experiment Runner

The project is designed to run nowcasting experiments by scanning for prediction method implementations, executing forecasts, evaluating metrics, and generating visualizations. Configuration is managed via a YAML file, and command-line arguments can override the default settings.

---

## 1. Environment and Dependencies

- **Python Version:** Python 3.6 and above
- **Dependencies:**
  
  - [PyYAML](https://pyyaml.org/) – for reading YAML configuration files  
    ```bash
    pip install pyyaml
    ```
  - [NumPy](https://numpy.org/) – for numerical computations  
    
    ```bash
    pip install numpy
    ```
  - Other dependencies are part of the Python standard library (e.g., `argparse`, `json`, `os`, `logging`, `re`, `datetime`, `pickle`, `concurrent.futures`, etc.)

---

## 2. Configuration File

The project uses a YAML configuration file (`run_experiment.yaml`) to manage global paths and certain experimental parameters. An example configuration is as follows:

```yaml
BASE_RESULTS_DIR: experiment_results
MODEL_BASE_PATH:
#  This list must contain at least one item
  - DGMR_candidate_models
  - NowcastNet_candidate_models
PYSTEPS_METHODS_LIST:
#  If it's needed avoid pysteps methods running, set to [] instead of empty(None)
#  []
  - PystepsAnvilForecast
  - PystepsExtrapolationForecast
  - PystepsLindaDeterministicForecast
  - PystepsLindaProbabilisticForecast
  - PystepsSprogForecast
  - PystepsStepsForecast
incremental: True
dates:
#  "min_density": "20230514_0933"
#  "min_complexity": "20230505_0331"
#  "min_intensity": "20240103_0359"
#  "max_density": "20231109_1519"
#  "max_complexity": "20231224_1454"
#  "max_intensity": "20231224_1429"
#  "median_density": "20230201_2317"
#  "median_complexity": "20230914_1251"
  "median_intensity": "20230426_2241"
#  "idx:198": 198
#  "idx:273": 273
#  "idx:145": 145
#  "idx:148": 148
  "idx:116": 116

```

- **BASE_RESULTS_DIR:**
  Specifies the root directory for storing experimental results. All forecast outputs, evaluation metrics, and visualizations will be saved under this directory.
-  **MODEL_BASE_PATH:**
  Lists the directories where pre-trained model checkpoints are stored.
-  **PYSTEPS_METHODS_LIST**
  List of the PySteps methods that need to run
-  **incremental**
  Incremental mode switch. When enabled, the program reads the previous experimental result cache and, based on that, executes experiments for new models or a new time range, appending the new results to the cache.
-  **dates**
  List of time intervals
  
- **Note:**
  - Pre-trained models should be stored in directories named according to the model type, following the format `{ModelName}_{OtherDetails}`.
  - The portion before the first underscore must uniquely match the identifier present in the corresponding prediction method implementation class located in `model_compare/experiments/forecast_methods/Impl`. For example, the directory `DGMR_candidate_models` should correspond to the implementation class `DGMRForecast` (where "DGMR" is the unique identifier linking the storage directory and the prediction method).

------

## 3. Code Logic Overview

### 3.1. Experimental Workflow

- **Configuration Loading:**
  Upon startup, the program reads the `run_experiment.yaml` file to load global path configurations (`BASE_RESULTS_DIR` and `MODEL_BASE_PATH`, etc.). 
- **Model Selection:**
  - The program traverses the directories specified in `MODEL_BASE_PATH` to find files with the `.ckpt` extension. These files, along with directory information (where the directory name prefix corresponds to the unique identifier in the prediction method implementation), are compiled into a list of models to process.
- **Prediction Method Scanning:**
  The program uses `AbstractClassScanner` to automatically scan for all prediction method implementations (located in `model_compare/experiments/forecast_methods/Impl`) that inherit from `ForecastMethodInterface`. A `scanner_filter` (e.g., `{'contains': [Constants.PYSTEPS_METHODS_NAME_KEYWORD.value, method_name]}`) is applied to filter out the relevant prediction method.
- **Experiment Execution:**
  For each model and date combination, the program uses `ThreadPoolExecutor` to asynchronously submit tasks that call `process_experiment`. The experimental workflow includes:
  - Creating a storage directory (located under `BASE_RESULTS_DIR/model_path/date_key`)
  - Instantiating a `NowcastingExperiment` object to run forecasts, evaluate metrics, and visualize the results
  - Saving evaluation metrics as a pickle file
- **Results Aggregation and Visualization:**
  After completing all experiments, the program aggregates average evaluation metrics for each model and uses plotting tools (e.g., `plot_average`) to generate bar charts. The charts and aggregated data are saved in corresponding directories under `BASE_RESULTS_DIR`.

### 3.2. Incremental Mode

- **Description:**
  When incremental mode is enabled (via the `--incremental` flag), the program first reads the cached experimental results from the previous run, then executes experiments only for new models or a new time range, appending the new results to the existing cache.

------

## 4. Adding New Prediction Methods

To add a new prediction method, follow these steps:

1. **Implement the Prediction Method:**
   - Depending on whether the new prediction method is an inference model or a parametric model, inherit from the appropriate base class (`InferenceModelMethod` or `ParametricModelMethod`) defined in `model_compare/experiments/forecast_methods/Interface/ForecastMethodBase.py`.
   - Implement all required abstract methods to ensure the new class can be instantiated. Note: The output shape of an abstract methods that must be implemented “generate” **must be TxWxH**
2. **Place the New Class:**
   - Save the new prediction method class in the `model_compare/experiments/forecast_methods/Impl` directory.
   - Ensure the new class name contains a unique identifier that corresponds to the storage directory naming convention (see Section 2).
3. **Model Storage Directory:**
   - Store the pre-trained checkpoint for the new model in a directory named following the format `{ModelName}_{OtherDetails}`. The prefix before the first underscore must uniquely match the identifier in the new prediction method class.
4. **Automatic Scanning:**
   - After completing the above steps, rerun the code. The `AbstractClassScanner` will automatically detect the new prediction method implementation, and the corresponding checkpoint in the designated directory will be used for forecasting.

------

## 5. Multi-Model Comparison

If you wish to compare multiple models (e.g., using multi-criteria decision methods such as TOPSIS):

- After running `run_experiment.py`, execute the script `DGMR_model_selecting.py` located at the same level as the `experiment_results` directory.
- The TOPSIS scores will be printed in the output log, and other aggregated scores will be visualized as bar charts, which are saved in the `experiment_results/DGMR_model_comparer_figures` directory.

------

## 6. Running the Experiment

Ensure that the `run_experiment.yaml` file is located in the project root and is correctly configured. Then, execute the following command from the project root:

```bash
python run_experiment.py
```

------

## 7. Logs and Output

- **Logging:**
  The program uses Python's built-in `logging` module to output progress, error messages, and debug information. The log format is:

  ```shell
  [Timestamp] - [Log Level] - [Message]
  ```

- **Experiment Output:**

  - Each model and date combination's results are stored under the corresponding directory in `BASE_RESULTS_DIR` (e.g., `experiment_results/model_path/date_key`).
  - Outputs include forecast data, evaluation metrics (saved as `metrics.pkl`), and visualizations.
  - Aggregated average metrics are saved as `metrics_average.pkl`, with associated charts and aggregated scores saved in their respective folders.