import yaml
from pathlib import Path
from Preprocessor import Preprocessor


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def run_preprocessor(config_path):
    """Run the preprocessor using configuration from YAML file."""
    # Load configuration
    config = load_config(config_path)

    # Get precip_cap and normalization strategy from config
    precip_cap = config["precip_cap"]
    train_valid_norm = config["train_valid"].get("normalization_strategy", "none")
    test_norm = config["test"].get("normalization_strategy", "none")

    # Update paths in config with precip_cap and normalization strategy
    base_path = "/work/pi_mzink_umass_edu/SPRITE/data"

    # Update train/validation paths
    train_valid_path = f"{base_path}/CASAData_sequenced_{precip_cap}"
    if train_valid_norm != "none":
        train_valid_path = f"{train_valid_path}/{train_valid_norm}"

    config["train_valid"]["output_directory"] = f"{train_valid_path}/train/"
    config["train_valid"]["validation_directory"] = f"{train_valid_path}/validation/"

    # Update test paths
    test_path = f"{base_path}/CASAData_sequenced_{precip_cap}"
    if test_norm != "none":
        test_path = f"{test_path}/{test_norm}"

    config["test"]["output_directory"] = f"{test_path}/test/"

    # Process training and validation data
    processor_train_valid = Preprocessor(**config["train_valid"])
    processor_train_valid.setup_process()

    # Process test data
    processor_test = Preprocessor(**config["test"])
    processor_test.setup_process()


if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = Path(__file__).parent
    config_path = script_dir / "config.yml"

    run_preprocessor(config_path)
