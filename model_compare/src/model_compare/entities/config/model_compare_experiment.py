# config.py
from dataclasses import dataclass, field
from typing import List, Dict, Any

import yaml


@dataclass
class RunExperimentConfig:
    BASE_RESULTS_DIR: str = "experiment_results"
    MODEL_BASE_PATH: List[str] = field(
        default_factory=lambda: ["DGMR_candidate_models", "NowcastNet_candidate_models", "UNet_candidate_models"]
    )
    ABS_SCANNER_SUFFIX_LIST: List[str] = field(default_factory=lambda: ["Forecast"])
    PYSTEPS_METHODS_LIST: List[str] = field(
        default_factory=lambda: [
            "PystepsAnvilForecast",
            "PystepsExtrapolationForecast",
            "PystepsLindaDeterministicForecast",
            "PystepsLindaProbabilisticForecast",
            "PystepsSprogForecast",
            "PystepsStepsForecast",
        ]
    )
    incremental: bool = False
    individually_plotting: bool = False
    experiment_repeat_time: int = 0
    dates: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, config_file: str) -> "RunExperimentConfig":
        with open(config_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def save(self, config_file: str) -> None:
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False, allow_unicode=True)

    def have_dates(self):
        return len(self.dates) > 0
