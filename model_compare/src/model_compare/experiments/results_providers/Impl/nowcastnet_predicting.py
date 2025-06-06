# Standard library imports
import logging
import os

# Third party imports
import torch
from torch.utils.data import DataLoader

# Local imports
from nowcastnet.models.nowcastnet_pytorch_lightning import Net
from nowcastnet.layers.entities.net_config import Configs
from model_compare.experiments.results_providers.Interface.PredictProviderInterface import PredictProviderInterface


class NowcastNetPredictor(PredictProviderInterface):
    def __init__(self, dataset, model_name, path_to_dir, forecast_steps, num_prev_files, sample_idx=0, model=None):
        # Store any necessary initialization parameters if needed
        super().__init__(dataset, model_name, path_to_dir, forecast_steps, num_prev_files, sample_idx, model)
        self.test_data_loader = DataLoader(self.dataset, batch_size=1)

        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Using device: %s", self.device)

    def load_model(self):
        if self.model is None:
            net_config = Configs()
            # net_config.total_length = (self.num_prev_files + 1) + self.forecast_steps

            self.model = Net(net_config).to(self.device)

        checkpoint_path = os.path.join(self.path_to_dir, f"{self.model_name}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

    def get_prediction(self):
        with torch.no_grad():
            test_inputs = self.test_data_loader.dataset[self.sample_idx][0]
            test_inputs = test_inputs[: self.num_prev_files + 1]
            test_inputs = torch.from_numpy(test_inputs)
            test_inputs = test_inputs.unsqueeze(0).to(self.device)  # Add batch dimension
            test_outputs = self.model(test_inputs.float())

            return test_outputs.squeeze(0).squeeze(-1).cpu().numpy()[-self.forecast_steps :]
