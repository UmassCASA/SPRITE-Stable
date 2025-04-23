import logging
import os
import torch
from torch.utils.data import DataLoader
from SmaAt_UNet.models import unet_casa_regression_lightning as unet_regr
from model_compare.experiments.results_providers.Interface.PredictProviderInterface import PredictProviderInterface


class UNetPredictor(PredictProviderInterface):
    def __init__(
        self,
        dataset,
        model_name,
        path_to_dir,
        forecast_steps,
        num_prev_files,
        sample_idx=0,
        model=None,
    ):
        super().__init__(
            dataset,
            model_name,
            path_to_dir,
            forecast_steps,
            num_prev_files,
            sample_idx,
            model,
        )
        self.test_data_loader = DataLoader(self.dataset, batch_size=1)

        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Using device: %s", self.device)

    def load_model(self):
        if self.model is None:
            # Initialize hyperparameters for UNet models
            hparams = {
                "n_channels": self.num_prev_files + 1,
                "n_classes": self.forecast_steps,
                "num_input_images": self.num_prev_files + 1,
                "num_output_images": self.forecast_steps,
                "bilinear": True,
                "reduction_ratio": 16,
                "kernels_per_layer": 2,
                "metrics_path": "unet_metrics",
                "job_id": 123321,
                "model": self.model_name,
            }

            # Determine model type and create the correct model
            if "UNetDSAttention" in self.model_name:
                self.model = unet_regr.UNetDS_Attention(hparams=hparams)
            elif "UNetAttention" in self.model_name:
                self.model = unet_regr.UNet_Attention(hparams=hparams)
            elif "UNetDS" in self.model_name:
                self.model = unet_regr.UNetDS(hparams=hparams)
            else:
                self.model = unet_regr.UNet(hparams=hparams)

            self.model = self.model.to(self.device)

        # Load checkpoint
        checkpoint_path = os.path.join(self.path_to_dir, f"{self.model_name}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def get_prediction(self):
        with torch.no_grad():
            # Get input data and reshape
            test_inputs = self.test_data_loader.dataset[self.sample_idx][0]
            test_inputs = test_inputs[: self.num_prev_files + 1]
            test_inputs = torch.from_numpy(test_inputs)

            # Reshape input from [S, H, W] to [1, S, H, W]
            test_inputs = test_inputs.unsqueeze(0).to(self.device)

            # Forward pass
            test_outputs = self.model(test_inputs.float())

            return test_outputs.squeeze(0).cpu().numpy()
