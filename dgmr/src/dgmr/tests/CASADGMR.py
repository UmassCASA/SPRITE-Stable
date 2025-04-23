import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from netCDF4 import Dataset as NC4Dataset

from sprite_core.config import Config
from dgmr import DGMR

NUM_INPUT_FRAMES = 4
NUM_TARGET_FRAMES = 18
TOTAL_FRAMES = NUM_INPUT_FRAMES + NUM_TARGET_FRAMES


class CASADGMR:
    def __init__(self, model_name, start_day, split="train", num_epochs=1, batch_size=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.initialize_model(model_name).to(self.device)
        self.data_loader = self.create_data_loader(split, num_epochs, start_day, batch_size)

    class NetCDFDataset(Dataset):
        def __init__(self, split, num_epochs, start_day):
            super().__init__()
            self.split = split
            self.local_folder_path = os.path.join(Config.DATA_DIR, split)
            self.num_epochs = num_epochs
            self.all_files = self._get_all_files(start_day)

        def __len__(self):
            return len(self.all_files) // (NUM_INPUT_FRAMES + NUM_TARGET_FRAMES)

        def _get_all_files(self, start_day):
            all_files = []
            day_folders = sorted([d for d in os.listdir(self.local_folder_path) if d >= start_day])
            for day_folder in day_folders:
                day_folder_path = os.path.join(self.local_folder_path, day_folder)
                if os.path.isdir(day_folder_path):
                    all_files.extend(sorted(os.path.join(day_folder_path, f) for f in os.listdir(day_folder_path)))
            return all_files

        def __getitem__(self, idx):
            start_idx = idx * (NUM_INPUT_FRAMES + NUM_TARGET_FRAMES)
            frame_paths = self.all_files[start_idx : start_idx + (NUM_INPUT_FRAMES + NUM_TARGET_FRAMES)]
            frames = [self._load_frame(fp) for fp in frame_paths]
            input_frames = np.stack(frames[:NUM_INPUT_FRAMES])
            target_frames = np.stack(frames[NUM_INPUT_FRAMES:])
            return torch.from_numpy(input_frames), torch.from_numpy(target_frames)

        @staticmethod
        def _load_frame(file_path):
            with NC4Dataset(file_path, "r") as nc_data:
                return np.ma.filled(nc_data.variables["RRdata"][:], 0)

    def initialize_model(self, model_name):
        model = DGMR(
            forecast_steps=18,
            input_channels=1,
            output_shape=256,
            latent_channels=768,
            context_channels=384,
            num_samples=3,
            visualize=True,
        )
        checkpoint_path = os.path.join(
            Config.ROOT_DIR, f"/work/pi_mzink_umass_edu/SPRITE/skillful_nowcasting/output/models/{model_name}.ckpt"
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

    def create_data_loader(self, split, num_epochs, start_day, batch_size):
        dataset = self.NetCDFDataset(split, num_epochs, start_day)
        return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))

    def process_batches(self):
        with torch.no_grad():
            for _idx, (inputs, targets) in enumerate(self.data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                break  # To process only one batch, remove for full processing

    def test_prediction(self):
        with torch.no_grad():
            for _idx, (inputs, targets) in enumerate(self.data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                # Remove batch dimension
                observations = torch.cat([inputs, targets], dim=1)
                predictions = torch.cat([inputs, outputs], dim=1)

                return observations, predictions


# Example of using the class
if __name__ == "__main__":
    model = CASADGMR("DGMR-V1_20240426_2122", "20180908")
    observations, predictions = model.test_prediction()

    print(observations.shape, predictions.shape)
