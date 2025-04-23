import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from dgmr import DGMR
from datasets import load_dataset
from numpy.random import default_rng
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import torchvision


NUM_INPUT_FRAMES = 4
NUM_TARGET_FRAMES = 4


# Load real test data
class TFDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, split):
        super().__init__()
        self.reader = load_dataset("openclimatefix/nimrod-uk-1km", "sample", split=split, streaming=True)
        self.iter_reader = self.reader

    def __len__(self):
        return 1000

    def __getitem__(self, item):
        try:
            row = next(self.iter_reader)
        except Exception:
            rng = default_rng()
            self.iter_reader = iter(self.reader.shuffle(seed=rng.integers(low=0, high=100000), buffer_size=1000))
            row = next(self.iter_reader)
        input_frames, target_frames = extract_input_and_target_frames(row["radar_frames"])
        return np.moveaxis(input_frames, [0, 1, 2, 3], [0, 2, 3, 1]), np.moveaxis(
            target_frames, [0, 1, 2, 3], [0, 2, 3, 1]
        )


# Function to plot data
def plot_data_and_save(data, title, directory="./OUTPUT"):
    batch_size, time_steps, channels, height, width = data.shape
    plt.figure(figsize=(time_steps * 4, 4))
    for i in range(time_steps):
        plt.subplot(1, time_steps, i + 1)
        plt.imshow(data[0, i, 0].cpu().detach().numpy(), cmap="gray")
        plt.title(f"{title} - Time {i + 1}")
        plt.axis("off")
    save_path = os.path.join(directory, f"{title.replace(' ', '_')}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved as {save_path}")


def extract_input_and_target_frames(radar_frames):
    """Extract input and target frames from a dataset row's radar_frames."""
    # We align our targets to the end of the window, and inputs precede targets.
    input_frames = radar_frames[-NUM_TARGET_FRAMES - NUM_INPUT_FRAMES : -NUM_TARGET_FRAMES]
    target_frames = radar_frames[-NUM_TARGET_FRAMES:]
    return input_frames, target_frames


def visualize_step(
    tensorboard_writer,
    x: torch.Tensor,
    y: torch.Tensor,
    y_hat: torch.Tensor,
    batch_idx: int,
    step: str,
    input_channels: int,
) -> None:
    images = x[0].cpu().detach()
    future_images = y[0].cpu().detach()
    generated_images = y_hat[0].cpu().detach()
    for i, t in enumerate(images):
        t = [torch.unsqueeze(img, dim=0) for img in t]
        image_grid = torchvision.utils.make_grid(t, nrow=input_channels)
        tensorboard_writer.add_image(f"{step}/Input_Image_Stack_Frame_{i}", image_grid, global_step=batch_idx)
    for i, t in enumerate(future_images):
        t = [torch.unsqueeze(img, dim=0) for img in t]
        image_grid = torchvision.utils.make_grid(t, nrow=input_channels)
        tensorboard_writer.add_image(f"{step}/Target_Image_Stack_Frame_{i}", image_grid, global_step=batch_idx)
    for i, t in enumerate(generated_images):
        t = [torch.unsqueeze(img, dim=0) for img in t]
        image_grid = torchvision.utils.make_grid(t, nrow=input_channels)
        tensorboard_writer.add_image(f"{step}/Predicted_Image_Stack_Frame_{i}", image_grid, global_step=batch_idx)

    print("Visualized a batch of data")


# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Initialize and load model
model = DGMR(
    forecast_steps=4,
    input_channels=1,
    output_shape=512,
    latent_channels=768,
    context_channels=384,
    num_samples=3,
    visualize=True,  # Ensure this is True
).to(device)
checkpoint_path = "/work/pi_mzink_umass_edu/SPRITE/skillful_nowcasting/TRAINED_MODEL/best.ckpt"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["state_dict"])
print("Model loaded successfully")


# Create DataLoader for test data
test_data_loader = DataLoader(TFDataset(split="test"), batch_size=2)
print("Test data loaded successfully")

# Evaluate the model on real test data
model.eval()
print("Model set to evaluation mode")

# torch.Size([2, 4, 1, 512, 512])

tensorboard_writer = SummaryWriter("WORK/TensorBoard/v1")


with torch.no_grad():
    for batch_idx, (test_inputs, test_targets) in enumerate(test_data_loader):
        print("Processing a batch of test data")
        print(f"Input shape: {test_inputs.shape}")

        # Move data to device
        test_inputs = test_inputs.to(device)
        test_targets = test_targets.to(device)

        # Forward pass
        test_outputs = model(test_inputs)

        # Visualization
        # plot_data_and_save(test_inputs, "Real Test Input Data")
        # plot_data_and_save(test_outputs, "Model Output on Test Data")
        # plot_data_and_save(test_targets, "Real Test Target Data")

        # Visualization using modified visualize_step
        visualize_step(
            tensorboard_writer,
            test_inputs,
            test_targets,
            test_outputs,
            batch_idx,
            "test",
            model.input_channels,  # Assuming this is defined in your model
        )
        break  # Remove this line to process the entire test set

# Needs to be corrected for appropriate paths
