import os
import matplotlib.pyplot as plt
from dgmr import DGMR
import torch
import torch.nn.functional as F


def plot_data_and_save(data, title, directory="./OUTPUT"):
    """
    This function plots the data and saves the plots in the specified directory.
    Each slice along the time axis is plotted as a subplot.
    Args:
    - data (torch.Tensor): The data to be plotted. Shape should be (batch_size, time_steps, channels, height, width).
    - title (str): Title of the plot.
    - directory (str): Directory where to save the plots.
    """
    batch_size, time_steps, channels, height, width = data.shape

    # Create a figure
    plt.figure(figsize=(time_steps * 4, 4))  # Adjust the figure size as needed

    # Loop through each time step
    for i in range(time_steps):
        plt.subplot(1, time_steps, i + 1)
        plt.imshow(data[0, i, 0].cpu().detach().numpy(), cmap="gray")
        # Individual title for each subplot
        individual_title = f"{title} - Time {i + 1}"
        plt.title(individual_title)
        plt.axis("off")

    # Save the plot in the specified directory
    save_path = os.path.join(directory, f"{title.replace(' ', '_')}.png")
    plt.savefig(save_path)
    plt.close()  # Close the plot to free up memory

    print(f"Plot saved as {save_path}")


# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Initialize your model with the specified parameters
# Make sure these parameters match the ones used during training
model = DGMR(
    forecast_steps=4,
    input_channels=1,
    output_shape=256,  # Adjusted to match the trained model's expectation
    latent_channels=768,  # Adjusted accordingly
    context_channels=384,  # Adjusted accordingly
    num_samples=3,
).to(device)

# Load your trained model weights from the checkpoint
checkpoint_path = "./TRAINED_MODEL/best.ckpt"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["state_dict"])

print("Model loaded successfully")

# Create input data and move it to the same device as the model
# Ensure the input size matches the model's expectations
x = torch.rand((2, 4, 1, 256, 256)).to(device)  # Adjusted input size to 256x256
y = torch.rand((2, 4, 1, 256, 256)).to(device)  # Adjusted target size to 256x256

# Forward pass
out = model(x)

# Compute loss
loss = F.mse_loss(y, out)
loss.backward()


# Plot and save the input data
plot_data_and_save(x, "Input Data")

# Plot and save the model's output
plot_data_and_save(out, "Model Output")

# Plot and save the target data
plot_data_and_save(y, "Target Data")

print(f"Loss: {loss.item()}")

# Use the visualizer step from the dgmr
# Input the real data
