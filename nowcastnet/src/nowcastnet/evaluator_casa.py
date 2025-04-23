# Third party imports
import torch

# Local imports
from nowcastnet.layers.entities.net_config import Configs
from nowcastnet.models.nowcastnet_pytorch_lightning import Net
from casa_datatools.CASADataModule import CASADataModule

batch_size = 16
datamodule = CASADataModule(
    batch_size=batch_size,
)
datamodule.setup()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net_config = Configs()
model = Net(net_config).to(device)

checkpoint_path = "./pre-trained-model/mrms_model.ckpt"
checkpoint = torch.load(checkpoint_path, map_location=device)

if "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)
model.eval()

test_data_loader = datamodule.test_dataloader()

with torch.no_grad():
    test_inputs, _ = test_data_loader.dataset[0]
    test_inputs = torch.from_numpy(test_inputs)
    test_inputs = test_inputs.unsqueeze(1).to(device)  # Add channel dimension
    test_inputs = test_inputs.unsqueeze(0).to(device)  # Add batch dimension
    test_outputs = model(test_inputs.float())
    print(test_outputs.shape)
