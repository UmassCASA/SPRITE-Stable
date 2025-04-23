# Third party imports
import torch
import torch.nn as nn

# Local imports
from nowcastnet.layers.generation.module import LBlock


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, num_frames=10):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=1),  # [B, 64, T/2, H/2, W/2]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=1),  # [B, 128, T/4, H/4, W/4]
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=1),  # [B, 256, T/8, H/8, W/8]
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 1, kernel_size=(1, 1, 1)),  # [B, 1, T/8, H/8, W/8]
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: [batch_size, channels, frames, height, width]
        output = self.main(x)
        return output


class Temporal_Discriminator(nn.Module):
    def __init__(self, num_frames_input=4, num_frames_predict=10, T=24, H=256, W=256):
        super().__init__()

        self.num_frames_input = num_frames_input
        self.num_frames_predict = num_frames_predict
        self.T = T
        self.H = H
        self.W = W

        self.conv2d = nn.Conv2d(in_channels=self.T, out_channels=64, kernel_size=(9, 9), stride=(2, 2), padding=(4, 4))
        self.conv3d_1 = nn.Conv3d(
            in_channels=1, out_channels=4, kernel_size=(4, 9, 9), stride=(1, 2, 2), padding=(1, 4, 4)
        )
        self.conv3d_2 = nn.Conv3d(
            in_channels=1, out_channels=8, kernel_size=(4, 9, 9), stride=(1, 2, 2), padding=(1, 4, 4)
        )
        flattened_size = (self.num_frames_predict + 1) * 4 + (self.T - self.num_frames_predict + 1) * 8 + 64
        self.main = nn.Sequential(
            LBlock(flattened_size, 128),
            LBlock(128, 256),
            LBlock(256, 512),
            LBlock(512, 512),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: [batch_size, channels, frames, height, width]
        conv1 = self.conv2d(x.squeeze(1))
        conv1 = conv1.unsqueeze(1)
        conv2 = self.conv3d_1(x[:, :, -self.num_frames_predict - 2 :, :, :])
        conv3 = self.conv3d_2(x[:, :, : self.num_frames_input + 2, :, :])

        input_seq = torch.cat(
            [
                conv1.squeeze(1),
                conv2.flatten(start_dim=1, end_dim=2),
                conv3.flatten(start_dim=1, end_dim=2),
            ],
            dim=1,
        )

        output = self.main(input_seq)
        return output
