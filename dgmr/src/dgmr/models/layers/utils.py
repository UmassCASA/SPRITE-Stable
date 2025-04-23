import torch

from dgmr.models.layers import CoordConv


def get_conv_layer(conv_type: str = "standard") -> torch.nn.Module:
    if conv_type == "standard":
        conv_layer = torch.nn.Conv2d
    elif conv_type == "coord":
        conv_layer = CoordConv
    elif conv_type == "3d":
        conv_layer = torch.nn.Conv3d
    else:
        raise ValueError(f"{conv_type} is not a recognized Conv method")
    return conv_layer


def unify_order_of_magnitude(target_tensor, input_tensor):
    if target_tensor == 0 or input_tensor == 0:
        return input_tensor
    log_target = torch.log10(target_tensor)
    log_input = torch.log10(input_tensor)
    return (10 ** torch.round(log_target - log_input)) * input_tensor
