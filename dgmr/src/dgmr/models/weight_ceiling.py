# Can be converted to a factory in case we want to use different weight functions

import torch


def weight_fn(y, precip_weight_cap=24.0):
    """
    Weight function for the grid cell loss.
    w(y) = max(y + 1, ceil)

    Args:
        y: Tensor of rainfall intensities.
        ceil: Custom ceiling for the weight function.

    Returns:
        Weights for each grid cell.
    """
    return torch.max(y + 1, torch.tensor(precip_weight_cap, device=y.device))
