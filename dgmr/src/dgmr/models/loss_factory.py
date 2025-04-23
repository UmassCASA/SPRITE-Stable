from enum import Enum
from typing import Optional, Callable
from torch import nn

from sprite_metrics.losses import GridCellLoss, ReconstructionLoss


class RegularizerType(Enum):
    GRID_CELL = "grid_cell"  # Default loss from skillful nowcasting paper
    RECONSTRUCTION = "rec"  # Charlotte's loss for better extreme values


class RegularizerLossFactory:
    """
    Factory class for creating regularizer loss modules.

    Notes:
        - "grid_cell" creates a GridCellLoss module - standard from skillful nowcasting paper (default)
        - "rec" creates a ReconstructionLoss module - Charlotte's loss for DGMR for better accounting of extreme values
    """

    @staticmethod
    def create_regularizer(
        regularizer_type: str = "grid_cell",  # Make grid_cell the default
        weight_fn: Optional[Callable] = None,
        precip_weight_cap: float = 24.0,
        **kwargs,
    ) -> nn.Module:
        """
        Create a regularizer loss module based on the specified type.

        Args:
            regularizer_type: Type of regularizer to create ("grid_cell" or "rec")
            weight_fn: Optional weight function for the loss
            precip_weight_cap: Cap for precipitation weights
            **kwargs: Additional arguments to pass to the loss module

        Returns:
            The specified loss module
        """
        try:
            reg_type = RegularizerType(regularizer_type.lower())
        except ValueError as err:
            raise ValueError(
                f"Unknown regularizer type: {regularizer_type}. Valid options are: {[t.value for t in RegularizerType]}"
            ) from err

        if reg_type == RegularizerType.RECONSTRUCTION:
            return ReconstructionLoss(weight_fn=weight_fn, precip_weight_cap=precip_weight_cap)
        else:  # Default to GRID_CELL
            return GridCellLoss(weight_fn=weight_fn, precip_weight_cap=precip_weight_cap)
