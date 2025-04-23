import numpy as np
from scipy.ndimage import gaussian_filter


class NormalizationStrategy:
    """Base class for normalization strategies"""

    def normalize(self, data):
        raise NotImplementedError


class Log1pNormalization(NormalizationStrategy):
    """Natural logarithm of (x + 1) normalization"""

    def normalize(self, data):
        return np.log1p(data)


class MinMaxNormalization(NormalizationStrategy):
    """Min-Max scaling to [0, 1] range"""

    def normalize(self, data):
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)


class LDCastLogNormalization(NormalizationStrategy):
    """Custom logarithmic transformation for precipitation data"""

    def __init__(self, threshold=0.1):
        self.threshold = threshold  # threshold in mm/h

    def normalize(self, data):
        result = np.zeros_like(data)
        mask_high = data >= self.threshold
        mask_low = data < self.threshold

        # Apply log10 transformation for values >= threshold
        result[mask_high] = np.log10(data[mask_high])

        # Apply constant value for values < threshold
        result[mask_low] = np.log10(0.02)

        return result


class GaussianSmoothingNormalization(NormalizationStrategy):
    """Applies Gaussian smoothing for antialiasing"""

    def __init__(self, sigma=0.5):
        self.sigma = sigma

    def normalize(self, data):
        # TODO: Check the shape of the data
        # If data is 4D (sequence, crops, height, width), apply smoothing to each 2D slice
        if len(data.shape) == 4:
            result = np.zeros_like(data)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    result[i, j] = gaussian_filter(data[i, j], sigma=self.sigma)
            return result
        # If data is 3D (sequence, height, width), apply smoothing to each 2D slice
        elif len(data.shape) == 3:
            result = np.zeros_like(data)
            for i in range(data.shape[0]):
                result[i] = gaussian_filter(data[i], sigma=self.sigma)
            return result
        # If data is 2D, apply smoothing directly
        else:
            return gaussian_filter(data, sigma=self.sigma)


class CompositeNormalization(NormalizationStrategy):
    """Combines multiple normalization strategies"""

    def __init__(self, strategies):
        self.strategies = strategies

    def normalize(self, data):
        result = data.copy()
        for strategy in self.strategies:
            result = strategy.normalize(result)
        return result


class DivideByConstantNormalization(NormalizationStrategy):
    """Divides input data by a constant value"""

    def __init__(self, constant=200):
        self.constant = constant

    def normalize(self, data):
        return data / self.constant


def get_normalization_strategy(strategy_name):
    """Factory function to create normalization strategy instances"""
    strategies = {
        "log1p": Log1pNormalization(),
        "minmax": MinMaxNormalization(),
        "ldcast_log": LDCastLogNormalization(),
        "gaussian": GaussianSmoothingNormalization(),
        "custom_log_minmax": CompositeNormalization([LDCastLogNormalization(), MinMaxNormalization()]),
        "log1p_minmax": CompositeNormalization([Log1pNormalization(), MinMaxNormalization()]),
        "ldcast_log_gaussian": CompositeNormalization([LDCastLogNormalization(), GaussianSmoothingNormalization()]),
        "divide_by_200": DivideByConstantNormalization(),
    }
    return strategies.get(strategy_name)
