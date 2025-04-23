from PIL import Image
import numpy as np


class PNGDataImporter:
    def __init__(self, **kwargs):
        # Store any necessary initialization parameters if needed
        self.importer_kwargs = kwargs

    def import_custom_png(self, filename):
        # Open the PNG file
        with Image.open(filename) as img:
            # Convert to NumPy array
            data = np.array(img)

            # Check if the data has more than 2 dimensions (indicating multiple channels)
            if data.ndim > 2:
                data = data[..., 0]  # Select the first channel

        # Assume the data unit is mm/h and the data is 8-bit grayscale
        precip_rate = data.astype(float) / 255.0 * 100.0  # Assume maximum precipitation rate is 100 mm/h

        # Quality information, can be set as needed
        quality = None

        # Metadata, can be set as needed
        metadata = {
            "unit": "mm/h",
            "transform": "dB",
            "zerovalue": 0.0,
            "threshold": 0.1,
            "projection": None,
            "x1": 0.0,
            "y1": 0.0,
            "x2": float(data.shape[1]),  # Image width
            "y2": float(data.shape[0]),  # Image height
            "accutime": 5,  # Assume data is accumulated every 5 minutes
            "yorigin": "upper",
            "institution": "CASA",
            "source": "Custom data",
        }

        return precip_rate, quality, metadata


# # Example usage:
# importer = PNGDataImporter()
#
# # Read time series data
# # Assuming a definition for io.read_timeseries which can handle the custom importer
# rainrate_field, quality, metadata = io.read_timeseries(
#     filenames, importer.import_custom_png, **importer.importer_kwargs
# )
#
# print("Shape after reading time series data:", rainrate_field.shape)
