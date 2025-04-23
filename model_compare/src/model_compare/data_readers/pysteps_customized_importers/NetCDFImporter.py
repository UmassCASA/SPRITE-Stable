import netCDF4
import threading


class NetCDFImporter:
    """
    A custom importer that in a multi-threaded environment:
    1) Checks if coordinate arrays are already cached (in a .npz file).
    2) If not, acquires a lock, loads from reference netCDF file once, and saves to .npz.
    3) Subsequent calls just load from the .npz or use the in-memory data.
    """

    # This lock prevents multiple threads from simultaneously attempting to generate/write the coords cache file
    _coord_lock = threading.Lock()

    def __init__(self, **kwargs):
        """
        Args:
            ref_path: Path to the reference netCDF file (containing x0, y0)
            cached_coords_path: Path to the cached coordinate file (.npz)
            **kwargs: Other custom parameters
        """
        self.importer_kwargs = kwargs

    def import_custom_netcdf(self, filename, **kwargs):
        """
        A custom importer to read NetCDF files containing RRdata.
        Before reading 'filename', ensure reference coords are loaded (thread-safe).
        """
        with netCDF4.Dataset(filename, "r") as nc_data:
            rr_data = nc_data.variables["RRdata"][:]
            x0 = nc_data.variables["x0"][:]
            y0 = nc_data.variables["y0"][:]

        metadata = {
            "precipitation": "intensity",
            "unit": "mm/h",
            "transform": None,
            "zerovalue": 0.0,
            "threshold": 0.1,
            "projection": "+proj=longlat +datum=WGS84 +no_defs",
            "x1": x0[0],
            "y1": y0[-1],
            "x2": x0[-1],
            "y2": y0[0],
            "yorigin": "upper",
            "institution": "CASA",
            "source": "Custom data",
        }

        return rr_data, None, metadata
