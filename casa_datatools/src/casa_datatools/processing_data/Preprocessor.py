import os
import numpy as np
import logging
from netCDF4 import Dataset
from multiprocessing import Pool
import inspect
from functools import wraps
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
from .normalization import get_normalization_strategy


# Configure the primary logger
logger = logging.getLogger("MainLogger")
logger.setLevel(logging.DEBUG)
main_handler = logging.StreamHandler()
main_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(main_handler)

# Configure the nan_logger
path_to_log = "/work/pi_mzink_umass_edu/SPRITE/outputs/DATA/preprocessing_logs/nan_tracking.log"
os.makedirs(os.path.dirname(path_to_log), exist_ok=True)
nan_logger = logging.getLogger("NaNLogger")
nan_handler = logging.FileHandler(path_to_log)
nan_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
nan_logger.addHandler(nan_handler)
nan_logger.setLevel(logging.INFO)

SAVE_IMG = False
SAVE_NC = False
SAVE_SEQ = True

# Params example
params_example = {
    "saturation_constant": 1.0,
    "q_min": 2 * 10**-4,
    "multiplier": 0.1,
    "crop_size": (256, 256),
    "spatial_offset": 32,
    "data_shape": (24, 366, 350),
    "qn_threshold": 8 * 10**-3,
    "precip_cap": 128,
    "local_directory": "path/to/data_directory/",
    "new_directory": "path/to/output_directory/",
    "use_first_3days_as_validation": False,
}


def check_params(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)

        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        check_local_directory = bound_args.arguments["local_directory"]
        if not isinstance(check_local_directory, str):
            raise TypeError("Path must be an string!")
        if not Path(check_local_directory).exists():
            raise ValueError(f"Path: {check_local_directory} not exists!")

        check_new_directory = bound_args.arguments["output_directory"]
        if not Path(check_new_directory).exists():
            os.makedirs(check_new_directory)

        check_crop_size = bound_args.arguments["crop_size"]
        check_data_shape = bound_args.arguments["data_shape"][1:]
        data_and_crop_shape = zip(check_data_shape, check_crop_size)
        data_larger = all(data > crop for data, crop in data_and_crop_shape)
        if not data_larger:
            raise ValueError("Data size need to be larger than crop size!")
        # crop_number_match = len(set([d // c for d, c in data_and_crop_shape])) <= 1
        crop_number_match = len({d // c for d, c in data_and_crop_shape}) <= 1

        if not crop_number_match:
            raise ValueError("The number of horizontal and vertical crops does not match!")

        check_spatial_offset = bound_args.arguments["spatial_offset"]
        if not check_spatial_offset <= min(check_crop_size) // 2:
            raise ValueError("spatial offset should not be larger than crop size! ")

        return func(*args, **kwargs)

    return wrapper


def save_nc(filename, save_dir, resized_rr_data):
    save_path = os.path.join(save_dir, filename)

    with Dataset(save_path, "w", format="NETCDF4") as nc_data:
        # Create dimensions based on the resized data. Assuming resized_rr_data is (1, 256, 256)
        nc_data.createDimension("x0", 256)  # Adjusted to resized shape
        nc_data.createDimension("y0", 256)  # Adjusted to resized shape
        nc_data.createDimension("z0", 1)  # Keeping the original 'time' dimension as 'z0'

        # Create variables with dimensions. The dimensions are named to match the original file's structure.
        x = nc_data.createVariable("x0", "f4", ("x0",))
        y = nc_data.createVariable("y0", "f4", ("y0",))
        z = nc_data.createVariable("z0", "f4", ("z0",))
        rr = nc_data.createVariable("RRdata", "f4", ("z0", "y0", "x0"))

        x[:] = np.linspace(start=0, stop=1, num=256)  # Example placeholder values
        y[:] = np.linspace(start=0, stop=1, num=256)  # Example placeholder values
        z[:] = np.array([0])  # Keeping a placeholder if you don't have a specific value for 'z0'
        rr[:] = resized_rr_data


def save_img(filename, save_dir, resized_rr_data):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, filename)
    plt.imsave(save_path + ".png", resized_rr_data, cmap="gray")


class Preprocessor:
    @check_params
    def __init__(
        self,
        local_directory,
        output_directory,
        saturation_constant,
        q_min,
        multiplier,
        crop_size,
        data_shape,
        spatial_offset=0,
        num_workers=16,
        qn_threshold=1,
        use_first_3days_as_validation=False,
        validation_directory=None,
        precip_cap=203.2,
        normalization_strategy=None,
    ):
        # Paths
        self.local_directory = local_directory
        self.output_directory = output_directory
        self.validation_directory = validation_directory  # processed validation
        self.num_workers = num_workers

        # Selecting
        self.s = saturation_constant
        self.q_min = q_min
        self.m = multiplier
        self.crop_size = crop_size  # tuple (h, w)
        self.spatial_offset = spatial_offset
        self.data_shape = data_shape  # tuple (T, H, W)
        self.qn_threshold = qn_threshold
        self.use_first_3days_as_validation = use_first_3days_as_validation
        self.precip_cap = precip_cap

        # Normalization
        self.normalizer = get_normalization_strategy(normalization_strategy) if normalization_strategy else None

    def setup_process(self):
        """
        Prepares data buckets for multiprocessing by organizing netCDF files into sequences.
        Also runs renames sequence folders after processing.
        """

        day_folders = sorted(os.listdir(self.local_directory))
        args_bucket = []  # To hold all argument sequences

        if self.use_first_3days_as_validation:
            # Extract first day of each month and remove them from the main list
            # first_days_of_month_folders = self.__get_first_days_of_month(day_folders)

            # Extract first three of each month and remove them from the main list
            first_days_of_month_folders = self.__get_first_three_days_of_month(day_folders)

            day_folders = [day for day in day_folders if day not in first_days_of_month_folders]

            # Create validation buckets
            args_bucket_validation = self.__create_arg_bucket(
                self.local_directory, self.validation_directory, first_days_of_month_folders
            )
            args_bucket.extend(args_bucket_validation)

        # Create training buckets
        args_bucket_train = self.__create_arg_bucket(self.local_directory, self.output_directory, day_folders)
        args_bucket.extend(args_bucket_train)

        # Process all buckets
        with Pool(self.num_workers) as p:
            p.map(self.process, args_bucket)

        logger.info(f"Finished processing {self.local_directory}.")

        # Rename sequence folders
        self.__rename_sequence_folders(self.output_directory)
        if self.use_first_3days_as_validation:
            self.__rename_sequence_folders(self.validation_directory)

    def __get_first_days_of_month(self, day_folders):
        first_days_of_month = []
        current_month = None

        for day in day_folders:
            day_date = datetime.strptime(day, "%Y%m%d")
            if current_month != day_date.month:
                first_days_of_month.append(day)
                current_month = day_date.month

        return first_days_of_month

    def __get_first_three_days_of_month(self, day_folders):
        first_three_days_of_month = []
        current_month = None
        days_found = 0

        for day in day_folders:
            day_date = datetime.strptime(day, "%Y%m%d")
            if current_month != day_date.month:
                current_month = day_date.month
                days_found = 0

            if days_found < 3:
                first_three_days_of_month.append(day)
                days_found += 1

        return first_three_days_of_month

    def __create_arg_bucket(self, src_dir, dst_dir, day_folders):
        """
        Creates array of arguments for each sequence of frames.
        """
        args = []
        args_bucket = []
        counter = 0  # Counter to track sequences

        for day in day_folders:
            files = os.listdir(os.path.join(src_dir, day))
            for filename in files:
                args.append((self.local_directory, day, filename, dst_dir))
                if args.__len__() >= self.data_shape[0]:
                    args_bucket.append((counter, args))
                    counter += 1
                    args = []

        return args_bucket

    def __extract_rr_data(self, file_path):
        try:
            with Dataset(file_path, "r") as nc_data:
                # Extract the frame and remove the leading dimension
                rr_data = nc_data.variables["RRdata"][:]  # 366, 350, 1
                # x_data = nc_data.variables["x0"][:]
                # y_data = nc_data.variables["y0"][:]

            # return self.__check_rr_data_0s(rr_data, file_path)
            return self.__check_rr_data(rr_data, file_path)

        except Exception as e:
            logger.error(f"Error reading file: {file_path}")
            logger.error(e)
            return None

    def __check_rr_data_0s(self, rr_data, file_path):
        if not isinstance(rr_data, np.ndarray):
            raise TypeError("rr_data must be a numpy ndarray.")

        if rr_data.shape[1:] != self.data_shape[1:]:
            logger.error(f"rr_data_sequence shape {rr_data.shape} does not match expected shape {self.data_shape}.")

        if isinstance(rr_data, np.ma.MaskedArray):
            nan_logger.info(f"Masked Array values found in {file_path}")
            np.ma.filled(rr_data, 0)

        # Clip values to cap the maximum rate at 128 and raise the minimum to 0
        # rr_data = np.clip(rr_data, 0, 128)
        rr_data = np.clip(rr_data, 0, self.precip_cap)

        # Quantize in increments of 1/32
        rr_data = np.round(rr_data * 32) / 32

        # Check for NaN values
        if np.isnan(rr_data).any():
            nan_logger.info(f"NaN values found in {file_path}")
            np.nan_to_num(rr_data)

        return rr_data

    def __check_rr_data(self, rr_data, file_path):
        if not isinstance(rr_data, np.ndarray):
            raise TypeError("rr_data must be a numpy ndarray.")

        # Check if the shape matches the expected data shape
        if rr_data.shape[1:] != self.data_shape[1:]:
            logger.error(f"rr_data_sequence shape {rr_data.shape} does not match expected shape {self.data_shape}.")

        # Replace values below 0 with 0
        rr_data[rr_data < 0] = 0

        # Replace masked array values with -1
        if isinstance(rr_data, np.ma.MaskedArray):
            nan_logger.info(f"Masked Array values found in {file_path}")
            rr_data = np.ma.filled(rr_data, -1)  # Replace masked values with -1

        # Replace NaN values with -1
        if np.isnan(rr_data).any():
            nan_logger.info(f"NaN values found in {file_path}")
            rr_data = np.nan_to_num(rr_data, nan=-1)  # Replace NaN with -1

        # Clip valid precipitation values between 0 and precip_cap
        valid_mask = rr_data >= 0  # Identify valid precipitation data
        rr_data[valid_mask] = np.clip(rr_data[valid_mask], 0, self.precip_cap)

        # Quantize valid precipitation values to increments of 1/32
        rr_data[valid_mask] = np.round(rr_data[valid_mask] * 32) / 32

        return rr_data

    def process(self, args_bucket):
        """
        Unpacks arguments. Extracts frames. Runs importance sampling.
        Saves images/nc files under days/nc files under sequences.
        """

        counter, args_seq = args_bucket
        rr_data_sequence = []

        # Get Arguments and extract frames
        for arg_frame in args_seq:
            local_directory, day, filename, new_directory = arg_frame
            file_path = os.path.join(local_directory, day, filename)
            logger.debug(f"processing: {file_path}")
            rr_data_read = self.__extract_rr_data(file_path)

            if rr_data_read is not None:
                rr_data_sequence.append(rr_data_read)

        rr_data_sequence = np.array(rr_data_sequence)
        rr_data_sequence = np.squeeze(rr_data_sequence)

        # Importance Sampling
        try:
            result = self.filter_by_acceptance_probability(rr_data_sequence)
            if result is None:
                logger.warning(f"No valid data after filtering for counter {counter}")
                return

            filtered_rr_data_sequence, qn = result
            if filtered_rr_data_sequence is None:
                logger.warning(f"Empty filtered_rr_data_sequence for counter {counter}")
                return

        except Exception as e:
            logger.error(e)
            raise

        # Apply normalization after importance sampling but before saving if strategy is set
        if self.normalizer is not None:
            filtered_rr_data_sequence = self.normalizer.normalize(filtered_rr_data_sequence)

        if SAVE_SEQ and qn is not None:
            seq_dir = f"seq-{self.data_shape[0]}-{counter}-{qn:.7f}"
            save_dir = os.path.join(new_directory, seq_dir)
            os.makedirs(save_dir, exist_ok=True)

        for i in range(filtered_rr_data_sequence.shape[0]):
            local_directory, day, filename, new_directory = args_seq[i]

            if SAVE_IMG or SAVE_NC:
                day_save_dir = os.path.join(new_directory, day)
                os.makedirs(day_save_dir, exist_ok=True)

            for j in range(filtered_rr_data_sequence.shape[1]):
                if SAVE_SEQ and qn is not None:
                    save_nc(filename, save_dir, np.array(filtered_rr_data_sequence[i, j]))
                    logger.info(f"Saved {filename} in {save_dir}")

                if SAVE_IMG:
                    filename += f"_crop{j}"
                    save_img(filename, day_save_dir, np.squeeze(filtered_rr_data_sequence[i, j]))

                if SAVE_NC:
                    save_nc(filename, day_save_dir, np.array(filtered_rr_data_sequence[i, j]))
                    logger.info(f"Saved {filename} in {day_save_dir}")

    def filter_by_acceptance_probability(self, rr_data_sequence):
        logger.debug(f"rr_data_sequence shape: {rr_data_sequence.shape}")
        if not rr_data_sequence.shape[0] > 0:
            return None
        if not rr_data_sequence.max() > 0:
            return None
        crops = self.__get_crops(rr_data_sequence, False)
        crop_xn_c = 1 - np.exp(-crops / self.s)
        crop_xn_c_sum = crop_xn_c.sum(axis=(0, 2, 3))
        c = self.crop_size[0] * self.crop_size[1] * rr_data_sequence.shape[0]

        qn = np.minimum(1, self.q_min + self.m * crop_xn_c_sum / c)

        logger.debug(f"Max qn :{qn.max()}, average qn :{qn.mean()}")

        selected_crops_index = np.where(qn >= self.qn_threshold)

        if not selected_crops_index[0].size > 0:
            return None

        if self.spatial_offset > 0:
            crops = self.__get_crops(rr_data_sequence, True)
            qn /= self.spatial_offset * self.spatial_offset

        filtered_rr_data_sequence = crops[:, selected_crops_index, :, :]

        # TODO: Save qn to finish the "unbiased importance-weighted" procedure (S(Xnl)/qnl)
        return filtered_rr_data_sequence, qn.mean()

    def __get_crops(self, rr_data_sequence, add_random_offset):
        h_starts = np.arange(0, self.data_shape[1] - self.crop_size[0] + 1, self.crop_size[0])
        w_starts = np.arange(0, self.data_shape[2] - self.crop_size[1] + 1, self.crop_size[1])

        crops = []
        for h in h_starts:
            for w in w_starts:
                if add_random_offset:
                    offset_h = np.random.randint(0, self.spatial_offset)
                    offset_w = np.random.randint(0, self.spatial_offset)
                    if h + self.crop_size[0] > self.data_shape[1]:
                        offset_h = -offset_h
                    if w + self.crop_size[1] > self.data_shape[2]:
                        offset_w = -offset_w
                    h += offset_h
                    w += offset_w
                crop = rr_data_sequence[:, h : h + self.crop_size[0], w : w + self.crop_size[1]]
                crops.append(crop)

        crops = np.transpose(np.array(crops), (1, 0, 2, 3))

        return crops

    def __rename_sequence_folders(self, root_dir):
        # List all folders
        folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

        # Sort folders based on their original numbering (the third number in the name)
        folders.sort(key=lambda x: int(x.split("-")[2]))

        # Iterate over sorted folders and rename them with new numbering
        for idx, old_name in enumerate(folders, start=1):
            parts = old_name.split("-")
            new_name = f"seq-{parts[1]}-{idx}-{parts[3]}"
            shutil.move(os.path.join(root_dir, old_name), os.path.join(root_dir, new_name))
            print(f"Renamed {old_name} to {new_name}")
