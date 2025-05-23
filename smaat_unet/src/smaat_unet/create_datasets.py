import h5py
import numpy as np
from tqdm import tqdm

from smaat_unet.root import ROOT_DIR


def create_dataset(input_length: int, image_ahead: int, rain_amount_thresh: float):
    """Create a dataset that has target images containing at least `rain_amount_thresh` (percent) of rain."""

    precipitation_folder = ROOT_DIR / "data" / "precipitation"
    with h5py.File(
        precipitation_folder / "RAD_NL25_RAC_5min_train_test_2016-2019.h5",
        "r",
    ) as orig_f:
        train_images = orig_f["train"]["images"]
        train_timestamps = orig_f["train"]["timestamps"]
        test_images = orig_f["test"]["images"]
        test_timestamps = orig_f["test"]["timestamps"]
        print("Train shape", train_images.shape)
        print("Test shape", test_images.shape)
        imgSize = train_images.shape[1]
        num_pixels = imgSize * imgSize

        filename = (
            precipitation_folder / f"train_test_2016-2019_input-length_{input_length}_img-"
            f"ahead_{image_ahead}_rain-threshold_{int(rain_amount_thresh * 100)}.h5"
        )

        with h5py.File(filename, "w") as f:
            train_set = f.create_group("train")
            test_set = f.create_group("test")
            train_image_dataset = train_set.create_dataset(
                "images",
                shape=(1, input_length + image_ahead, imgSize, imgSize),
                maxshape=(None, input_length + image_ahead, imgSize, imgSize),
                dtype="float32",
                compression="gzip",
                compression_opts=9,
            )
            train_timestamp_dataset = train_set.create_dataset(
                "timestamps",
                shape=(1, input_length + image_ahead, 1),
                maxshape=(None, input_length + image_ahead, 1),
                dtype=h5py.special_dtype(vlen=str),
                compression="gzip",
                compression_opts=9,
            )
            test_image_dataset = test_set.create_dataset(
                "images",
                shape=(1, input_length + image_ahead, imgSize, imgSize),
                maxshape=(None, input_length + image_ahead, imgSize, imgSize),
                dtype="float32",
                compression="gzip",
                compression_opts=9,
            )
            test_timestamp_dataset = test_set.create_dataset(
                "timestamps",
                shape=(1, input_length + image_ahead, 1),
                maxshape=(None, input_length + image_ahead, 1),
                dtype=h5py.special_dtype(vlen=str),
                compression="gzip",
                compression_opts=9,
            )

            origin = [[train_images, train_timestamps], [test_images, test_timestamps]]
            datasets = [
                [train_image_dataset, train_timestamp_dataset],
                [test_image_dataset, test_timestamp_dataset],
            ]
            for origin_id, (images, timestamps) in enumerate(origin):
                image_dataset, timestamp_dataset = datasets[origin_id]

                # Pre-calculate all valid indices that meet the rain threshold
                sequence_length = input_length + image_ahead
                valid_indices = []

                # Use vectorized operations to find valid indices
                for i in tqdm(range(sequence_length, len(images)), desc="Finding valid indices"):
                    rain_pixels = np.sum(images[i] > 0)
                    if rain_pixels >= num_pixels * rain_amount_thresh:
                        valid_indices.append(i)

                # Pre-allocate the final dataset size
                total_sequences = len(valid_indices)
                image_dataset.resize(total_sequences, axis=0)
                timestamp_dataset.resize(total_sequences, axis=0)

                # Batch process the sequences
                for idx, i in enumerate(tqdm(valid_indices, desc="Creating sequences")):
                    imgs = images[i - sequence_length : i]
                    timestamps_img = timestamps[i - sequence_length : i]
                    image_dataset[idx] = imgs
                    timestamp_dataset[idx] = timestamps_img


if __name__ == "__main__":
    print("Creating dataset with at least 20% of rain pixel in target image")
    create_dataset(input_length=12, image_ahead=6, rain_amount_thresh=0.2)
    print("Creating dataset with at least 50% of rain pixel in target image")
    create_dataset(input_length=12, image_ahead=6, rain_amount_thresh=0.5)
