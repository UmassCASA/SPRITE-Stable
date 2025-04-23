import os
import numpy as np
from netCDF4 import Dataset
import csv
from multiprocessing import Pool


def process_files(root_dir, output_csv):
    """
    Processes all .nc files in the directories under root_dir, extracting the top 3 maximum
    values from each of the first three frames in the files and saving the results
    to a CSV file with multiprocessing to accelerate the process.

    Args:
        root_dir (str): Root directory containing subdirectories of .nc files.
        output_csv (str): Path to the output CSV file.
    """
    file_paths = []
    for root, _dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".nc"):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)

    pool = Pool()  # Pool will use all available CPUs by default
    results = pool.map(extract_max_values, file_paths)
    pool.close()
    pool.join()

    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "max_value1", "max_value2", "max_value3"])
        for result in results:
            writer.writerow(result)


def extract_max_values(file_path):
    """
    Extracts the top 3 maximum values from the first three frames of the .nc file and returns
    the filename and its max values, handling potential NaN values.

    Args:
        file_path (str): Path to the .nc file.

    Returns:
        List[str, float, float, float]: List containing the filename and maximum values
        from the first three frames.
    """
    try:
        with Dataset(file_path, "r") as nc:
            var_names = list(nc.variables.keys())
            data_var = var_names[-1]  # Assuming the last variable is the data variable
            data = nc.variables[data_var][:3]  # Extract only the first three frames
            max_values = []
            for frame in data:
                sorted_values = np.sort(np.ravel(frame))[::-1]  # Flatten, sort descending
                top_3_max = sorted_values[:3]  # Top 3 values
                max_values.extend(top_3_max)
            return [os.path.basename(file_path)] + max_values
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return [os.path.basename(file_path), None, None, None]


# Usage
root_directory = "/work/pi_mzink_umass_edu/SPRITE/skillful_nowcasting/netCDFData_processed/"
output_csv_path = "/work/pi_mzink_umass_edu/SPRITE/skillful_nowcasting/output/max_values_casa_updated.csv"
process_files(root_directory, output_csv_path)
print("Completed")
