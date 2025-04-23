import os
import time
import multiprocessing
import gzip
import shutil
import pysftp
from config import Config
from paramiko.ssh_exception import SSHException


class DataDownloader:
    """
    Class to download and process CASA data.
    """

    def __init__(self, ssh_host, ssh_username, private_key, remote_dir, local_dir_base):
        self.ssh_host = ssh_host
        self.ssh_username = ssh_username
        self.private_key = private_key
        self.remote_dir = remote_dir
        self.local_dir_base = local_dir_base
        self.num_processes = multiprocessing.cpu_count()

    def safe_sftp_connection(self, retries=5, delay=2):
        """Establishes an SFTP connection with retries and exponential backoff."""
        cnopts = pysftp.CnOpts()
        cnopts.hostkeys = None
        for attempt in range(retries):
            try:
                return pysftp.Connection(
                    self.ssh_host, username=self.ssh_username, private_key=self.private_key, cnopts=cnopts
                )
            except SSHException as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                time.sleep(delay * (2**attempt))
                if attempt == retries - 1:
                    raise
        return None

    def download_files(self, tasks):
        """Download all files using a single persistent SFTP connection."""
        with self.safe_sftp_connection() as sftp:
            for remote_path_file, local_path_gz_file in tasks:
                try:
                    sftp.get(remote_path_file, local_path_gz_file)
                    print(f"Downloaded {remote_path_file}")
                except Exception as e:
                    print(f"Failed to download {remote_path_file}: {e}")

    def generate_unzip_tasks(self, split):
        """Generate the task list for already downloaded files."""
        local_directory_gz = os.path.join(self.local_dir_base, f"{split}_gz/")
        local_directory = os.path.join(self.local_dir_base, split)

        unzip_tasks = []

        for day_folder in os.listdir(local_directory_gz):
            local_path_gz_day = os.path.join(local_directory_gz, day_folder)
            local_path_day = os.path.join(local_directory, day_folder)

            if not os.path.isdir(local_path_gz_day):
                continue

            os.makedirs(local_path_day, exist_ok=True)

            for file_name in os.listdir(local_path_gz_day):
                if file_name.endswith(".gz"):
                    local_path_gz_file = os.path.join(local_path_gz_day, file_name)
                    local_path_file = os.path.join(local_path_day, os.path.splitext(file_name)[0])

                    unzip_tasks.append((local_path_gz_file, local_path_file))

        return unzip_tasks

    def unzip_existing_files(self, split):
        """Unzip all files in the specified split using multiprocessing."""
        unzip_tasks = self.generate_unzip_tasks(split)

        with multiprocessing.Pool(self.num_processes) as pool:
            pool.starmap(unzip_file, unzip_tasks)

        local_directory_gz = os.path.join(self.local_dir_base, f"{split}_gz/")
        shutil.rmtree(local_directory_gz)
        print(f"Finished unzipping all files for {split}")

    def download_and_process(self, split, start_day, end_day=""):
        local_directory_gz = os.path.join(self.local_dir_base, f"{split}_gz/")
        local_directory = os.path.join(self.local_dir_base, split)

        os.makedirs(local_directory, exist_ok=True)
        os.makedirs(local_directory_gz, exist_ok=True)

        download_tasks = []
        unzip_tasks = []

        with self.safe_sftp_connection() as sftp:
            sftp.chdir(self.remote_dir)
            day_folders = sftp.listdir()
            day_folders.sort()

            offset = 0
            for day in day_folders:
                if start_day <= day < (end_day if end_day else day + "1"):
                    remote_path_day = os.path.join(self.remote_dir, day)
                    local_path_gz_day = os.path.join(local_directory_gz, day)
                    local_path_day = os.path.join(local_directory, day)

                    os.makedirs(local_path_gz_day, exist_ok=True)
                    os.makedirs(local_path_day, exist_ok=True)

                    files = [f for f in sftp.listdir(remote_path_day) if f.endswith(".gz")]
                    number_of_frames = len(files)
                    frames_to_download_indices = [
                        (x * 5 + offset) % number_of_frames for x in range((number_of_frames + offset) // 5)
                    ]

                    for index in frames_to_download_indices:
                        file_name = files[index]
                        remote_path_file = os.path.join(remote_path_day, file_name)
                        local_path_gz_file = os.path.join(local_path_gz_day, file_name)
                        local_path_file = os.path.join(local_path_day, os.path.splitext(file_name)[0])

                        download_tasks.append((remote_path_file, local_path_gz_file))
                        unzip_tasks.append((local_path_gz_file, local_path_file))

                    offset = (offset + number_of_frames) % 5

        # Download all files
        self.download_files(download_tasks)

        # Unzip all files using multiprocessing
        with multiprocessing.Pool(self.num_processes) as pool:
            pool.starmap(unzip_file, unzip_tasks)

        shutil.rmtree(local_directory_gz)
        print(f"Finished downloading and processing {split}")


def unzip_file(local_path_gz_file, local_path_file):
    """Unzip a single downloaded file."""
    try:
        with gzip.open(local_path_gz_file, "rb") as f_in, open(local_path_file, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(local_path_gz_file)
    except Exception as e:
        print(f"Failed to unzip {local_path_gz_file}: {e}")


if __name__ == "__main__":
    downloader = DataDownloader(
        Config.CASA_SSH_HOST,
        Config.CASA_SSH_USERNAME,
        Config.CASA_PRIVATE_KEY,
        Config.REMOTE_DIR,
        Config.ORIG_DATA_DIR,
    )
    # Unzip existing train files
    # downloader.unzip_existing_files("train")

    # To continue with downloading and processing:
    # downloader.download_and_process("train", "20160301", "20221221")
    # downloader.download_and_process("test", "20221221", "20240501")

    downloader.download_and_process("train", "20160601", "20220601")
    downloader.download_and_process("test", "20220601", "20240601")
