from torch.utils.data import DataLoader, RandomSampler
from pytorch_lightning import LightningDataModule
from train.flower.dataloader.net_cdf_dataset_fl import NetCDFDataset


class DGMRDataModule(LightningDataModule):
    """
    Example of LightningDataModule for NETCDF dataset.
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html

    pin_memory: use pinned memory and enables faster data transfer to CUDA-enabled GPUs.
    num_workers: setting it to 0 means that the main process will do the data loading.
    batch_size: 1 batch is 1 sequence of radar frames which is 22 frames in total.
    """

    def __init__(
        self, num_workers=0, pin_memory=True, batch_size=16, client_id=0, num_clients=1, data_shard=0, total_shards=1
    ):
        super().__init__()
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.num_sequences_per_epoch = 100
        self.client_id = client_id
        self.num_clients = num_clients
        self.data_shard = data_shard
        self.total_shards = total_shards

    def setup(self, stage=None):
        self.train_dataset = NetCDFDataset(
            split="train",
            num_clients=self.num_clients,
            client_id=self.client_id,
            num_input_frams=4,
            num_total_frams=22,
            data_shard=self.data_shard,
            total_shards=self.total_shards,
        )
        self.val_dataset = NetCDFDataset(
            split="validation",
            num_clients=self.num_clients,
            client_id=self.client_id,
            num_input_frams=4,
            num_total_frams=22,
        )
        self.test_dataset = NetCDFDataset(
            split="test", num_clients=self.num_clients, client_id=self.client_id, num_input_frams=4, num_total_frams=22
        )

    def train_dataloader(self):
        # Use RandomSampler to select only a limited number of sequences per epoch
        sampler = RandomSampler(self.train_dataset, replacement=False, num_samples=self.num_sequences_per_epoch)
        # Use the already initialized dataset
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return train_loader

    def val_dataloader(self):
        # Use the already initialized dataset
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,  # Data is sequential
        )
        return val_loader

    def test_dataloader(self):
        # Use the already initialized dataset
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,  # Data is sequential
        )
        return test_loader
