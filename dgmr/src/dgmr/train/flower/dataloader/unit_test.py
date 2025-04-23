import unittest
import numpy as np
from unittest.mock import patch
from net_cdf_dataset_fl import NetCDFDataset


class TestNetCDFDataset(unittest.TestCase):
    @patch("os.listdir")
    def setUp(self, mock_listdir):
        num_clients = 4
        client_id = 0
        data_shard = 0
        total_shards = 2

        self.dataset = NetCDFDataset(
            split="train",
            num_clients=num_clients,
            client_id=client_id,
            num_input_frams=4,
            num_total_frams=22,
            data_shard=data_shard,
            total_shards=total_shards,
        )

    def test_crop_and_resize(self):
        # Test the crop and resize function
        mock_image = np.ones((256, 256))

        # Verify that for client_id=0, the correct portion is cropped
        resized_image = self.dataset._crop_and_resize(mock_image, client_id=0)
        self.assertEqual(resized_image.shape, (1, 256, 256))
        self.assertTrue(np.all(resized_image == 1))

        # Check if the resize introduces any non-integer values (since original values are all 1)
        self.assertTrue(np.all(resized_image == 1))

    def test_invalid_image_size(self):
        # Test if ValueError is raised for invalid image size
        with self.assertRaises(ValueError):
            NetCDFDataset(
                split="train",
                num_input_frams=4,
                num_total_frams=6,
                client_id=0,
                num_clients=5,  # Not a perfect square
                data_shard=0,
                total_shards=2,
            )


if __name__ == "__main__":
    unittest.main()
