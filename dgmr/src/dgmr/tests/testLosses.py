import torch
from sprite_metrics.losses import PrecipitationCRPS, compute_csi, compute_psd
import unittest


class TestPrecipitationMetrics(unittest.TestCase):
    def setUp(self):
        # assume predictions and observations are in shape of (batch_size, height, width)
        self.predictions = torch.randn(10, 256, 256)
        self.observations = torch.randn(10, 256, 256)
        # self.predictions = torch.ones(10, 256, 256)
        # self.observations = torch.ones(10, 256, 256)
        self.sample_weight = torch.ones(10)
        self.threshold = 0.5  # Selecting a threshold for binarization operations
        self.crps_logistic = PrecipitationCRPS(method="logistic", sigma=1.0)
        self.crps_gaussian = PrecipitationCRPS(method="Gaussian", sigma=1.0)

        if torch.cuda.is_available():
            self.predictions.cuda()
            self.observations.cuda()

    def test_csi(self):
        csi_score = compute_csi(self.predictions, self.observations, self.threshold)
        self.assertTrue(torch.is_tensor(csi_score))
        print(f"CSI result: {csi_score}")

    def test_psd(self):
        psd_score = compute_psd(self.predictions, self.observations, self.threshold)
        self.assertTrue(torch.is_tensor(psd_score))
        print(f"PSD result: {psd_score}")

    def test_crps_logistic(self):
        crps_score_global = self.crps_logistic.compute_crps_global(self.predictions, self.observations)
        self.assertTrue(torch.is_tensor(crps_score_global))

        crps_score_local = self.crps_logistic.compute_crps_local(self.predictions, self.observations, 4)
        self.assertTrue(torch.is_tensor(crps_score_local))
        print(f"CRPS(logistic) result: {crps_score_global}(global), {crps_score_local}(local)")

    def test_crps_gaussian(self):
        crps_score_global = self.crps_gaussian.compute_crps_global(self.predictions, self.observations)
        self.assertTrue(torch.is_tensor(crps_score_global))

        crps_score_local = self.crps_gaussian.compute_crps_local(self.predictions, self.observations, 4)
        self.assertTrue(torch.is_tensor(crps_score_local))
        print(f"CRPS(gaussian) result: {crps_score_global}(global), {crps_score_local}(local)")

    # def test_crps_NRG(self):
    #     crps_score_NRG = self.metrics.compute_crps_NRG(self.observations, self.predictions)
    #     self.assertTrue(torch.is_tensor(crps_score_NRG))
    #
    # def test_crps_PWM(self):
    #     crps_score_PWM = self.metrics.compute_crps_PWM(self.observations, self.predictions, self.sample_weight)
    #     self.assertTrue(torch.is_tensor(crps_score_PWM))

    def test_forward(self):
        results = self.crps_logistic(self.predictions, self.observations, self.threshold)
        self.assertIsInstance(results, dict)
        self.assertTrue(all(torch.is_tensor(score) for score in results.values()))


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
