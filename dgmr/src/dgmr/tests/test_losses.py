"""Test loss functions"""

import torch
import pytest

from CASADGMR import CASADGMR
from sprite_metrics.losses import (
    SSIMLoss,
    MS_SSIMLoss,
    SSIMLossDynamic,
    tv_loss,
    PrecipitationCRPS,
    compute_csi,
    compute_psd,
)


@pytest.fixture(scope="session")
def data():
    model = CASADGMR("DGMR-V1_20240426_2122", "20180908")
    observations, predictions = model.test_prediction()
    return {
        "Random Different Inputs": (torch.randn(1, 22, 1, 256, 256) * 100, torch.randn(1, 22, 1, 256, 256) * 0.1),
        "Random Same Inputs": (torch.ones(1, 22, 1, 256, 256), torch.ones(1, 22, 1, 256, 256)),
        "CASA Different Inputs": (observations, predictions),
        "CASA Same Inputs": (observations, observations.clone()),
    }


@pytest.mark.parametrize(
    "label", ["Random Different Inputs", "Random Same Inputs", "CASA Different Inputs", "CASA Same Inputs"]
)
def test_csi(data, label):
    predictions, observations = data[label]
    threshold = 0.5
    csi_score = compute_csi(predictions, observations, threshold)
    print(f"\nCSI result for {label}: {csi_score}")
    assert torch.is_tensor(csi_score), f"{label} – CSI computation failed"


@pytest.mark.parametrize(
    "label", ["Random Different Inputs", "Random Same Inputs", "CASA Different Inputs", "CASA Same Inputs"]
)
def test_psd(data, label):
    predictions, observations = data[label]
    threshold = 0.5
    psd_score = compute_psd(predictions, observations, threshold)
    print(f"\nPSD result for {label}: {psd_score}")
    assert torch.is_tensor(psd_score), f"{label} – PSD computation failed"


@pytest.mark.parametrize(
    "label", ["Random Different Inputs", "Random Same Inputs", "CASA Different Inputs", "CASA Same Inputs"]
)
def test_crps_logistic(data, label):
    predictions, observations = data[label]
    crps_logistic = PrecipitationCRPS(method="logistic", sigma=3)

    crps_score_global = crps_logistic.compute_crps_global(predictions, observations)
    crps_score_local = crps_logistic.compute_crps_local(predictions, observations, 4)

    print(f"\nCRPS(logistic) result for {label}:\n\t{crps_score_global}(global), {crps_score_local}(local)")
    assert torch.is_tensor(crps_score_global), f"{label} – CRPS global computation failed"
    assert torch.is_tensor(crps_score_local), f"{label} – CRPS local computation failed"


@pytest.mark.parametrize(
    "label", ["Random Different Inputs", "Random Same Inputs", "CASA Different Inputs", "CASA Same Inputs"]
)
def test_crps_gaussian(data, label):
    predictions, observations = data[label]
    crps_gaussian = PrecipitationCRPS(method="Gaussian", sigma=3.0)

    crps_score_global = crps_gaussian.compute_crps_global(predictions, observations)
    crps_score_local = crps_gaussian.compute_crps_local(predictions, observations, 4)

    print(f"\nCRPS(gaussian) result for {label}:\n\t{crps_score_global}(global), {crps_score_local}(local)")

    assert torch.is_tensor(crps_score_global), f"{label} – CRPS global computation failed"
    assert torch.is_tensor(crps_score_local), f"{label} – CRPS local computation failed"


@pytest.mark.parametrize(
    "label", ["Random Different Inputs", "Random Same Inputs", "CASA Different Inputs", "CASA Same Inputs"]
)
def test_forward(data, label):
    predictions, observations = data[label]

    crps_logistic = PrecipitationCRPS()
    results = crps_logistic(predictions, observations)

    assert torch.is_tensor(results), f"{label} – CRPS forward computation failed"


# def test_crps_NRG(setup_data):
#     for predictions, observations, label in setup_data:
#         crps_score_NRG = metrics.compute_crps_NRG(observations, predictions)
#         assert torch.is_tensor(crps_score_NRG), f"{label} – CRPS NRG computation failed"

# def test_crps_PWM(setup_data):
#     for predictions, observations, label in setup_data:
#         crps_score_PWM = metrics.compute_crps_PWM(observations, predictions, sample_weight)
#         assert torch.is_tensor(crps_score_PWM), f"{label} – CRPS PWM computation failed"


def test_ssim_loss():
    x = torch.rand((2, 3, 32, 32))
    y = torch.rand((2, 3, 32, 32))

    loss = SSIMLoss()
    assert float(loss(x=x, y=x)) == 0
    assert float(loss(x=x, y=y)) != 0

    loss = SSIMLoss(convert_range=True)
    assert float(loss(x=x, y=y)) != 0


def test_ms_ssim_loss():
    x = torch.rand((2, 3, 256, 256))
    y = torch.rand((2, 3, 256, 256))

    loss = MS_SSIMLoss()
    assert float(loss(x=x, y=x)) == 0
    assert float(loss(x=x, y=y)) != 0

    loss = MS_SSIMLoss(convert_range=True)
    assert float(loss(x=x, y=y)) != 0


def test_ssim_loss_dynamic():
    x = torch.rand((2, 3, 256, 256))
    y = torch.rand((2, 3, 256, 256))
    curr_image = torch.rand((2, 3, 256, 256))

    loss = SSIMLossDynamic()
    assert float(loss(x=x, y=x, curr_image=curr_image)) == 0
    assert float(loss(x=x, y=y, curr_image=curr_image)) != 0

    loss = SSIMLossDynamic(convert_range=True)
    assert float(loss(x=x, y=y, curr_image=curr_image)) != 0


def test_tv_loss():
    x = torch.ones((2, 3, 256, 256))
    x[0, 0, 0, 0] = 2.5

    assert float(tv_loss(img=x, tv_weight=2)) == 2 * (1.5**2 + 1.5**2)
