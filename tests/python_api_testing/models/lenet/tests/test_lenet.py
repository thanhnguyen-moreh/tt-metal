from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import pytest
import tt_lib
import torch
from loguru import logger
from lenet_utils import load_torch_lenet, prepare_image
from tt.lenet import lenet5
from utility_functions_new import comp_pcc
from utility_functions_new import torch2tt_tensor


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_lenet_inference(
    pcc, mnist_sample_input, model_location_generator, reset_seeds
):
    num_classes = 10
    batch_size = 1
    with torch.no_grad():
        # Initialize the device
        device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
        tt_lib.device.InitializeDevice(device)
        tt_lib.device.SetDefaultDevice(device)


        # Initialize Torch model
        pt_model_path = model_location_generator("tt_dnn-models/LeNet/model.pt")
        torch_LeNet, _ = load_torch_lenet(pt_model_path, num_classes)

        # Initialize TT model
        tt_lenet = lenet5(num_classes, device, model_location_generator)

        image = prepare_image(mnist_sample_input)

        torch_output = torch_LeNet(image).unsqueeze(1).unsqueeze(1)
        _, torch_predicted = torch.max(torch_output.data, -1)

        tt_image = torch2tt_tensor(image, device, tt_lib.tensor.Layout.ROW_MAJOR)

        tt_output = tt_lenet(tt_image)
        tt_output = tt_output.cpu()
        tt_output = torch.Tensor(tt_output.data()).reshape(tt_output.shape())

        _, tt_predicted = torch.max(tt_output.data, -1)

        pcc_passing, pcc_output = comp_pcc(torch_output, tt_output, pcc)
        logger.info(f"Output {pcc_output}")
        assert pcc_passing, f"Model output does not meet PCC requirement {pcc}."

    tt_lib.device.CloseDevice(device)
