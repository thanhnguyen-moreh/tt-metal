from pathlib import Path
import sys
import torch
import pytest
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../../..")

from python_api_testing.models.utility_functions_new import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
    comp_allclose,
    comp_pcc,
)
import tt_lib
from python_api_testing.models.swin.tt.swin_embeddings import TtSwinEmbeddings
from python_api_testing.models.swin.swin_utils import get_shape
from transformers import SwinModel


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_swin_embeddings_inference(imagenet_sample_input, pcc, reset_seeds):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)


    image = imagenet_sample_input
    base_address = f"embeddings"

    model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

    # Torch swinembedding
    torch_model = model.embeddings

    # Tt swinembedding
    tt_model = TtSwinEmbeddings(
        config=model.config,
        state_dict=model.state_dict(),
        base_address=base_address,
        device=device,
        host=host,
    )

    # Run torch model
    torch_output = torch_model(image)

    # Run tt model
    tt_image = tt_lib.tensor.Tensor(
        image.reshape(-1).tolist(),
        get_shape(image.shape),
        tt_lib.tensor.DataType.BFLOAT16,
        tt_lib.tensor.Layout.ROW_MAJOR,
    )

    tt_output = tt_model(tt_image)

    # Compare outputs
    tt_output_torch = tt_to_torch_tensor(tt_output[0])
    tt_output_torch = tt_output_torch.squeeze(0)
    does_pass, pcc_message = comp_pcc(torch_output[0], tt_output_torch, pcc)

    logger.info(comp_allclose(torch_output[0], tt_output_torch))
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)
    if does_pass:
        logger.info("SwinEmbedding Passed!")
    else:
        logger.warning("SwinEmbedding Failed!")

    assert does_pass
