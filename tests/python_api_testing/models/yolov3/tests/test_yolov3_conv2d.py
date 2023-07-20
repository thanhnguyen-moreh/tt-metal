import torch.nn as nn
import numpy as np
from loguru import logger
from pathlib import Path
import sys
import torch

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from python_api_testing.models.yolov3.reference.models.common import autopad
from python_api_testing.models.yolov3.reference.utils.dataloaders import LoadImages
from python_api_testing.models.yolov3.reference.utils.general import check_img_size
from python_api_testing.models.yolov3.reference.models.yolo import Conv
from python_api_testing.models.yolov3.reference.models.common import DetectMultiBackend
from python_api_testing.models.yolov3.tt.yolov3_conv2d import TtConv2D
import tt_lib

from utility_functions_new import (
    comp_allclose_and_pcc,
    comp_pcc,
    torch2tt_tensor,
    tt2torch_tensor,
)


def test_conv2d_module(model_location_generator):
    torch.manual_seed(1234)
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    INDEX = 0
    base_address = f"model.model.{INDEX}.conv"

    # Load yolo
    model_path = model_location_generator("tt_dnn-models/Yolo/models/")
    data_path = model_location_generator("tt_dnn-models/Yolo/data/")

    data_image_path = str(data_path / "images")
    data_coco = str(data_path / "coco128.yaml")
    model_config_path = str(data_path / "yolov3.yaml")
    weights_loc = str(model_path / "yolov3.pt")

    reference_model = DetectMultiBackend(
        weights_loc, device=torch.device("cpu"), dnn=False, data=data_coco, fp16=False
    )
    state_dict = reference_model.state_dict()
    torch_model = reference_model.model.model[INDEX].conv

    in_channels = torch_model.in_channels
    out_channels = torch_model.out_channels
    kernel_size = torch_model.kernel_size[0]
    stride = torch_model.stride[0]
    padding = torch_model.padding[0]
    groups = torch_model.groups
    dilation = torch_model.dilation[0]

    tt_model = TtConv2D(
        base_address=base_address,
        state_dict=state_dict,
        device=device,
        c1=in_channels,
        c2=out_channels,
        k=kernel_size,
        s=stride,
        p=padding,
        g=groups,
        d=dilation,
    )

    # Load data
    stride = max(stride, 32)  # model stride
    imgsz = check_img_size((640, 640), s=stride)  # check image size
    dataset = LoadImages(data_image_path, img_size=imgsz, stride=stride, auto=True)

    real_input = True

    # Run inference
    with torch.no_grad():
        if real_input:
            path, im, _, _, _ = next(iter(dataset))
            im = torch.from_numpy(im)
            im = im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]

        else:
            im = torch.rand(1, 3, 640, 640)

        # Inference- fused
        pt_out = torch_model(im)

        tt_im = torch2tt_tensor(im, device)
        tt_out = tt_model(tt_im)

    # Compare PCC
    tt_out = tt_out.cpu()
    tt_out = tt_out.to(tt_lib.tensor.Layout.ROW_MAJOR)

    tt_out = tt2torch_tensor(tt_out)
    tt_lib.device.CloseDevice(device)

    logger.debug(f"pt_out shape {pt_out.shape}")
    logger.debug(f"tt_out shape {tt_out.shape}")

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("Yolov3 TtConv2D Passed!")
    else:
        logger.warning("Yolov3 TtConv2D Failed!")

    assert does_pass
