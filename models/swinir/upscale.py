import glob
import os
import shutil
import tempfile
import torch
import shutil
import numpy as np
from collections import OrderedDict
import cv2
import tempfile
from shared.helpers import clean_folder
from .constants import DEVICE_SWINIR
from .helpers import get_image_pair, setup
import time
from PIL import Image
from typing import Any
import requests
from io import BytesIO


@torch.inference_mode()
@torch.cuda.amp.autocast()
def upscale(image: np.ndarray | Image.Image | str, upscaler: Any) -> Image.Image:
    if image is None:
        raise ValueError("Image is required for the upscaler.")

    args = upscaler["args"]
    pipe = upscaler["pipe"]
    output_image = None

    # check if image is a url and download it if sso
    if is_url(image):
        image = download_image(image)

    elif isinstance(image, Image.Image):
        image = np.array(image)

    # setup folder and path
    border, window_size = setup(args)
    test_results = OrderedDict()
    test_results["psnr"] = []
    test_results["ssim"] = []
    test_results["psnr_y"] = []
    test_results["ssim_y"] = []
    test_results["psnr_b"] = []
    # psnr, ssim, psnr_y, ssim_y, psnr_b = 0, 0, 0, 0, 0

    # read image
    img_lq, img_gt = get_image_pair(args, image)  # image to HWC-BGR, float32
    img_lq = np.transpose(
        img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1)
    )  # HCW-BGR to CHW-RGB
    img_lq = (
        torch.from_numpy(img_lq).float().unsqueeze(0).to(DEVICE_SWINIR)
    )  # CHW-RGB to NCHW-RGB

    # inference
    inf_start_time = time.time()

    with torch.no_grad():
        # pad input image to be a multiple of window_size
        _, _, h_old, w_old = img_lq.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[
            :, :, : h_old + h_pad, :
        ]
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[
            :, :, :, : w_old + w_pad
        ]
        output = pipe(img_lq)
        output = output[..., : h_old * args.scale, : w_old * args.scale]

    inf_end_time = time.time()
    print(
        f"-- Upscale - Inference in: {round((inf_end_time - inf_start_time) * 1000)} ms --"
    )

    save_start_time = time.time()
    # save image
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    # float32 to uint8
    output = (output * 255.0).round().astype(np.uint8)
    output_image = output
    save_end_time = time.time()
    print(
        f"-- Upscale - Image save in: {round((save_end_time - save_start_time) * 1000)} ms --"
    )

    start = time.time()
    imageRGB = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGBA)
    pil_image = Image.fromarray(imageRGB)
    end = time.time()
    print(f"-- Upscale - Array to PIL Image in: {round((end - start) * 1000)} ms --")
    return pil_image


def is_url(url: str) -> bool:
    return url.startswith("http://") or url.startswith("https://")


def download_image(url: str) -> np.array:
    start = time.time()
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to download image from: {url}")
    end = time.time()
    print(f"-- Upscale - Download image in: {round((end - start) * 1000)} ms --")

    # Convert the image from PIL format to numpy array
    image_rgb = np.array(Image.open(BytesIO(response.content)))

    # Convert from RGB to BGR for OpenCV compatibility
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    return image_bgr
