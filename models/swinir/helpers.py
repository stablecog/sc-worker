import argparse
from .network_swinir import SwinIR as net
from .constants import MODELS_SWINIR
import torch
import cv2
import numpy as np
from . import util_calculate_psnr_ssim
import os


def define_model_swinir(args):
    # 001 classical image sr
    if args.task == "classical_sr":
        model = net(
            upscale=args.scale,
            in_chans=3,
            img_size=args.training_patch_size,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="pixelshuffle",
            resi_connection="1conv",
        )
        param_key_g = "params"

    # 002 lightweight image sr
    # use 'pixelshuffledirect' to save parameters
    elif args.task == "lightweight_sr":
        model = net(
            upscale=args.scale,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6],
            embed_dim=60,
            num_heads=[6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="pixelshuffledirect",
            resi_connection="1conv",
        )
        param_key_g = "params"

    # 003 real-world image sr
    elif args.task == "real_sr":
        if not args.large_model:
            # use 'nearest+conv' to avoid block artifacts
            model = net(
                upscale=args.scale,
                in_chans=3,
                img_size=64,
                window_size=8,
                img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2,
                upsampler="nearest+conv",
                resi_connection="1conv",
            )
        else:
            # larger model size; use '3conv' to save parameters and memory; use ema for GAN training
            model = net(
                upscale=args.scale,
                in_chans=3,
                img_size=64,
                window_size=8,
                img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
                embed_dim=240,
                num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                mlp_ratio=2,
                upsampler="nearest+conv",
                resi_connection="3conv",
            )
        param_key_g = "params_ema"

    # 004 grayscale image denoising
    elif args.task == "gray_dn":
        model = net(
            upscale=1,
            in_chans=1,
            img_size=128,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="",
            resi_connection="1conv",
        )
        param_key_g = "params"

    # 005 color image denoising
    elif args.task == "color_dn":
        model = net(
            upscale=1,
            in_chans=3,
            img_size=128,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="",
            resi_connection="1conv",
        )
        param_key_g = "params"

    # 006 grayscale JPEG compression artifact reduction
    # use window_size=7 because JPEG encoding uses 8x8; use img_range=255 because it's sligtly better than 1
    elif args.task == "jpeg_car":
        model = net(
            upscale=1,
            in_chans=1,
            img_size=126,
            window_size=7,
            img_range=255.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="",
            resi_connection="1conv",
        )
        param_key_g = "params"

    # 006 color JPEG compression artifact reduction
    # use window_size=7 because JPEG encoding uses 8x8; use img_range=255 because it's sligtly better than 1
    elif args.task == "color_jpeg_car":
        model = net(
            upscale=1,
            in_chans=3,
            img_size=126,
            window_size=7,
            img_range=255.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="",
            resi_connection="1conv",
        )
        param_key_g = "params"

    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(
        pretrained_model[param_key_g]
        if param_key_g in pretrained_model.keys()
        else pretrained_model,
        strict=True,
    )

    return model


def setup(args):
    # 001 classical image sr/ 002 lightweight image sr
    if args.task in ["classical_sr", "lightweight_sr"]:
        save_dir = f"results/swinir_{args.task}_x{args.scale}"
        folder = args.folder_gt
        border = args.scale
        window_size = 8

    # 003 real-world image sr
    elif args.task in ["real_sr"]:
        save_dir = f"results/swinir_{args.task}_x{args.scale}"
        if args.large_model:
            save_dir += "_large"
        folder = args.folder_lq
        border = 0
        window_size = 8

    # 004 grayscale image denoising/ 005 color image denoising
    elif args.task in ["gray_dn", "color_dn"]:
        save_dir = f"results/swinir_{args.task}_noise{args.noise}"
        folder = args.folder_gt
        border = 0
        window_size = 8

    # 006 JPEG compression artifact reduction
    elif args.task in ["jpeg_car", "color_jpeg_car"]:
        save_dir = f"results/swinir_{args.task}_jpeg{args.jpeg}"
        folder = args.folder_gt
        border = 0
        window_size = 7

    return folder, save_dir, border, window_size


def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    # 001 classical image sr/ 002 lightweight image sr (load lq-gt image pairs)
    if args.task in ["classical_sr", "lightweight_sr"]:
        img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        img_lq = (
            cv2.imread(
                f"{args.folder_lq}/{imgname}x{args.scale}{imgext}", cv2.IMREAD_COLOR
            ).astype(np.float32)
            / 255.0
        )

    # 003 real-world image sr (load lq image only)
    elif args.task in ["real_sr"]:
        img_gt = None
        img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

    # 004 grayscale image denoising (load gt image and generate lq image on-the-fly)
    elif args.task in ["gray_dn"]:
        img_gt = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        np.random.seed(seed=0)
        img_lq = img_gt + np.random.normal(0, args.noise / 255.0, img_gt.shape)
        img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = np.expand_dims(img_lq, axis=2)

    # 005 color image denoising (load gt image and generate lq image on-the-fly)
    elif args.task in ["color_dn"]:
        img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        np.random.seed(seed=0)
        img_lq = img_gt + np.random.normal(0, args.noise / 255.0, img_gt.shape)

    # 006 grayscale JPEG compression artifact reduction (load gt image and generate lq image on-the-fly)
    elif args.task in ["jpeg_car"]:
        img_gt = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img_gt.ndim != 2:
            img_gt = util_calculate_psnr_ssim.bgr2ycbcr(img_gt, y_only=True)
        result, encimg = cv2.imencode(
            ".jpg", img_gt, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg]
        )
        img_lq = cv2.imdecode(encimg, 0)
        img_gt = np.expand_dims(img_gt, axis=2).astype(np.float32) / 255.0
        img_lq = np.expand_dims(img_lq, axis=2).astype(np.float32) / 255.0

    # 006 JPEG compression artifact reduction (load gt image and generate lq image on-the-fly)
    elif args.task in ["color_jpeg_car"]:
        img_gt = cv2.imread(path)
        result, encimg = cv2.imencode(
            ".jpg", img_gt, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg]
        )
        img_lq = cv2.imdecode(encimg, 1)
        img_gt = img_gt.astype(np.float32) / 255.0
        img_lq = img_lq.astype(np.float32) / 255.0

    return imgname, img_lq, img_gt


def get_args_swinir():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="real_sr",
        help="classical_sr, lightweight_sr, real_sr, " "gray_dn, color_dn, jpeg_car",
    )
    parser.add_argument(
        "--scale", type=int, default=1, help="scale factor: 1, 2, 3, 4, 8"
    )  # 1 for dn and jpeg car
    parser.add_argument("--noise", type=int, default=15, help="noise level: 15, 25, 50")
    parser.add_argument(
        "--jpeg", type=int, default=40, help="scale factor: 10, 20, 30, 40"
    )
    parser.add_argument(
        "--training_patch_size",
        type=int,
        default=128,
        help="patch size used in training SwinIR. "
        "Just used to differentiate two different settings in Table 2 of the paper. "
        "Images are NOT tested patch by patch.",
    )
    parser.add_argument(
        "--large_model",
        action="store_true",
        help="use large model, only provided for real image sr",
    )
    parser.add_argument(
        "--model_path", type=str, default=MODELS_SWINIR["real_sr"]["large"]
    )
    parser.add_argument(
        "--folder_lq",
        type=str,
        default=None,
        help="input low-quality test image folder",
    )
    parser.add_argument(
        "--folder_gt",
        type=str,
        default=None,
        help="input ground-truth test image folder",
    )

    return parser.parse_args("")
