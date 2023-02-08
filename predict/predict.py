import time
import os

import torch

from models.stable_diffusion.generate import generate
from models.stable_diffusion.constants import (
    SD_MODEL_CHOICES,
    SD_SCHEDULER_DEFAULT,
    SD_SCHEDULER_CHOICES,
    SD_MODEL_DEFAULT_KEY,
)
from models.stable_diffusion.helpers import (
    png_image_to_bytes,
)
from models.nllb.translate import translate_text
from models.swinir.upscale import upscale
from models.swinir.helpers import get_args_swinir, define_model_swinir
from models.swinir.constants import TASKS_SWINIR, MODELS_SWINIR, DEVICE_SWINIR

from lingua import LanguageDetectorBuilder
from typing import List


class Predictor(BasePredictor):
    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Input prompt.", default=""),
        negative_prompt: str = Input(description="Input negative prompt.", default=""),
        width: int = Input(
            description="Width of output image.",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        height: int = Input(
            description="Height of output image.",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        num_outputs: int = Input(
            description="Number of images to output. If the NSFW filter is triggered, you may get fewer outputs than this.",
            ge=1,
            le=10,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=30
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        scheduler: str = Input(
            default=SD_SCHEDULER_DEFAULT,
            choices=SD_SCHEDULER_CHOICES,
            description="Choose a scheduler.",
        ),
        model: str = Input(
            default=SD_MODEL_DEFAULT_KEY,
            choices=SD_MODEL_CHOICES,
            description="Choose a model. Defaults to 'Stable Diffusion v1.5'.",
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed.", default=None
        ),
        prompt_flores_200_code: str = Input(
            description="Prompt language code (FLORES-200). It overrides the language auto-detection.",
            default=None,
        ),
        negative_prompt_flores_200_code: str = Input(
            description="Negative prompt language code (FLORES-200). It overrides the language auto-detection.",
            default=None,
        ),
        prompt_prefix: str = Input(description="Prompt prefix.", default=None),
        negative_prompt_prefix: str = Input(
            description="Negative prompt prefix.", default=None
        ),
        output_image_extension: str = Input(
            description="Output type of the image. Can be 'png' or 'jpeg' or 'webp'.",
            choices=["png", "jpeg", "webp"],
            default="jpeg",
        ),
        output_image_quality: int = Input(
            description="Output quality of the image. Can be 1-100.", default=90
        ),
        image_to_upscale: Path = Input(
            description="Input image for the upscaler (Swinir).", default=None
        ),
        process_type: str = Input(
            description="Choose a process type. Can be 'generate', 'upscale' or 'generate_and_upscale'. Defaults to 'generate'",
            choices=["generate", "upscale", "generate_and_upscale"],
            default="generate",
        ),
        translator_cog_url: str = Input(
            description="URL of the translator cog. If it's blank, TRANSLATOR_COG_URL environment variable will be used (if it exists).",
            default=None,
        ),
    ) -> dict[str, List[dict[str, str]] | int]:
        processStart = time.time()
        print("//////////////////////////////////////////////////////////////////")
        print(f"⏳ Process started: {process_type} ⏳")
        output_images = []
        nsfw_count = 0

        if process_type == "generate" or process_type == "generate_and_upscale":
            if translator_cog_url is None:
                translator_cog_url = os.environ.get("TRANSLATOR_COG_URL", None)

            t_prompt = prompt
            t_negative_prompt = negative_prompt
            if translator_cog_url is not None:
                [t_prompt, t_negative_prompt] = translate_text(
                    prompt,
                    prompt_flores_200_code,
                    negative_prompt,
                    negative_prompt_flores_200_code,
                    translator_cog_url,
                    self.detect_language,
                    "Prompt & Negative Prompt",
                )
            else:
                print("-- Translator cog URL is not set. Skipping translation. --")

            txt2img_pipe = None
            if model != SD_MODEL_DEFAULT_KEY:
                txt2img_pipe = self.txt2img_alt_pipes[model]
            else:
                txt2img_pipe = self.txt2img_pipe

            print(
                f"🖥️ Generating - Model: {model} - Width: {width} - Height: {height} - Steps: {num_inference_steps} - Outputs: {num_outputs} 🖥️"
            )
            startTime = time.time()
            generate_output_images, generate_nsfw_count = generate(
                t_prompt,
                t_negative_prompt,
                prompt_prefix,
                negative_prompt_prefix,
                width,
                height,
                num_outputs,
                num_inference_steps,
                guidance_scale,
                scheduler,
                seed,
                model,
                txt2img_pipe,
            )
            output_images = generate_output_images
            nsfw_count = generate_nsfw_count
            endTime = time.time()
            print(
                f"🖥️ Generated in {round((endTime - startTime) * 1000)} ms - Model: {model} - Width: {width} - Height: {height} - Steps: {num_inference_steps} - Outputs: {num_outputs} 🖥️"
            )

        if process_type == "upscale" or process_type == "generate_and_upscale":
            startTime = time.time()
            if process_type == "upscale":
                upscale_output_image = upscale(
                    image_to_upscale, self.upscaler_model, self.upscaler_args
                )
                output_images = [upscale_output_image]
            else:
                upscale_output_images = []
                for image in output_images:
                    upscale_output_image = upscale(
                        image, self.upscaler_model, self.upscaler_args
                    )
                    upscale_output_images.append(upscale_output_image)
                output_images = upscale_output_images
            endTime = time.time()
            print(f"⭐️ Upscaled in: {round((endTime - startTime) * 1000)} ms ⭐️")

        # Prepare output objects
        output_objects = []
        output_len = len(output_images)
        for i, image in enumerate(output_images):
            start_time_save = time.time()
            image_bytes = png_image_to_bytes(image)
            obj = {
                "image_bytes": image_bytes,
                "target_quality": output_image_quality,
                "target_extension": output_image_extension,
            }
            output_objects.append(obj)
            end_time_save = time.time()
            print(
                f"-- Image {i+1}/{output_len} converted to bytes in: {round((end_time_save - start_time_save) * 1000)} ms --"
            )

        result = {
            "outputs": output_objects,
            "nsfw_count": nsfw_count,
        }
        processEnd = time.time()
        print(
            f"✅ Process completed in: {round((processEnd - processStart) * 1000)} ms ✅"
        )
        print("//////////////////////////////////////////////////////////////////")

        return result
