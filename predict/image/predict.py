import time
from models.aesthetics_scorer.generate import (
    AestheticScoreResult,
    generate_aesthetic_scores,
)

from models.kandinsky.constants import (
    KANDINSKY_2_1_MODEL_NAME,
    KANDINKSY_2_2_MODEL_NAME,
    KANDINSKY_2_1_SCHEDULER_CHOICES,
    LOAD_KANDINSKY_2_1,
    LOAD_KANDINSKY_2_2,
)
from models.kandinsky.generate import generate as generate_with_kandinsky
from models.kandinsky.generate import generate_2_2 as generate_with_kandinsky_2_2
from models.stable_diffusion.constants import (
    SD_MODEL_CHOICES,
    SD_MODEL_DEFAULT_KEY,
    SD_SCHEDULER_CHOICES,
    SD_SCHEDULER_DEFAULT,
)

from models.stable_diffusion.generate import generate as generate_with_sd
from models.nllb.translate import translate_text_set_via_api
from models.nllb.constants import TRANSLATOR_COG_URL
from models.swinir.upscale import upscale

from typing import List

from .classes import PredictOutput, PredictResult
from .constants import SIZE_LIST
from .setup import ModelsPack
from models.open_clip.main import (
    open_clip_get_embeds_of_images,
    open_clip_get_embeds_of_texts,
)
from pydantic import BaseModel, Field, validator
from shared.helpers import log_gpu_memory, return_value_if_in_list, wrap_text
from tabulate import tabulate


class PredictInput(BaseModel):
    prompt: str = Field(description="Input prompt.", default="")
    negative_prompt: str = Field(description="Input negative prompt.", default="")
    num_outputs: int = Field(
        description="Number of images to output. If the NSFW filter is triggered, you may get fewer outputs than this.",
        ge=1,
        le=10,
        default=1,
    )
    init_image_url: str = Field(
        description="Init image url to be used with img2img.",
        default=None,
    )
    mask_image_url: str = Field(
        description="Mask image url to be used with img2img.",
        default=None,
    )
    prompt_strength: float = Field(
        description="The strength of the prompt when using img2img, between 0-1. When 1, it'll essentially ignore the image.",
        ge=0,
        le=1,
        default=None,
    )
    num_inference_steps: int = Field(
        description="Number of denoising steps", ge=1, le=500, default=30
    )
    guidance_scale: float = Field(
        description="Scale for classifier-free guidance.", ge=1, le=20, default=7.5
    )
    model: str = Field(
        default=SD_MODEL_DEFAULT_KEY,
        description=f'Choose a model. Defaults to "{SD_MODEL_DEFAULT_KEY}".',
    )
    scheduler: str = Field(
        default=SD_SCHEDULER_DEFAULT,
        description=f'Choose a scheduler. Defaults to "{SD_SCHEDULER_DEFAULT}".',
    )
    seed: int = Field(
        description="Random seed. Leave blank to randomize the seed.", default=None
    )
    prompt_flores_200_code: str = Field(
        description="Prompt language code (FLORES-200). It overrides the language auto-detection.",
        default=None,
    )
    negative_prompt_flores_200_code: str = Field(
        description="Negative prompt language code (FLORES-200). It overrides the language auto-detection.",
        default=None,
    )
    prompt_prefix: str = Field(description="Prompt prefix.", default=None)
    negative_prompt_prefix: str = Field(
        description="Negative prompt prefix.", default=None
    )
    output_image_extension: str = Field(
        description="Output type of the image. Can be 'png' or 'jpeg' or 'webp'.",
        default="jpeg",
    )
    output_image_quality: int = Field(
        description="Output quality of the image. Can be 1-100.", default=90
    )
    image_to_upscale: str = Field(
        description="Input image for the upscaler (SwinIR).", default=None
    )
    process_type: str = Field(
        description="Choose a process type. Can be 'generate', 'upscale' or 'generate_and_upscale'. Defaults to 'generate'",
        default="generate",
    )
    width: int = Field(
        description="Width of output image.",
        default=512,
    )
    height: int = Field(
        description="Height of output image.",
        default=512,
    )
    translator_cog_url: str = Field(
        description="URL of the translator cog. If it's blank, TRANSLATOR_COG_URL environment variable will be used (if it exists).",
        default=TRANSLATOR_COG_URL,
    )
    skip_translation: bool = Field(
        description="Whether to skip translating the prompt and the negative prompt or not.",
        default=False,
    )
    skip_safety_checker: bool = Field(
        description="Whether to skip the safety checker or not.", default=False
    )

    @validator("model")
    def validate_model(cls, v):
        rest = []
        if LOAD_KANDINSKY_2_1:
            rest += [KANDINSKY_2_1_MODEL_NAME]
        if LOAD_KANDINSKY_2_2:
            rest += [KANDINKSY_2_2_MODEL_NAME]
        choices = SD_MODEL_CHOICES + rest
        return return_value_if_in_list(v, choices)

    @validator("scheduler")
    def validate_scheduler(cls, v):
        choices = SD_SCHEDULER_CHOICES + KANDINSKY_2_1_SCHEDULER_CHOICES
        return return_value_if_in_list(v, choices)

    @validator("height")
    def validate_height(cls, v: int, values):
        if values["process_type"] == "upscale":
            return v
        return return_value_if_in_list(
            v,
            SIZE_LIST,
        )

    @validator("width")
    def validate_width(cls, v: int, values):
        if values["process_type"] == "upscale":
            return v
        return return_value_if_in_list(
            v,
            SIZE_LIST,
        )

    @validator("output_image_extension")
    def validate_output_image_extension(cls, v):
        return return_value_if_in_list(v, ["png", "jpeg", "webp"])

    @validator("process_type")
    def validate_process_type(cls, v):
        return return_value_if_in_list(
            v, ["generate", "upscale", "generate_and_upscale"]
        )


def predict(
    input: PredictInput,
    models_pack: ModelsPack,
) -> PredictResult:
    process_start = time.time()
    print("//////////////////////////////////////////////////////////////////")
    print(f"⏳ Process started: {input.process_type} ⏳")
    log_gpu_memory(message="GPU status before inference")
    output_images = []
    nsfw_count = 0
    open_clip_embeds_of_images = None
    open_clip_embed_of_prompt = None
    saved_safety_checker = None

    if input.process_type == "generate" or input.process_type == "generate_and_upscale":
        t_prompt = input.prompt
        t_negative_prompt = input.negative_prompt
        if input.translator_cog_url is not None and input.skip_translation is False:
            [t_prompt, t_negative_prompt] = translate_text_set_via_api(
                text_1=input.prompt,
                flores_1=input.prompt_flores_200_code,
                text_2=input.negative_prompt,
                flores_2=input.negative_prompt_flores_200_code,
                translator_url=input.translator_cog_url,
                detector=models_pack.translator["detector"],
                label="Prompt & Negative Prompt",
            )
        prompt_is_translated = input.prompt is not None and t_prompt != input.prompt
        neg_prompt_is_translated = (
            input.negative_prompt is not None
            and t_negative_prompt != input.negative_prompt
        )

        generator_pipe = None
        if input.model == KANDINSKY_2_1_MODEL_NAME:
            generator_pipe = models_pack.kandinsky
        elif input.model == KANDINKSY_2_2_MODEL_NAME:
            generator_pipe = models_pack.kandinsky_2_2
        else:
            generator_pipe = models_pack.sd_pipes[input.model]

        if input.skip_safety_checker and hasattr(generator_pipe, "safety_checker"):
            saved_safety_checker = generator_pipe.safety_checker
            generator_pipe.safety_checker = None

        log_table = [
            ["Model", input.model],
            ["Width", input.width],
            ["Height", input.height],
            ["Steps", input.num_inference_steps],
            ["Guidance Scale", input.guidance_scale],
            ["Outputs", input.num_outputs],
            ["Scheduler", input.scheduler],
            ["Seed", input.seed],
            [
                "Init Image URL",
                wrap_text(input.init_image_url)
                if input.init_image_url is not None
                else None,
            ],
            [
                "Mask Image URL",
                wrap_text(input.mask_image_url)
                if input.mask_image_url is not None
                else None,
            ],
            ["Prompt Strength", input.prompt_strength],
        ]
        if prompt_is_translated:
            log_table.append(["Original Prompt", wrap_text(input.prompt)])
        log_table.append(["Final Prompt", wrap_text(t_prompt)])
        log_table.append(["Prompt Translated", prompt_is_translated])
        if neg_prompt_is_translated:
            log_table.append(["Original Neg. Prompt", wrap_text(input.negative_prompt)])
        log_table.append(["Final Neg. Prompt", wrap_text(t_negative_prompt)])
        log_table.append(["Neg. Prompt Translated", neg_prompt_is_translated])
        print(
            tabulate(
                [["🖼️  Generation 🟡", "Started"]] + log_table, tablefmt="double_grid"
            )
        )

        startTime = time.time()
        args = {
            "prompt": t_prompt,
            "negative_prompt": t_negative_prompt,
            "prompt_prefix": input.prompt_prefix,
            "negative_prompt_prefix": input.negative_prompt_prefix,
            "width": input.width,
            "height": input.height,
            "num_outputs": input.num_outputs,
            "num_inference_steps": input.num_inference_steps,
            "guidance_scale": input.guidance_scale,
            "init_image_url": input.init_image_url,
            "mask_image_url": input.mask_image_url,
            "prompt_strength": input.prompt_strength,
            "scheduler": input.scheduler,
            "seed": input.seed,
            "model": input.model,
            "pipe": generator_pipe,
        }

        if input.model == KANDINSKY_2_1_MODEL_NAME:
            generate_output_images, generate_nsfw_count = generate_with_kandinsky(
                **args,
                safety_checker=None
                if input.skip_safety_checker
                else models_pack.safety_checker,
            )
        elif input.model == KANDINKSY_2_2_MODEL_NAME:
            generate_output_images, generate_nsfw_count = generate_with_kandinsky_2_2(
                **args,
                safety_checker=None
                if input.skip_safety_checker
                else models_pack.safety_checker,
            )
        else:
            generate_output_images, generate_nsfw_count = generate_with_sd(**args)
        output_images = generate_output_images
        nsfw_count = generate_nsfw_count

        endTime = time.time()
        print(
            tabulate(
                [["🖼️  Generation 🟢", f"{round((endTime - startTime) * 1000)} ms"]]
                + log_table,
                tablefmt="double_grid",
            ),
        )

        start_open_clip_prompt = time.time()
        open_clip_embed_of_prompt = open_clip_get_embeds_of_texts(
            [t_prompt],
            models_pack.open_clip["model"],
            models_pack.open_clip["tokenizer"],
        )[0]
        end_open_clip_prompt = time.time()
        print(
            f"📜 Open CLIP prompt embedding in: {round((end_open_clip_prompt - start_open_clip_prompt) * 1000)} ms 📜"
        )

        if len(output_images) > 0:
            start_open_clip_image = time.time()
            open_clip_embeds_of_images = open_clip_get_embeds_of_images(
                output_images,
                models_pack.open_clip["model"],
                models_pack.open_clip["processor"],
            )
            end_open_clip_image = time.time()
            print(
                f"🖼️ Open CLIP image embeddings in: {round((end_open_clip_image - start_open_clip_image) * 1000)} ms - {len(output_images)} images 🖼️"
            )
        else:
            open_clip_embeds_of_images = []
            print(
                "🖼️ No non-NSFW images generated. Skipping Open CLIP image embeddings. 🖼️"
            )

    if input.process_type == "upscale" or input.process_type == "generate_and_upscale":
        startTime = time.time()
        if input.process_type == "upscale":
            upscale_output_image = upscale(input.image_to_upscale, models_pack.upscaler)
            output_images = [upscale_output_image]
        else:
            upscale_output_images = []
            for image in output_images:
                upscale_output_image = upscale(image, models_pack.upscaler)
                upscale_output_images.append(upscale_output_image)
            output_images = upscale_output_images
        endTime = time.time()
        print(f"⭐️ Upscaled in: {round((endTime - startTime) * 1000)} ms ⭐️")

    # Aesthetic Score
    s_aes = time.time()
    aesthetic_scores: List[AestheticScoreResult] = []
    for i, image in enumerate(output_images):
        aesthetic_score_result = generate_aesthetic_scores(
            img=image,
            rating_model=models_pack.aesthetics_scorer["rating_model"],
            artifacts_model=models_pack.aesthetics_scorer["artifact_model"],
            clip_processor=models_pack.open_clip["processor"],
            vision_model=models_pack.open_clip["model"].vision_model,
        )
        aesthetic_scores.append(aesthetic_score_result)
        print(
            f"🎨 Image {i+1} | Rating Score: {aesthetic_score_result.rating_score} | Artifact Score: {aesthetic_score_result.artifact_score}"
        )
    e_aes = time.time()
    print(f"🎨 Calculated aesthetic scores in: {round((e_aes - s_aes) * 1000)} ms")

    # Prepare output objects
    output_objects: List[PredictOutput] = []
    for i, image in enumerate(output_images):
        obj = PredictOutput(
            pil_image=image,
            target_quality=input.output_image_quality,
            target_extension=input.output_image_extension,
            open_clip_image_embed=open_clip_embeds_of_images[i]
            if open_clip_embeds_of_images is not None
            else None,
            open_clip_prompt_embed=open_clip_embed_of_prompt
            if open_clip_embed_of_prompt is not None
            else None,
            aesthetic_rating_score=aesthetic_scores[i].rating_score,
            aesthetic_artifact_score=aesthetic_scores[i].artifact_score,
        )
        output_objects.append(obj)

    result = PredictResult(
        outputs=output_objects,
        nsfw_count=nsfw_count,
    )
    process_end = time.time()

    if (
        saved_safety_checker is not None
        and generator_pipe is not None
        and hasattr(generator_pipe, "safety_checker")
    ):
        generator_pipe.safety_checker = saved_safety_checker

    print(f"✅ Process completed in: {round((process_end - process_start) * 1000)} ms ✅")
    print("//////////////////////////////////////////////////////////////////")

    return result
