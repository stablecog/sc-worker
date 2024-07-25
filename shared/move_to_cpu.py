import logging
import time
from models.stable_diffusion.constants import SD_MODELS
from predict.image.setup import ModelsPack
from models.constants import DEVICE_CPU, DEVICE_CUDA
from .helpers import (
    move_pipe_to_device,
)
from models.kandinsky.constants import (
    KANDINSKY_2_2_KEEP_IN_CPU_WHEN_IDLE,
    KANDINSKY_2_2_MODEL_NAME,
)


def move_other_models_to_cpu(
    main_model_name: str, main_model_pipe: str, models_pack: ModelsPack
):
    # Skip the move if the model is not to be kept in CPU when idle
    is_kandinsky_2_2 = main_model_name == KANDINSKY_2_2_MODEL_NAME
    sd_spec = SD_MODELS.get(main_model_name, None)
    if (is_kandinsky_2_2 and KANDINSKY_2_2_KEEP_IN_CPU_WHEN_IDLE is False) or (
        sd_spec is not None and sd_spec.get("keep_in_cpu_when_idle", False) is False
    ):
        logging.info(
            f"🎛️ 🔵 Skipping moving other models to {DEVICE_CPU} for -> {main_model_name}, {main_model_pipe}"
        )
        return

    model_count = 0
    s = time.time()
    logging.info(
        f"🎛️ 🟡 Moving other models to {DEVICE_CPU} for -> {main_model_name}, {main_model_pipe}"
    )

    # Move other Stable Diffusion models to CPU if needed
    for model_name, pipe_set in models_pack.sd_pipe_sets.items():
        if main_model_name != model_name and SD_MODELS[model_name].get(
            "keep_in_cpu_when_idle"
        ):
            if (
                pipe_set.text2img is not None
                and pipe_set.text2img.device.type == DEVICE_CUDA
            ):
                model_count += 1
                models_pack.sd_pipe_sets[model_name].text2img = move_pipe_to_device(
                    pipe=pipe_set.text2img,
                    model_name=f"{model_name} text2img",
                    device=DEVICE_CPU,
                )
            if (
                pipe_set.img2img is not None
                and pipe_set.img2img.device.type == DEVICE_CUDA
            ):
                model_count += 1
                models_pack.sd_pipe_sets[model_name].img2img = move_pipe_to_device(
                    pipe=pipe_set.img2img,
                    model_name=f"{model_name} img2img",
                    device=DEVICE_CPU,
                )
            if (
                pipe_set.inpaint is not None
                and pipe_set.inpaint.device.type == DEVICE_CUDA
            ):
                model_count += 1
                models_pack.sd_pipe_sets[model_name].inpaint = move_pipe_to_device(
                    pipe=pipe_set.inpaint,
                    model_name=f"{model_name} inpaint",
                    device=DEVICE_CPU,
                )
            if (
                pipe_set.refiner is not None
                and pipe_set.refiner.device.type == DEVICE_CUDA
            ):
                model_count += 1
                models_pack.sd_pipe_sets[model_name].refiner = move_pipe_to_device(
                    pipe=pipe_set.refiner,
                    model_name=f"{model_name} refiner",
                    device=DEVICE_CPU,
                )

    # If model isn't Kandinsky, move Kandinsky 2.2 to CPU if needed
    if (
        main_model_name != KANDINSKY_2_2_MODEL_NAME
        and KANDINSKY_2_2_KEEP_IN_CPU_WHEN_IDLE
    ):
        if (
            models_pack.kandinsky_2_2.text2img is not None
            and models_pack.kandinsky_2_2.text2img.device.type == DEVICE_CUDA
        ):
            model_count += 1
            models_pack.kandinsky_2_2.text2img = move_pipe_to_device(
                pipe=models_pack.kandinsky_2_2.text2img,
                model_name=f"{KANDINSKY_2_2_MODEL_NAME} text2img",
                device=DEVICE_CPU,
            )
        if (
            models_pack.kandinsky_2_2.inpaint is not None
            and models_pack.kandinsky_2_2.inpaint.device.type == DEVICE_CUDA
        ):
            model_count += 1
            models_pack.kandinsky_2_2.inpaint = move_pipe_to_device(
                pipe=models_pack.kandinsky_2_2.inpaint,
                model_name=f"{KANDINSKY_2_2_MODEL_NAME} inpaint",
                device=DEVICE_CPU,
            )
        if (
            models_pack.kandinsky_2_2.prior is not None
            and models_pack.kandinsky_2_2.prior.device.type == DEVICE_CUDA
        ):
            model_count += 1
            models_pack.kandinsky_2_2.prior = move_pipe_to_device(
                pipe=models_pack.kandinsky_2_2.prior,
                model_name=f"{KANDINSKY_2_2_MODEL_NAME} prior",
                device=DEVICE_CPU,
            )

    # Move other SD pipes to CPU if needed
    pipe_set = models_pack.sd_pipe_sets.get(main_model_name, None)
    if (
        pipe_set is not None
        and sd_spec is not None
        and sd_spec.get("keep_in_cpu_when_idle", False) is True
    ):
        if main_model_pipe == "text2img":
            if (
                pipe_set.img2img is not None
                and pipe_set.img2img.device.type == DEVICE_CUDA
            ):
                model_count += 1
                models_pack.sd_pipe_sets[model_name].img2img = move_pipe_to_device(
                    pipe=models_pack.sd_pipe_sets[model_name].img2img,
                    model_name=f"{model_name} img2img",
                    device=DEVICE_CPU,
                )
            if (
                pipe_set.inpaint is not None
                and pipe_set.inpaint.device.type == DEVICE_CUDA
            ):
                model_count += 1
                models_pack.sd_pipe_sets[model_name].inpaint = move_pipe_to_device(
                    pipe=models_pack.sd_pipe_sets[model_name].inpaint,
                    model_name=f"{model_name} inpaint",
                    device=DEVICE_CPU,
                )
        if main_model_pipe == "img2img":
            if (
                pipe_set.text2img is not None
                and pipe_set.text2img.device.type == DEVICE_CUDA
            ):
                model_count += 1
                models_pack.sd_pipe_sets[model_name].text2img = move_pipe_to_device(
                    pipe=models_pack.sd_pipe_sets[model_name].text2img,
                    model_name=f"{model_name} text2img",
                    device=DEVICE_CPU,
                )
            if (
                pipe_set.inpaint is not None
                and pipe_set.inpaint.device.type == DEVICE_CUDA
            ):
                model_count += 1
                models_pack.sd_pipe_sets[model_name].inpaint = move_pipe_to_device(
                    pipe=models_pack.sd_pipe_sets[model_name].inpaint,
                    model_name=f"{model_name} inpaint",
                    device=DEVICE_CPU,
                )
        if main_model_pipe == "inpaint":
            if (
                pipe_set.text2img is not None
                and pipe_set.text2img.device.type == DEVICE_CUDA
            ):
                model_count += 1
                models_pack.sd_pipe_sets[model_name].text2img = move_pipe_to_device(
                    pipe=models_pack.sd_pipe_sets[model_name].text2img,
                    model_name=f"{model_name} text2img",
                    device=DEVICE_CPU,
                )
            if (
                pipe_set.img2img is not None
                and pipe_set.img2img.device.type == DEVICE_CUDA
            ):
                model_count += 1
                models_pack.sd_pipe_sets[model_name].img2img = move_pipe_to_device(
                    pipe=models_pack.sd_pipe_sets[model_name].img2img,
                    model_name=f"{model_name} img2img",
                    device=DEVICE_CPU,
                )

    e = time.time()
    logging.info(
        f"🎛️ 🟢 Moved {model_count} other models to {DEVICE_CPU} in {e - s:.2f}s for -> {main_model_name}, {main_model_pipe}"
    )
