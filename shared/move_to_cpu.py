import logging
import time
from predict.image.setup import ModelsPack
from models.constants import DEVICE_CPU, DEVICE_CUDA
from .helpers import (
    move_pipe_to_device,
)
from models.kandinsky.constants import (
    KANDINSKY_2_2_KEEP_IN_CPU_WHEN_IDLE,
    KANDINSKY_2_2_MODEL_NAME,
)
from .constants import SD_MODELS


def send_other_models_to_cpu(
    main_model_name: str, main_model_pipe: str, models_pack: ModelsPack
):
    s = time.time()
    logging.info(
        f"🖥️ Sending other models to CPU -> {main_model_name}, {main_model_pipe}"
    )

    # Send other Stable Diffusion models to CPU if needed
    for model_name, pipe_set in models_pack.sd_pipes.items():
        if main_model_name != model_name and SD_MODELS[model_name].get(
            "keep_in_cpu_when_idle"
        ):
            if (
                pipe_set.text2img is not None
                and pipe_set.text2img.device.type == DEVICE_CUDA
            ):
                models_pack.sd_pipes[model_name].text2img = move_pipe_to_device(
                    pipe=pipe_set.text2img,
                    model_name=f"{model_name} text2img",
                    device=DEVICE_CPU,
                )
            if (
                pipe_set.img2img is not None
                and pipe_set.img2img.device.type == DEVICE_CUDA
            ):
                models_pack.sd_pipes[model_name].img2img = move_pipe_to_device(
                    pipe=pipe_set.img2img,
                    model_name=f"{model_name} img2img",
                    device=DEVICE_CPU,
                )
            if (
                pipe_set.inpaint is not None
                and pipe_set.inpaint.device.type == DEVICE_CUDA
            ):
                models_pack.sd_pipes[model_name].inpaint = move_pipe_to_device(
                    pipe=pipe_set.inpaint,
                    model_name=f"{model_name} inpaint",
                    device=DEVICE_CPU,
                )
            if (
                pipe_set.refiner is not None
                and pipe_set.refiner.device.type == DEVICE_CUDA
            ):
                models_pack.sd_pipes[model_name].refiner = move_pipe_to_device(
                    pipe=pipe_set.refiner,
                    model_name=f"{model_name} refiner",
                    device=DEVICE_CPU,
                )

    # Model isn't Kandinsky, send Kandinsky 2.2 to CPU if needed
    if (
        main_model_name != KANDINSKY_2_2_MODEL_NAME
        and KANDINSKY_2_2_KEEP_IN_CPU_WHEN_IDLE
    ):
        if (
            models_pack.kandinsky_2_2.text2img is not None
            and models_pack.kandinsky_2_2.text2img.device.type == DEVICE_CUDA
        ):
            models_pack.kandinsky_2_2.text2img = move_pipe_to_device(
                pipe=models_pack.kandinsky_2_2.text2img,
                model_name=f"{KANDINSKY_2_2_MODEL_NAME} text2img",
                device=DEVICE_CPU,
            )
        if (
            models_pack.kandinsky_2_2.inpaint is not None
            and models_pack.kandinsky_2_2.inpaint.device.type == DEVICE_CUDA
        ):
            models_pack.kandinsky_2_2.inpaint = move_pipe_to_device(
                pipe=models_pack.kandinsky_2_2.inpaint,
                model_name=f"{KANDINSKY_2_2_MODEL_NAME} inpaint",
                device=DEVICE_CPU,
            )
        if (
            models_pack.kandinsky_2_2.prior is not None
            and models_pack.kandinsky_2_2.prior.device.type == DEVICE_CUDA
        ):
            models_pack.kandinsky_2_2.prior = move_pipe_to_device(
                pipe=models_pack.kandinsky_2_2.prior,
                model_name=f"{KANDINSKY_2_2_MODEL_NAME} prior",
                device=DEVICE_CPU,
            )

    # Send other Stable Diffusion pipes to CPU if needed
    main_model = models_pack.sd_pipes.get(main_model_name, None)
    if main_model is not None and main_model.get("keep_in_cpu_when_idle"):
        if (
            main_model_name != "text2img"
            and main_model.text2img is not None
            and main_model.text2img.device.type == DEVICE_CUDA
        ):
            models_pack.sd_pipes[main_model_name].text2img = move_pipe_to_device(
                pipe=main_model.text2img,
                model_name=f"{main_model_name} text2img",
                device=DEVICE_CPU,
            )
        if (
            main_model_pipe != "img2img"
            and main_model.img2img is not None
            and main_model.img2img.device.type == DEVICE_CUDA
        ):
            models_pack.sd_pipes[main_model_name].img2img = move_pipe_to_device(
                pipe=main_model.img2img,
                model_name=f"{main_model_name} img2img",
                device=DEVICE_CPU,
            )
        if (
            main_model_pipe != "inpaint"
            and main_model.inpaint is not None
            and main_model.inpaint.device.type == DEVICE_CUDA
        ):
            models_pack.sd_pipes[main_model_name].inpaint = move_pipe_to_device(
                pipe=main_model.inpaint,
                model_name=f"{main_model_name} inpaint",
                device=DEVICE_CPU,
            )

    # Send other Kandinsky 2.2 pipes to CPU if needed
    if (
        main_model_name == KANDINSKY_2_2_MODEL_NAME
        and KANDINSKY_2_2_KEEP_IN_CPU_WHEN_IDLE
    ):
        if (
            main_model_pipe != "text2img"
            and models_pack.kandinsky_2_2.text2img is not None
            and models_pack.kandinsky_2_2.text2img.device.type == DEVICE_CUDA
        ):
            models_pack.kandinsky_2_2.text2img = move_pipe_to_device(
                pipe=models_pack.kandinsky_2_2.text2img,
                model_name=f"{KANDINSKY_2_2_MODEL_NAME} text2img",
                device=DEVICE_CPU,
            )

        if (
            main_model_pipe != "inpaint"
            and models_pack.kandinsky_2_2.inpaint is not None
            and models_pack.kandinsky_2_2.inpaint.device.type == DEVICE_CUDA
        ):
            models_pack.kandinsky_2_2.inpaint = move_pipe_to_device(
                pipe=models_pack.kandinsky_2_2.inpaint,
                model_name=f"{KANDINSKY_2_2_MODEL_NAME} inpaint",
                device=DEVICE_CPU,
            )

    e = time.time()
    logging.info(
        f"🖥️🟢 Sent other models to CPU in {e - s:.2f}s -> {main_model_name}, {main_model_pipe}"
    )
