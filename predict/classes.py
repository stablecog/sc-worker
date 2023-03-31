from PIL import Image


class PredictOutput:
    def __init__(
        self,
        pil_image: Image.Image,
        target_extension: str,
        target_quality: int,
        open_clip_image_embed: list[float],
        open_clip_prompt_embed: list[float],
    ):
        self.pil_image = pil_image
        self.target_extension = target_extension
        self.target_quality = target_quality
        self.open_clip_image_embed = open_clip_image_embed
        self.open_clip_prompt_embed = open_clip_prompt_embed


class PredictResult:
    def __init__(
        self,
        outputs: list[PredictOutput],
        nsfw_count: int,
    ):
        self.outputs = outputs
        self.nsfw_count = nsfw_count
