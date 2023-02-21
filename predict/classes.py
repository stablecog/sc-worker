from PIL import Image


class PredictOutput:
    def __init__(
        self,
        pil_image: Image.Image,
        target_extension: str,
        target_quality: int,
        image_embed: list[float],
        prompt_embed: list[float],
    ):
        self.pil_image = pil_image
        self.target_extension = target_extension
        self.target_quality = target_quality
        self.image_embed = image_embed
        self.prompt_embed = prompt_embed


class PredictResult:
    def __init__(
        self,
        outputs: list[PredictOutput],
        nsfw_count: int,
    ):
        self.outputs = outputs
        self.nsfw_count = nsfw_count
