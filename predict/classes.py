from PIL import Image


class PredictOutput:
    def __init__(
        self,
        pil_image: Image.Image,
        target_extension: str,
        target_quality: int,
        clip_image_embedding: list[float],
        clip_prompt_embedding: list[float],
    ):
        self.pil_image = pil_image
        self.target_extension = target_extension
        self.target_quality = target_quality
        self.image_embedding = clip_image_embedding
        self.prompt_embedding = clip_prompt_embedding


class PredictResult:
    def __init__(
        self,
        outputs: list[PredictOutput],
        nsfw_count: int,
    ):
        self.outputs = outputs
        self.nsfw_count = nsfw_count
