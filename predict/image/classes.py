from PIL import Image


class PredictOutput:
    def __init__(
        self,
        pil_image: Image.Image,
        target_extension: str,
        target_quality: int,
    ):
        self.pil_image = pil_image
        self.target_extension = target_extension
        self.target_quality = target_quality


class PredictResult:
    def __init__(
        self,
        outputs: list[PredictOutput],
        signed_urls: list[str],
        nsfw_count: int,
    ):
        self.outputs = outputs
        self.signed_urls = signed_urls
        self.nsfw_count = nsfw_count
