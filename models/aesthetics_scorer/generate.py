import torch
from .model import preprocess
from models.constants import DEVICE


def normalize(value, range_min, range_max):
    # Ensure the range is valid
    if range_min == range_max:
        raise ValueError("Minimum and maximum range values cannot be the same.")
    if range_min > range_max:
        raise ValueError(
            "Minimum range value cannot be greater than the maximum range value."
        )

    # Normalize the value
    normalized_value = (value - range_min) / (range_max - range_min)
    return max(0, min(normalized_value, 1))  # Clamp between 0 and 1


class AestheticScoreResult:
    def __init__(
        self,
        rating_score: float,
        artifact_score: float,
    ):
        self.rating_score = rating_score
        self.artifact_score = artifact_score


def generate_aesthetic_scores(
    img, rating_model, artifacts_model, vision_model, clip_processor
) -> AestheticScoreResult:
    inputs = clip_processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        vision_output = vision_model(**inputs)
    pooled_output = vision_output.pooler_output
    embedding = preprocess(pooled_output)
    with torch.no_grad():
        rating = rating_model(embedding)
        artifact = artifacts_model(embedding)
    return AestheticScoreResult(
        rating_score=normalize(rating.detach().cpu().item(), 0, 10),
        artifact_score=normalize(artifact.detach().cpu().item(), 0, 5),
    )
