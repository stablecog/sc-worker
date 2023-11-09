import torch
from model import preprocess
from models.constants import DEVICE


def generate_aesthetic_scores(
    img, rating_model, artifacts_model, vision_model, clip_processor
) -> tuple(float, float):
    inputs = clip_processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        vision_output = vision_model(**inputs)
    pooled_output = vision_output.pooler_output
    embedding = preprocess(pooled_output)
    with torch.no_grad():
        rating = rating_model(embedding)
        artifact = artifacts_model(embedding)
    return rating.detach().cpu().item(), artifact.detach().cpu().item()
