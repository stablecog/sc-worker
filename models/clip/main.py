from PIL import Image
from models.constants import DEVICE
from .constants import CLIP_TOKEN_LENGTH_MAX
from typing import List
import torch

def clip_get_embeds_of_images(images: List[Image.Image], model, processor):
    with torch.no_grad():
        inputs = processor(images=images, return_tensors="pt")
        inputs = inputs.to(DEVICE)
        image_embeddings = model.get_image_features(**inputs)
        image_embeddings = image_embeddings.cpu().numpy().tolist()
        return image_embeddings


def clip_get_embeds_of_texts(texts: str, model, tokenizer):
    with torch.no_grad():
        inputs = tokenizer(texts, padding=True, return_tensors="pt", truncation=True, max_length=CLIP_TOKEN_LENGTH_MAX)
        inputs = inputs.to(DEVICE)
        text_embeddings = model.get_text_features(**inputs)
        text_embeddings = text_embeddings.cpu().numpy().tolist()
        return text_embeddings
