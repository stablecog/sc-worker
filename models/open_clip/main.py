from PIL import Image
from models.constants import DEVICE
from .constants import OPEN_CLIP_TOKEN_LENGTH_MAX
from typing import List
import torch
from shared.helpers import time_it, time_code_block


@time_it
def open_clip_get_embeds_of_images(images: List[Image.Image], model, processor):
    with torch.no_grad():
        with time_code_block(prefix=f"Processed {len(images)} image(s)"):
            inputs = processor(images=images, return_tensors="pt")
        inputs = inputs.to(DEVICE)
        with time_code_block(prefix=f"Embedded {len(images)} image(s)"):
            image_embeddings = model.get_image_features(**inputs)
        with time_code_block(prefix=f"Moved {len(images)} embedding(s) to CPU"):
            image_embeddings = image_embeddings.cpu().numpy().tolist()
        return image_embeddings


@time_it
def open_clip_get_embeds_of_texts(texts: str, model, tokenizer):
    with torch.no_grad():
        with time_code_block(prefix=f"Tokenized {len(texts)} text(s)"):
            inputs = tokenizer(
                texts,
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=OPEN_CLIP_TOKEN_LENGTH_MAX,
            )
        inputs = inputs.to(DEVICE)
        with time_code_block(prefix=f"Embedded {len(texts)} text(s)"):
            text_embeddings = model.get_text_features(**inputs)
        with time_code_block(prefix=f"Moved {len(texts)} embeddings(s) to CPU"):
            text_embeddings = text_embeddings.cpu().numpy().tolist()
        return text_embeddings
