from PIL import Image
from models.constants import DEVICE
from .constants import OPEN_CLIP_TOKEN_LENGTH_MAX
from typing import List
import torch
from shared.helpers import time_it, time_code_block
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
)
from concurrent.futures import ThreadPoolExecutor, as_completed

CLIP_IMAGE_SIZE = 224


def create_clip_transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


clip_transform = create_clip_transform(CLIP_IMAGE_SIZE)


def clip_preprocessor(images: List[Image.Image], return_tensors="pt"):
    def process_image(img):
        return clip_transform(img)

    with ThreadPoolExecutor() as executor:
        # Submit all images for processing
        futures = [executor.submit(process_image, img) for img in images]

        # Wait for all futures to complete and collect results
        results = [future.result() for future in as_completed(futures)]

    return torch.stack(results)


@time_it
def open_clip_get_embeds_of_images(images: List[Image.Image], model, processor):
    with torch.no_grad():
        with time_code_block(prefix=f"Preprocessed {len(images)} image(s)"):
            inputs = clip_preprocessor(images=images, return_tensors="pt")
        inputs = inputs.to(DEVICE)
        with time_code_block(prefix=f"Embedded {len(images)} image(s)"):
            image_embeddings = model.get_image_features(pixel_values=inputs)
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
