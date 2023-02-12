from PIL import Image
from models.constants import DEVICE


def get_embeds_of_images(images: list[Image.Image], model, processor):
    inputs = processor(images=images, return_tensors="pt")
    inputs = inputs.to(DEVICE)
    image_embeddings = model.get_image_features(**inputs)
    image_embeddings = image_embeddings.cpu().numpy().tolist()
    return image_embeddings


def get_embeds_of_texts(texts: str, model, tokenizer):
    inputs = tokenizer(texts, padding=True, return_tensors="pt")
    inputs = inputs.to(DEVICE)
    text_embeddings = model.get_text_features(**inputs)
    text_embeddings = text_embeddings.cpu().numpy().tolist()
    return text_embeddings
