from PIL import Image


def get_features_of_images(images: list[Image.Image], model, processor):
    inputs = processor(images=images, return_tensors="pt")
    inputs = inputs.to("cuda")
    image_features = model.get_image_features(**inputs)
    return image_features
