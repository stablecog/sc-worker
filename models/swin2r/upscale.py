import torch
import numpy as np
import cv2
from PIL import Image
import time

@torch.cuda.amp.autocast()
def upscale(image, upscaler):
  processor = upscaler["processor"]
  pipe = upscaler["pipe"]
  inputs = processor(image, return_tensors="pt").to("cuda")
  with torch.no_grad():
    inf_start_time = time.time()
    outputs = pipe(**inputs)
    inf_end_time = time.time()
    print(
      f"-- Upscale - Inference time: {round((inf_end_time - inf_start_time) * 1000)} ms --"
    )
  output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
  output = np.moveaxis(output, source=0, destination=-1)
  output = (output * 255.0).round().astype(np.uint8)
  pil_image = Image.fromarray(output)
  return pil_image