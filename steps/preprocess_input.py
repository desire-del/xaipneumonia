from PIL import Image
import numpy as np


def preprocess_input(image_path):

    img = Image.open(image_path)
    img = img.resize((256, 256))
    img = img.convert("RGB")
    img_array = np.array(img) / 255.0 
    img_array = np.expand_dims(img_array, axis=0)

    return img_array.astype(np.float32)

def preprocess_batch_input(image_paths):
    images = []
    for image_path in image_paths:
        img = Image.open(image_path)
        img = img.resize((240, 240))
        img = img.convert("RGB")
        img_array = np.array(img) / 255.0 
        images.append(np.expand_dims(img_array, axis=0))

    return np.concatenate(images, axis=0).astype(np.float32)