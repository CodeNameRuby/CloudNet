# utils.py
import numpy as np
from PIL import Image
import os

def load_and_resize_image(path, target_size):
    img = Image.open(path).convert('RGB')
    img = img.resize(target_size)
    return np.array(img) / 255.0

def load_and_resize_mask(path, target_size):
    mask = Image.open(path).convert('L')
    mask = mask.resize(target_size)
    return np.expand_dims(np.array(mask) / 255.0, axis=-1)

def save_mask(mask, save_path, threshold=0.5):
    """
    Save predicted mask as a binary image.
    """
    binary_mask = (mask > threshold).astype(np.uint8) * 255
    img = Image.fromarray(binary_mask)
    img.save(save_path)

def overlay_mask_on_image(image, mask, alpha=0.4):
    """
    Overlay mask (binary) on image (RGB), both must be NumPy arrays.
    """
    color_mask = np.zeros_like(image)
    color_mask[:, :, 0] = mask.squeeze() * 255  # Red channel for mask
    overlay = image * (1 - alpha) + color_mask * alpha
    return overlay.astype(np.uint8)
