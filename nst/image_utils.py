import torch
import pathlib
from PIL import Image
import numpy as np
import scipy.ndimage as nd


def create_random_image(h=224, w=224):
    """Create a random image (channel-first)."""
    shape = (3, h, w)
    image = np.random.uniform(low=0.0, high=1, size=shape).astype(np.float32)
    return image

def resize_to_image(reference_image, image):
    """Resize image to the same shape as the reference image."""
    h_ref, w_ref = reference_image.shape[-2:]
    h_image, w_image = image.shape[-2:]
    h_ratio = h_ref / h_image
    w_ratio = w_ref / w_image
    image_resized = nd.zoom(image, (1, h_ratio, w_ratio), order=1)
    return image_resized


def channel_first(image):
    transposed = np.transpose(image, (2, 0, 1))
    return transposed

def channel_last(image):
    transposed = np.transpose(image, (1, 2, 0))
    return transposed


def load_image_from(path: pathlib.Path):
    """Load an image as a np.ndarray (3, h, w)"""
    image = np.array(Image.open(path)).astype(np.float32)
    image /= 255.
    return channel_first(image)[:3, ...]

def prepare_input_image(input_image: np.ndarray, requires_grad=False):
    input_image = input_image.astype(dtype=np.float32)
    input_image = torch.from_numpy(input_image)
    input_image = torch.unsqueeze(input_image, 0)  # minibatch
    input_image.requires_grad = requires_grad
    return input_image