"""This module provides tools for image (np.ndarray) manipulations."""
from __future__ import annotations

import pathlib

import numpy as np
import scipy.ndimage as nd
from activation_tracker.model import ModelWithActivations
from PIL import Image


def convert_to_255scale(image):
    """Convert an image to 255 scale."""
    clipped = np.clip(image, 0., 1.)
    image_255 = 255 * clipped
    return image_255.astype(np.uint8)


def channel_last(image):
    """Channel first to channel last."""
    transposed = np.transpose(image, (1, 2, 0))
    return transposed


def channel_first(image):
    """Channel last to channel first."""
    transposed = np.transpose(image, (2, 0, 1))
    return transposed


def load_image_from(path: pathlib.Path):
    """Load an image as a np.ndarray (3, h, w)"""
    image = np.array(Image.open(path)).astype(np.float32)
    image /= 255.
    return channel_first(image)


def create_random_image(h=500, w=500):
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
