"""This module provides tools for NST optimization."""
from __future__ import annotations

import numpy as np
import torch
from activation_tracker.model import ModelWithActivations
from torchmetrics.functional import total_variation
from tqdm import tqdm

if torch.cuda.is_available():
    dev = 'cuda:0'
else:
    dev = 'cpu'
device = torch.device(dev)


def gram_matrix(feature_maps):
    """Compute Gram Matrix for given feature maps."""
    a, b, c, d = feature_maps.size()
    features = feature_maps.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)


def maps_to_gram(list_of_feats):
    """Get Gram Matrices for a list of feature maps."""
    grams = [gram_matrix(feature_maps) for feature_maps in list_of_feats]
    return grams


def get_nst_loss(
    ref_content,
    ref_style,
    image_content,
    image_style,
    content_weight=1,
    style_weight=1,
):
    """Compute NST loss without regularization."""
    style_weight = style_weight * 100000
    l_content = torch.nn.functional.mse_loss(ref_content, image_content)
    l_style = [
        torch.nn.functional.mse_loss(
            ref_style[i], image_style[i],
        ) for i in range(len(ref_style))
    ]
    l_style = torch.mean(torch.stack(l_style))
    loss = content_weight * l_content + style_weight * l_style
    return loss


def prepare_input_image(input_image: np.ndarray, requires_grad=False):
    """Prepare the image to be used in optimization."""
    input_image = input_image.astype(dtype=np.float32)
    input_image = torch.from_numpy(input_image)
    input_image = input_image.to(device)
    input_image = torch.unsqueeze(input_image, 0)  # minibatch
    input_image.requires_grad = requires_grad
    return input_image


def normalized_tv(image):
    """Compute total variation normalized by number of pixels."""
    size = image.shape[-2] * image.shape[-1]
    return total_variation(image) / size


def prepare_for_optimization(
    model,
    content_image,
    style_image,
    input_image,
):
    """Prepare model and inputs for NST optimization."""
    model.to(device)
    content_image = prepare_input_image(content_image, requires_grad=False)
    style_image = prepare_input_image(style_image, requires_grad=False)
    input_image = prepare_input_image(input_image, requires_grad=True)
    return model, content_image, style_image, input_image


def get_reference(
    model,
    content_image_torch,
    style_image_torch,
):
    """Calculate content and style from original images."""
    model(content_image_torch)
    ref_content = model.activations_values['content'][0]
    model(style_image_torch)
    ref_style = maps_to_gram(model.activations_values['style'])
    return ref_content, ref_style


def optimize_image(
    content_image: np.ndarray,
    content_weight: float,
    style_image: np.ndarray,
    style_weight: float,
    input_image: np.ndarray,
    model: ModelWithActivations,
    n_iterations: int = 60,
    regularization_coeff: float = 1.,
    lr: float = 0.3,
):
    """Run NST optimization."""
    model, content_image_torch, style_image_torch, input_image = prepare_for_optimization(
        model,
        content_image,
        style_image,
        input_image,
    )
    ref_content, ref_style = get_reference(
        model, content_image_torch, style_image_torch,
    )
    optimizer = torch.optim.Adam([input_image], lr=lr)
    processed_images = []

    for _ in tqdm(range(n_iterations)):
        optimizer.zero_grad()
        model(input_image)
        image_content = model.activations_values['content'][0]
        image_style = maps_to_gram(model.activations_values['style'])
        tv = 10 * regularization_coeff * normalized_tv(input_image)
        loss = get_nst_loss(
            ref_content,
            ref_style,
            image_content,
            image_style,
            content_weight,
            style_weight,
        )
        loss += tv
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            input_image.clamp_(0, 1)
        processed_images.append(
            np.copy(input_image.cpu().detach().numpy().squeeze()),
        )
    return processed_images
