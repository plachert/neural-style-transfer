"""This module provides activation filters for NST."""
from __future__ import annotations

import torch
from activation_tracker.activation import Activation
from activation_tracker.activation import ActivationFilter


class ConvLikeActivationFilter(ActivationFilter):
    """Preserve activations with shapes (n, c, h, w)."""

    def filter_activations(self, activations: list[Activation]) -> list[Activation]:
        return [activation for activation in activations if len(activation.output_shape) == 4]


class VerboseActivationFilter(ActivationFilter):
    """This one is needed for streamlit app to filter activations by selection."""

    def __init__(self, layers: list[tuple([int, str, torch.Size])]):
        self.layers = layers

    def filter_activations(self, activations: list[Activation]) -> list[Activation]:
        activations_result = []
        for idx, activation in enumerate(activations):
            layer_type = activation.layer_type
            if (idx, layer_type) in self. layers:
                activations_result.append(activation)
        return activations_result

    @staticmethod
    def list_all_available_parameters(activations):
        return [(idx, activation.layer_type) for idx, activation in enumerate(activations)]
