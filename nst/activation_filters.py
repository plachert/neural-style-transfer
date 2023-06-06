from activation_tracker.activation import ActivationFilter, Activation
import torch


class ConvLikeActivationFilter(ActivationFilter):
    def filter_activations(self, activations: list[Activation]) -> list[Activation]:
        return [activation for activation in activations if len(activation.output_shape) == 4]
    

class VerboseActivationFilter(ActivationFilter):
    def __init__(self, layers: list[tuple([int, str, torch.Size])]):
        self.layers = layers
        
    def filter_activations(self, activations: list[Activation]) -> list[Activation]:
        activations_result = []
        for idx, activation in enumerate(activations):
            layer_type = activation.layer_type
            output_shape = activation.output_shape
            if (idx, layer_type, output_shape) in self. layers:
                activations_result.append(activation)
        return activations_result
    
    @staticmethod
    def list_all_available_parameters(activations):
        return [(idx, activation.layer_type, activation.output_shape) for idx, activation in enumerate(activations)]
