# Neural Style Transfer
A streamlit application for experimenting with [Neural Style Transfer](https://en.wikipedia.org/wiki/Neural_style_transfer).

![](https://github.com/plachert/neural-style-transfer/blob/main/examples/nst_shot_demo.gif)

## Table of Contents
* [Description](#description)
* [Getting Started](#getting-started)
* [Registering new models](#registering-new-models)

##  Description
Neural style transfer is a cool fun project that doesn't require much resources. There are already many repositories and tutorials on NST. One of the greatest materials I found on NST is this great series by [Aleksa Gordić](https://www.youtube.com/watch?v=S78LQebx6jo&list=PLBoQnSflObcmbfshq9oNs41vODgXG-608).

The aim of this repository is not to explain NST, but to provide a convinient tool to experiment with the algorithm. It's a simple streamlit application where you can play with the parameters of the algorithm and see how they affect the generated image.


## Getting Started
1. `git clone https://github.com/plachert/neural-style-transfer.git`
2. `pip install -r requirements.txt` (run it in your virtual env in project dir)
3. `streamlit run streamlit_app.py`

You should see the following screen:
![](https://github.com/plachert/neural-style-transfer/blob/main/examples/show_start_app.png)

First you need to pick a model that you want to use in the NST algorithm. Currently only one model is available, but you can easily add a new model (see [Registering new models](#registering-new-models))

The parameters are divided into three categories:
* Content params - here you can upload a content image and define the layer that will be used when calculating content loss.
* Style params - pick a style image (it will be resized to the size of the content image) and select layers that will be used when calculating style loss. Here you can choose more than one layer.
* Optimization params:
    * Initialize with - you can initialize the input image with random numbers or with the values of either the content or style image depending on the effect you want to achieve.
    * Content/Style weight - contribution of each loss to the total loss
    * Iterations - number of optimization steps
    * Regularization coeff - weight of the total variance
    * Learning rate - learning rate of Adam optimizer

When you selected all the required parameters just hit `Run NST` and wait for the result.
![](https://github.com/plachert/neural-style-transfer/blob/main/examples/show_nst.gif)

## Registering new models
If you want to use some other model you should add it to the `nst/config.py` following the `Config` template.
```python
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

SUPPORTED_CONFIGS = {}


def register_config(cls):
    instance = cls()
    SUPPORTED_CONFIGS[cls.__name__] = instance
    return cls


class Config:
    @property
    def classifier(self) -> nn.Module:
        raise NotImplementedError

    @property
    def processor(self) -> Callable:
        raise NotImplementedError

    @property
    def deprocessor(self) -> Callable:
        raise NotImplementedError

    @property
    def example_input(self) -> torch.Tensor:
        raise NotImplementedError


@register_config
class VGG16ImageNet(Config):
    ...


@register_config
class YourModel(Config):

    @property
    def classifier(self):
        """
        Return torch.nn.Module. You can use torchvision or your own models.
        """

    @property
    def processor(self):
        """
        Return a function that processes original image
        """

    @property
    def deprocessor(self):
        """
        Return a function that inverts the processing.
        """

    @property
    def example_input(self):
        """
        Return an example input for the classifier.
        It is used by Activation Tracker to inspect the layers of the model.
        """
```

## Licence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/plachert/deep-dream-visualiser/blob/main/LICENSE)
