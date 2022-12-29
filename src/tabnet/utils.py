import copy

import torch
from torch import nn


def load_weights_from_pretrained(
    model: nn.Module, pretrained_model: nn.Module, file_path: str = None
):
    update_state_dict = copy.deepcopy(model.state_dict())

    if file_path is not None:
        pretrained_model.load_state_dict(torch.load(file_path))

    for param, weights in pretrained_model.state_dict().items():
        if param.startswith("encoder"):
            # Convert encoder's layers name to match
            new_param = "tabnet." + param
        else:
            new_param = param
        if model.state_dict().get(new_param) is not None:
            # update only common layers
            update_state_dict[new_param] = weights

    model.load_state_dict(update_state_dict)
