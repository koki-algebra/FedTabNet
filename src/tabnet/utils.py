import copy
from typing import List
import zipfile
import io

import torch
from torch import nn, Tensor


def load_weights_from_pretrained(
    model: nn.Module, pretrained_model: nn.Module, file_path: str = None
):
    update_state_dict = copy.deepcopy(model.state_dict())

    if file_path is not None:
        if file_path.endswith(".zip"):
            try:
                with zipfile.ZipFile(file_path) as z:
                    with z.open("network.pt") as f:
                        try:
                            saved_state_dict = torch.load(f)
                        except io.UnsupportedOperation:
                            saved_state_dict = torch.load(io.BytesIO(f.read()))
            except KeyError:
                raise KeyError("Your zip file is missing at least one component")
        elif file_path.endswith(".pth"):
            saved_state_dict = torch.load(file_path)
        else:
            raise ValueError("File extension must be .zip or .pth")

        # load model parameters
        pretrained_model.load_state_dict(saved_state_dict)

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


def tensor_exists(tensor: Tensor, l: List[Tensor]) -> bool:
    for val in l:
        if tensor.shape == val.shape:
            if tensor.eq(val).any():
                return True

    return False
