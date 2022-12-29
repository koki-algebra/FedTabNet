import copy
from typing import Tuple, List, Dict

import torch
from torch import nn, Tensor

from .utils import tensor_exists


def federated_averaging(models: List[nn.Module], weights: List[int]) -> nn.Module:
    if models == [] or weights == []:
        return None

    model, total_weights = weighted_sum(models, weights)
    model_params = model.state_dict()
    with torch.no_grad():
        for name, params in model_params.items():
            model_params[name] = torch.div(params, total_weights)
    model.load_state_dict(model_params)
    return model


def weighted_sum(models, weights) -> Tuple[nn.Module, int]:
    if models == [] or weights == []:
        return None
    model = copy.deepcopy(models[0])
    model_sum_params: Dict[str, Tensor] = copy.deepcopy(models[0].state_dict())

    # remove shared params
    seen: Dict[str, List[Tensor]] = {}
    for key, value in model_sum_params.items():
        if not tensor_exists(value, list(seen.values())):
            seen[key] = value

    with torch.no_grad():
        for name, params in seen.items():
            params *= weights[0]
            for i in range(1, len(models)):
                model_params = dict(models[i].state_dict())
                params += model_params[name] * weights[i]
            model_sum_params[name] = params
    model.load_state_dict(model_sum_params)
    return model, sum(weights)
