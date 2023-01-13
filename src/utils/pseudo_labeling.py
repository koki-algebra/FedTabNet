import torch


def generate_pseudo_labels(model: torch.nn.Module, X: torch.Tensor) -> torch.Tensor:
    # predict
    output, _ = model(X)
    pred_prob = torch.softmax(output, dim=1)

    # generate pseudo-labels
    pseudo_labels = torch.argmax(pred_prob, dim=1)

    return pseudo_labels


def pseudo_labeling_schedular(t: int, T_1: int, T_2: int, alpha: float) -> float:
    if T_1 > T_2:
        raise ValueError("T_2 must be smaller than T_1")

    if t < T_1:
        return 0
    elif T_1 <= t and t <= T_2:
        return ((t - T_1) / (T_2 - T_1)) * alpha
    else:
        return alpha
