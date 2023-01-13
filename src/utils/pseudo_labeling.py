import torch


def generate_pseudo_labels(model: torch.nn.Module, X: torch.Tensor) -> torch.Tensor:
    # predict
    output, _ = model(X)
    pred_prob = torch.softmax(output, dim=1)

    # generate pseudo-labels
    pseudo_labels = torch.argmax(pred_prob, dim=1)

    return pseudo_labels
