import torch


def identifiability(output_std):
    """
    Compute identifiability 1/std of prediction
    output_std: Predicted parameter marginal standard deviation
    """
    return torch.mean(1 / output_std, axis=0)


def accuracy(output_mean, target):
    """
    mean absolute error
    output_mean: Predicted parameter mean
    target: Ground truth labels.
    """
    acc = torch.mean(abs(output_mean - target), axis=0)
    return acc


def rel_accuracy(output_mean, target):
    """
    mean relative absolute error
    output_mean: Predicted parameter mean
    target: Ground truth labels.
    """
    racc = torch.mean(
        abs(output_mean - target)
        / torch.maximum(abs(output_mean), abs(target)),
        axis=0,
    )
    return racc
