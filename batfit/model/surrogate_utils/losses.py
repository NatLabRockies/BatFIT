import torch

def mse_loss(output, target):
    """
    Custom mean squared error loss function.
    output: Predicted values.
    target: Ground truth labels.
    """
    loss = torch.mean((output - target) ** 2)
    return loss


def mae_loss(output, target):
    """
    Custom mean squared error loss function.
    output: Predicted values.
    target: Ground truth labels.
    """
    loss = torch.mean(torch.abs(output - target))
    return loss


