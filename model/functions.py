import torch.optim as optim
import torch.nn.functional as F


def sgd_optimizer(model, lr=0.01, l2_factor=0, momentum=0.9):
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=l2_factor)


def calculate_l1_loss(model, loss, lambda_l1=0.001):
    l1_regularization = 0.
    for param in model.parameters():
        l1_regularization += param.abs().sum()
    loss = loss + lambda_l1*l1_regularization
    return loss


def cross_entropy_loss():
    return F.cross_entropy
