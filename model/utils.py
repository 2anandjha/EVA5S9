import torch
import matplotlib.pyplot as plt
from cuda import enable_cuda


def set_seed(value=123):
    torch.manual_seed(value)
