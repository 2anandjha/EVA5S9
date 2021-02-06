import torch


def cuda_is_available():
    return torch.cuda.is_available()


def enable_cuda():
    use_cuda = cuda_is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device
