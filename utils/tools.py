import torch

def GussianF(x,mu,sigma):
    result = torch.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return result