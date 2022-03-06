import numpy as np
import torch


def GaussianNLL(y, mean, std):

    nll = 0.5 * torch.sum(torch.square(y - mean) / std**2 + 2 * torch.log(std) +
                          torch.log(2 * np.pi), dim=1)
    return nll


def KLGaussianGaussian(mean_1, std_1, mean_2, std_2):

    kl = torch.sum(0.5 * (2 * torch.log(std_2) - 2 * torch.log(std_1) +
                   (std_1**2 + (mean_1 - mean_2)**2) / std_2**2 - 1), dim=1)

    return kl
