import torch


def o_at(a_t):
    return a_t


def o_xt(x_t):
    return torch.normal(x_t, 0.01)


def o_ht(distance):
    return torch.exp(-torch.square(distance)/2.0/0.3/0.3)
