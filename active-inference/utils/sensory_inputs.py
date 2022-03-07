import torch
from .hyper_parameters import *


def o_at(a_t: float):
    t_a_t = torch.Tensor([a_t]).view(n_pop, d_o, -1)
    return t_a_t


def o_xt(x_t: float):
    t_x_t = torch.Tensor([x_t]).view(n_pop, d_o, -1)
    return torch.normal(t_x_t, 0.01)


def o_ht(distance: float):
    t_distance = torch.Tensor([distance]).view(n_pop, d_o, -1)
    return torch.exp(-torch.square(t_distance)/2.0/0.3/0.3)
