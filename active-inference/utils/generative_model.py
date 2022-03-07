import os
import torch
from torch import nn
from torch.nn import functional as F

from .hyper_parameters import *

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/")


class NetModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.sigmas = []
        self.epsilons = []
        self.params_path = os.path.join(
            MODEL_PATH, "{}-params.pt".format(self.__class__.__name__))
        self.sigmas_path = os.path.join(
            MODEL_PATH, "{}-sigmas.pt".format(self.__class__.__name__))
        self.epsilons_path = os.path.join(
            MODEL_PATH, "{}-epsilons.pt".format(self.__class__.__name__))

    def save_model(self):
        torch.save(self.state_dict(), self.params_path)
        torch.save(self.sigmas, self.sigmas_path)
        torch.save(self.epsilons, self.epsilons_path)

    def load_model(self):
        self.load_state_dict(torch.load(self.params_path))
        self.sigmas = torch.load(self.sigmas_path)
        self.epsilons = torch.load(self.epsilons_path)

    def grad_mu(self, FEt, epsilon, sigma):
        # FEt_mean = FEt.mean(dim=0).mean(dim=1)

        N = F.softplus(sigma) + sig_min

        # return torch.dot(FEt_mean, epsilon) / N / n_pop
        return FEt * epsilon / N

    def grad_sigma(self, FEt, epsilon, sigma):
        # FEt_mean = FEt.mean(dim=0).mean(dim=1)

        N = F.softplus(sigma) + sig_min

        eq_1 = (torch.square(epsilon) - 1) / N
        eq_2 = torch.exp(sigma) / (1 + torch.exp(sigma))

        # return torch.dot(FEt_mean, eq_1 * eq_2) / n_pop
        return FEt * eq_1 * eq_2

    def init_sigmas(self):
        for param in self.parameters():
            # sigma = torch.ones_like(param, requires_grad=True) * sig_init
            sigma = nn.parameter.Parameter(torch.ones_like(
                param) * sig_init, requires_grad=True)
            self.sigmas.append(sigma)

    def init_params(self):
        for idx, param in enumerate(self.parameters()):
            # epsilon_half = torch.randn(n_pop/2, param.shape[1], param.shape[2])
            # epsilon = torch.cat((epsilon_half, -1 * epsilon_half), 0)
            epsilon = torch.randn(param.shape)
            with torch.no_grad():
                param.add_(epsilon * F.softplus(self.sigmas[idx]) + sig_min)
            self.epsilons.append(epsilon)

    def init_model(self):
        self.init_sigmas()
        self.init_params()

    def calc_grad(self, FEt):
        for idx, param in enumerate(self.parameters()):
            sigma = self.sigmas[idx]
            epsilon = self.epsilons[idx]

            param.grad = self.grad_mu(FEt, epsilon, sigma)
            sigma.grad = self.grad_sigma(FEt, epsilon, sigma)


class SNetModel(NetModel):
    def __init__(self, d_s):
        super(SNetModel, self).__init__()
        self.st_mean = nn.Linear(d_s, d_s)
        self.st_std = nn.Linear(d_s, d_s)

        self.mean = nn.Tanh()
        self.std = nn.Softplus()

    def forward(self, s_t):
        mean = self.mean(self.st_mean(s_t))
        std = self.std(self.st_std(s_t))

        return mean, std


class QNetModel(NetModel):
    def __init__(self, d_s, d_o):
        super(QNetModel, self).__init__()

        self.hst_st = nn.Linear(d_s, d_s)
        self.hst_oxt = nn.Linear(d_o, d_s)
        self.hst_oht = nn.Linear(d_o, d_s)
        self.hst_oat = nn.Linear(d_o, d_s)

        self.hst_2 = nn.Linear(d_s, d_s)

        self.std_l = nn.Linear(d_s, d_s)

        self.layer_1 = nn.Tanh()
        self.layer_2 = nn.Tanh()

        self.mean = nn.Linear(d_s, d_s)
        self.std = nn.Softplus()

    def forward(self, s_t, o_xt, o_ht, o_at):
        output = self.layer_1(
            self.hst_st(s_t)
            + self.hst_oxt(o_xt)
            + self.hst_oht(o_ht)
            + self.hst_oat(o_at)
        )

        output = self.layer_2(self.hst_2(output))

        mean = self.mean(output)
        std = self.std(self.std_l(output))

        return mean, std


class ANetModel(NetModel):
    def __init__(self, d_s, d_a):
        super(ANetModel, self).__init__()

        self.aht = nn.Linear(d_s, d_s)

        self.std_l = nn.Linear(d_s, d_a)

        self.mean = nn.Linear(d_s, d_a)
        self.std = nn.Softplus()

    def forward(self, s_t):
        output = self.aht(s_t)

        mean = self.mean(output)
        std = self.std(self.std_l(output))

        return mean, std


class ONetModel(NetModel):
    def __init__(self, d_s, d_o):
        super(ONetModel, self).__init__()
        self.o_st_1 = nn.Linear(d_s, d_s)
        self.o_st_2 = nn.Linear(d_s, d_s)
        self.o_st_3 = nn.Linear(d_s, d_s)

        self.layer_1 = nn.Tanh()
        self.layer_2 = nn.Tanh()
        self.layer_3 = nn.Tanh()

        self.o_xt_mean = nn.Linear(d_s, d_o)
        self.o_ht_mean = nn.Linear(d_s, d_o)
        self.o_at_mean = nn.Linear(d_s, d_o)

        self.o_xt_std_l = nn.Linear(d_s, d_o)
        self.o_ht_std_l = nn.Linear(d_s, d_o)
        self.o_at_std_l = nn.Linear(d_s, d_o)

        self.o_xt_std = nn.Softplus()
        self.o_ht_std = nn.Softplus()
        self.o_at_std = nn.Softplus()

    def forward(self, s_t):
        output = self.layer_1(self.o_st_1(s_t))
        output = self.layer_2(self.o_st_2(output))
        output = self.layer_3(self.o_st_3(output))

        o_xt_mean = self.o_xt_mean(output)
        o_ht_mean = self.o_ht_mean(output)
        o_at_mean = self.o_at_mean(output)

        o_xt_std = self.o_xt_std(self.o_xt_std_l(output))
        o_ht_std = self.o_ht_std(self.o_ht_std_l(output))
        o_at_std = self.o_at_std(self.o_at_std_l(output))

        return (o_xt_mean, o_xt_std), (o_ht_mean, o_ht_std), (o_at_mean, o_at_std)
