import torch
from torch import optim

from unity import EngineType, Environment
from utils import si, gm, fn
from utils.hyper_parameters import *

env = Environment()
env.init_env()

env.set_engine_type(EngineType.Both)
env.reset()


F = 0

s_t = torch.zeros((1, d_s))
x_t = env.get_position()[0]
a_t = env.get_action()

a_model = gm.ANetModel(d_s, d_a)
s_model = gm.SNetModel(d_s)
q_model = gm.QNetModel(d_s, d_o)
o_model = gm.ONetModel(d_s, d_o)

a_model.init_model()
s_model.init_model()
q_model.init_model()
o_model.init_model()


def sample_based_approximation_of_F():
    at_mean, at_std = a_model(s_t)

    at = torch.normal(at_mean, at_std)

    env.set_action(at)

    x_t = env.get_position()[0]
    a_t = env.get_action()
    distance = env.get_distance()[0]

    o_xt = si.o_xt(x_t)
    o_ht = si.o_ht(distance)
    o_at = si.o_at(a_t)

    hst_mean, hst_std = q_model(s_t, o_xt, o_ht, o_at)
    hs_t = torch.normal(hst_mean, hst_std)

    (o_xt_mean, o_xt_std),\
        (o_ht_mean, o_ht_std),\
        (o_at_mean, o_at_std) = o_model(hs_t)

    l_o_xt = fn.GaussianNLL(o_xt, o_xt_mean, o_xt_std)
    l_o_ht = fn.GaussianNLL(o_xt, o_ht_mean, o_ht_std)
    l_o_at = fn.GaussianNLL(o_xt, o_at_mean, o_at_std)

    st_mean, st_std = s_model(s_t)

    KL_st = fn.KLGaussianGaussian(hst_mean, hst_std, st_mean, st_std)

    FEt = KL_st + l_o_xt + l_o_ht + l_o_at

    s_t.copy_(hs_t)

    return FEt


def optimisation_of_F_bound(s_t):
    params = (a_model.parameters(), s_model.parameters(),
              q_model.parameters(), o_model.parameters())

    sigmas = (a_model.sigmas, s_model.sigmas,
              q_model.sigmas, o_model.sigmas)

    optim_params = optim.Adam(params,
                              lr=adam_alpha,
                              betas=(adam_beta_1, adam_beta_2),
                              eps=adam_epsilon)

    optim_sigmas = optim.Adam(sigmas,
                              lr=adam_alpha,
                              betas=(adam_beta_1, adam_beta_2),
                              eps=adam_epsilon)

    for i in range(steps):
        FEt = sample_based_approximation_of_F(s_t)

        optim_params.zero_grad()
        optim_sigmas.zero_grad()

        a_model.calc_grad(FEt)
        s_model.calc_grad(FEt)
        q_model.calc_grad(FEt)
        o_model.calc_grad(FEt)

        optim_params.step()
        optim_sigmas.step()


optimisation_of_F_bound(s_t)
