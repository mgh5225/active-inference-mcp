import torch
import numpy as np
from torch import optim, nn

from unity import EngineType, Environment
from utils import sensory_inputs as si, generative_model as gm
from utils.hyper_parameters import *


def get_parameters():
    params = nn.ParameterList()
    params.extend(list(a_model.parameters()))
    params.extend(list(s_model.parameters()))
    params.extend(list(q_model.parameters()))
    params.extend(list(o_model.parameters()))

    return params


def get_sigmas():
    sigmas = nn.ParameterList()
    sigmas.extend(a_model.sigmas)
    sigmas.extend(s_model.sigmas)
    sigmas.extend(q_model.sigmas)
    sigmas.extend(o_model.sigmas)

    return sigmas


def sample_based_approximation_of_F():
    F = torch.zeros(n_pop, d_o)
    for i in range(n_run_steps):
        at_mean, at_std = a_model(s_t)

        at = torch.tanh(torch.normal(at_mean, at_std))

        env_at = at.view(n_pop, -1).cpu().numpy()
        env.set_action(env_at)

        x_t = env.get_position()
        a_t = env.get_action()
        distance = env.get_distance()

        o_xt = si.o_xt(x_t)
        o_ht = si.o_ht(distance)
        o_at = si.o_at(a_t)

        hst_mean, hst_std = q_model(s_t, o_xt, o_ht, o_at)
        hs_t = torch.normal(hst_mean, hst_std)

        (o_xt_mean, o_xt_std),\
            (o_ht_mean, o_ht_std),\
            (o_at_mean, o_at_std) = o_model(hs_t)

        loss = nn.GaussianNLLLoss(reduction="none")
        kl = nn.KLDivLoss(reduction="none")

        l_o_xt = loss(o_xt_mean, o_xt, torch.square(o_xt_std)).view(n_pop, d_o)
        l_o_ht = loss(o_ht_mean, o_ht, torch.square(o_ht_std)).view(n_pop, d_o)
        l_o_at = loss(o_at_mean, o_at, torch.square(o_at_std)).view(n_pop, d_o)

        st_mean, st_std = s_model(s_t)
        st = torch.normal(st_mean, st_std)

        KL_st = kl(hs_t, st).mean(dim=-1).view(n_pop, d_o)

        FEt = KL_st + l_o_xt + l_o_ht + l_o_at

        s_t.copy_(hs_t)

        F = F + FEt

    return F


def optimisation_of_F_bound():
    optim_params = optim.Adam(get_parameters(),
                              lr=adam_alpha,
                              betas=(adam_beta_1, adam_beta_2),
                              eps=adam_epsilon)

    optim_sigmas = optim.Adam(get_sigmas(),
                              lr=adam_alpha,
                              betas=(adam_beta_1, adam_beta_2),
                              eps=adam_epsilon)

    for i in range(steps):
        with torch.no_grad():
            FEt = sample_based_approximation_of_F()
            if i % 100 == 0:
                print("[{}]Free Energy: {}".format(i+1, FEt.mean().item()))
            if i % 1000 == 0:
                a_model.save_model()
                s_model.save_model()
                q_model.save_model()
                o_model.save_model()
            FEs.append(FEt)

        optim_params.zero_grad()
        optim_sigmas.zero_grad()

        a_model.calc_grad(FEt)
        s_model.calc_grad(FEt)
        q_model.calc_grad(FEt)
        o_model.calc_grad(FEt)

        optim_params.step()
        optim_sigmas.step()


if __name__ == "__main__":
    env = Environment()
    env.init_env()

    env.set_engine_type(EngineType.Both)
    env.set_max_steps(steps)
    env.set_n_pop(n_pop)
    env.reset()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FEs = []

    s_t = torch.zeros(n_pop, 1, d_s)

    a_model = gm.ANetModel(d_s, d_a)
    s_model = gm.SNetModel(d_s)
    q_model = gm.QNetModel(d_s, d_o)
    o_model = gm.ONetModel(d_s, d_o)

    try:
        a_model.load_model()
        s_model.load_model()
        q_model.load_model()
        o_model.load_model()
    except:
        a_model.init_model()
        s_model.init_model()
        q_model.init_model()
        o_model.init_model()

    try:
        optimisation_of_F_bound()
    except Exception as e:
        print(e)

    env.close()
