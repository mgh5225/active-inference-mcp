import torch
import numpy as np
from torch import optim, nn
from torch.nn import functional as F

from unity import EngineType, Environment
from utils import sensory_inputs as si, generative_model as gm, functions as fn
from utils.hyper_parameters import *


def run_model():
    at_mean, at_std = a_model(s_t)

    at = F.tanh(torch.normal(at_mean, at_std))

    env_at = at.cpu().numpy().reshape(1)[0]
    env.set_action(env_at)

    x_t = env.get_position()[0, 0]
    a_t = env.get_action()[0]
    distance = env.get_distance()[0, 0]

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


if __name__ == "__main__":
    env = Environment()
    env.init_env()

    env.set_engine_type(EngineType.Both)
    env.reset()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FEs = []

    s_t = torch.zeros((1, d_s))

    a_model = gm.ANetModel(d_s, d_a)
    s_model = gm.SNetModel(d_s)
    q_model = gm.QNetModel(d_s, d_o)
    o_model = gm.ONetModel(d_s, d_o)

    a_model.load_model()
    s_model.load_model()
    q_model.load_model()
    o_model.load_model()

    distance = env.get_distance()[0, 0]
    with torch.no_grad():
        while abs(distance) > 0.05:
            run_model()

    env.close()
