import torch
from torch import nn
import numpy as np

from unity import EngineType, Environment
from utils import sensory_inputs as si, generative_model as gm
from utils.hyper_parameters import *


def run_model():
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

    s_t.copy_(hs_t)


if __name__ == "__main__":
    env = Environment()
    env.init_env()

    env.set_engine_type(EngineType.Both)
    env.set_n_pop(n_pop)
    env.reset()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

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

        distance = env.get_distance()
        with torch.no_grad():
            while not np.less_equal(np.abs(distance), 0.05).any():
                run_model()
    except Exception as e:
        print(e)

    env.close()
