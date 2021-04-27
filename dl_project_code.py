import dl_utils
import numpy as np
#from tensorforce.environments import Environment
#from tensorforce.agents import Agent
#from pyswmm import Simulation, Nodes, Links
import pyswmm.toolkitapi as tkai
#import random
#import pandas as pd
#import matplotlib.pyplot as plt
#%% testing out environment
test_env = dl_utils.swmm_env(model_name = 'theta', threshold = 0.2, scaling = 1)
route_step = test_env.env.sim._model.getSimAnalysisSetting(tkai.SimulationParameters.RouteStep.value)
done = False
while not done:
    done = test_env.step(actions = np.asarray([1, 1]))

df = dl_utils.create_df_of_outputs(test_env, route_step)

fig_name = '1_uncontrolled'
dl_utils.plt_key_states(fig_name, df, test_env)

















