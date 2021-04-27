import dl_utils
import numpy as np
#from tensorforce.environments import Environment
#from tensorforce.agents import Agent
#from pyswmm import Simulation, Nodes, Links
import pyswmm.toolkitapi as tkai
#import random
#import pandas as pd
#import matplotlib.pyplot as plt
#import tensorflow as tf
#import dl_utils
#import numpy as np
from tensorforce.environments import Environment
#import pyswmm.toolkitapi as tkai
from tensorforce.agents import Agent

threshold = 0.2
scaling = 3
agent = 'tensorforce'
max_ep_tstep = 1e6
update = 64
optimizer = dict(optimizer = "adam", learning_rate = 1e-3)
objective = 'policy_gradient'
reward_estimation = dict(horizon = 5)
summarizer = dict(directory='summaries', summaries = 'all')
saver = dict(directory = 'model', frequency = 10, unit = "episodes")



#%% creating custom tensorflow environment and agent
train_env = dl_utils.custom_tensorflow_env(model_name = 'theta', threshold = threshold, scaling = scaling)
route_step = train_env.swmm_env.env.sim._model.getSimAnalysisSetting(tkai.SimulationParameters.RouteStep.value)

tr_env = Environment.create(environment = train_env)

t_ag = Agent.create(agent = agent, environment = tr_env, max_episode_timesteps = max_ep_tstep,
                    update = update, optimizer = optimizer,
                    objective = objective,
                    reward_estimation = reward_estimation,
                    summarizer = summarizer,
                    saver = saver)


#%% training
num_episodes = 3
from tensorforce.execution import Runner
runner = Runner(
    agent=t_ag,
    environment=tr_env,
)
runner.run(num_episodes=num_episodes)

t_ag.save(directory='model-numpy', format='numpy', append='episodes')

# Close agent separately, since created separately

#runner.run(num_episodes=100, evaluation=True)
t_ag.close()
tr_env.close()
runner.close()

#%% plot results
df = dl_utils.create_df_of_outputs(tr_env.swmm_env, route_step)
fig_name = '3_using_runner_' + str(num_episodes) + '_episodes'
dl_utils.plt_key_states(fig_name, df, tr_env.swmm_env)

#%% load trained model
t_ag = Agent.load(directory='model-numpy', format='numpy', environment=tr_env)





