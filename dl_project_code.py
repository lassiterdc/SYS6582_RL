import dl_utils
import numpy as np
#from tensorforce.environments import Environment
#from tensorforce.agents import Agent
#from pyswmm import Simulation, Nodes, Links
import pyswmm.toolkitapi as tkai
#import random
import pandas as pd
#import matplotlib.pyplot as plt
#import tensorflow as tf
#import dl_utils
#import numpy as np
from tensorforce.environments import Environment
#import pyswmm.toolkitapi as tkai
from tensorforce.agents import Agent
import time

# environment hyperparameters
threshold = 1.75
scaling = 4
action_penalty = 0
advance_seconds = 15 * 60 # how often to take action and evaluate controls

# agent hyperparameters
agent = 'tensorforce'
max_ep_tstep = None
update = 64
optimizer = dict(optimizer = "adam", learning_rate = 1e-3)
objective = 'policy_gradient'
reward_estimation = dict(horizon = 5)
summarizer = dict(directory='_summaries', summaries = 'all')
saver = dict(directory = '_model', frequency = 10, unit = "episodes")

# runner parameters
num_episodes = 50

# timing
start_time = time.time()
#%% creating baseline scenario
baseline_model_name = 'theta_undvlpd_train'
baseline_config = 'theta_baseline'

baseline_env = dl_utils.swmm_baseline_env(baseline_model_name, baseline_config)
route_step = baseline_env.env.sim._model.getSimAnalysisSetting(tkai.SimulationParameters.RouteStep.value)

done = False
while not done:
    done = baseline_env.step()

baseline_df = dl_utils.create_df_of_outputs(baseline_env, route_step, set_idx_to_hours = False)
#%% testing out environment
model_name = 'theta_train'
config_name = 'theta'

ucntrld_env = dl_utils.swmm_env(model_name = model_name, config_name = config_name,
                                threshold = threshold, scaling = scaling,
                                action_penalty = action_penalty,
                                baseline_df = baseline_df.copy(),
                                advance_seconds = advance_seconds)

route_step = ucntrld_env.env.sim._model.getSimAnalysisSetting(tkai.SimulationParameters.RouteStep.value)
done = False
while not done:
    done, reward = ucntrld_env.step(actions = [1, 1])

fig_name = '1_uncontrolled_train'

df = dl_utils.create_df_of_outputs(ucntrld_env, route_step)
dl_utils.plt_key_states(fig_name, df, ucntrld_env)

#%% training set up 
model_name = 'theta_train'
config_name = 'theta'

train_env = dl_utils.custom_tensorflow_env(model_name = model_name,
                                           config_name = config_name,
                                           threshold = threshold, 
                                           scaling = scaling,
                                           action_penalty = action_penalty,
                                           baseline_df = baseline_df.copy(),
                                           advance_seconds = advance_seconds)

route_step = train_env.swmm_env.env.sim._model.getSimAnalysisSetting(tkai.SimulationParameters.RouteStep.value)

tr_env = Environment.create(environment = train_env)

t_ag = Agent.create(agent = agent, environment = tr_env, 
                    max_episode_timesteps = max_ep_tstep,
                    update = update, optimizer = optimizer,
                    objective = objective,
                    reward_estimation = reward_estimation)
#                    summarizer = summarizer,
#                    saver = saver)

#%% training
from tensorforce.execution import Runner

runner = Runner(
    agent=t_ag,
    environment=tr_env,
)
runner.run(num_episodes=num_episodes)

t_ag.save(directory='model-numpy', format='numpy', append='episodes')

t_ag.close()
tr_env.close()
runner.close()

df = dl_utils.create_df_of_outputs(train_env.swmm_env, route_step)
# create multi-indexed df
time = df.index
df = df.set_index([('episode', 0), time])
df.index.rename(['episode', 'time'], inplace = True)

# plot all episodes overlapping
idx = pd.IndexSlice
quants = [0.1, .3 , .5, .7, .9, 1]
eps = np.arange(1, num_episodes+1, 1)

q_cnt = -1
for q in quants:
    q_cnt += 1
    ep = np.quantile(eps, q, interpolation = 'nearest')
    df_temp = df.copy().loc[idx[ep, :], :]
    fig_name = '2_training_' + str(ep) + '_episodes'
    dl_utils.plt_key_states(fig_name, df_temp.droplevel('episode'), train_env.swmm_env)

import time
end_time = time.time()
print('Time elapsed: ' + str(round((end_time - start_time)/60/60, 2)) + ' hours')

#%% testing set up 
#model_name = 'theta_test'
#config_name = 'theta'
#baseline_model_name = 'theta_undvlopd_test'
#
#test_env = dl_utils.custom_tensorflow_env(model_name = model_name,
#                                           config_name = config_name,
#                                           threshold = threshold, 
#                                           scaling = scaling)
#
#route_step = test_env.swmm_env.env.sim._model.getSimAnalysisSetting(tkai.SimulationParameters.RouteStep.value)
#
#tst_env = Environment.create(environment = test_env)
#
#ag = Agent.load(directory='model-numpy', format='numpy', environment = tst_env)
#
#
##%% testing
#num_episodes = 1
#from tensorforce.execution import Runner
#
#runner = Runner(
#    agent=ag,
#    environment=tst_env,
#)
#
#runner.run(num_episodes=num_episodes)
#
#ag.save(directory='model-numpy', format='numpy', append='episodes')
#
#ag.close()
#tst_env.close()
#runner.close()
#
#
#df = dl_utils.create_df_of_outputs(test_env.swmm_env, route_step)
#fig_name = '3_testing_' + str(num_episodes) + '_episodes'
#dl_utils.plt_key_states(fig_name, df, test_env.swmm_env)

































