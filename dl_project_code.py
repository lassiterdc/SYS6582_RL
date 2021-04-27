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
#%% testing out environment
#train_env = dl_utils.swmm_env(model_name = 'theta', threshold = 0.2, scaling = 1)
#route_step = train_env.env.sim._model.getSimAnalysisSetting(tkai.SimulationParameters.RouteStep.value)
#done = False
#while not done:
#    done, reward = train_env.step(actions = np.asarray([1, 1]))
#
#df = dl_utils.create_df_of_outputs(train_env, route_step)
#
#fig_name = '1_uncontrolled'
#dl_utils.plt_key_states(fig_name, df, train_env)


#%% creating custom tensorflow environment and agent
train_env = dl_utils.custom_tensorflow_env(model_name = 'theta_train', threshold = threshold, scaling = scaling)
route_step = train_env.swmm_env.env.sim._model.getSimAnalysisSetting(tkai.SimulationParameters.RouteStep.value)

tr_env = Environment.create(environment = train_env)

t_ag = Agent.create(agent = agent, environment = tr_env, max_episode_timesteps = max_ep_tstep,
                    update = update, optimizer = optimizer,
                    objective = objective,
                    reward_estimation = reward_estimation,
                    summarizer = summarizer,
                    saver = saver)

#%% Train for 100 episodes
#adjustmnt_tstep = 15 # minutes
#
#episodes = 1
#for ep in range(episodes):
#    print(ep)
#    states = t_env.reset()
#    terminal = False
#    reward = 0
#    actions = [.5, .5]
#    loop_counter = -1
#    while not terminal:
#        loop_counter += 1
##        print(loop_counter)
##        print(actions)
#        states, terminal, reward = t_env.execute(actions=actions)
##        print('here')
##        print(terminal)
#        if terminal:
##            print('finished')
#            break
#        if (loop_counter % (adjustmnt_tstep /(route_step / 60))) == 0 and loop_counter > 0:
#            print('triggered')
#            actions = t_ag.act(states=states)
#            
#            t_ag.observe(terminal=terminal, reward=reward)
##            print('---')
##            print(loop_counter)
##            print(states)

#%% plot results
#df = dl_utils.create_df_of_outputs(test_env.swmm_env, route_step)
#fig_name = '2_after_' + str(episodes) + '_episodes'
#dl_utils.plt_key_states(fig_name, df, test_env.swmm_env)

#%% training
num_episodes = 3
from tensorforce.execution import Runner

runner = Runner(
    agent=t_ag,
    environment=t_env,
#    num_parallel = 8,
#    remote = 'multiprocessing'
#    max_episode_timesteps=1e6
)
runner.run(num_episodes=num_episodes)

t_ag.save(directory='model-numpy', format='numpy', append='episodes')

# Close agent separately, since created separately

#runner.run(num_episodes=100, evaluation=True)
t_ag.close()
t_env.close()
runner.close()

#%% plot results
df = dl_utils.create_df_of_outputs(test_env.swmm_env, route_step)
fig_name = '3_using_runner_' + str(num_episodes) + '_episodes'
dl_utils.plt_key_states(fig_name, df, test_env.swmm_env)

#%% testing

#load trained model
train_env = dl_utils.custom_tensorflow_env(model_name = 'theta_test', threshold = threshold, scaling = scaling)
route_step = test_env.swmm_env.env.sim._model.getSimAnalysisSetting(tkai.SimulationParameters.RouteStep.value)
tr_env = Environment.create(environment = train_env)
t_ag = Agent.load(directory='model-numpy', format='numpy', environment=environment)


