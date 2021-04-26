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
    done, reward = test_env.step(actions = np.asarray([1, 1]))

df = dl_utils.create_df_of_outputs(test_env, route_step)

fig_name = '1_uncontrolled'
dl_utils.plt_key_states(fig_name, df, test_env)



#%% creating custom tensorflow environment and agent
import dl_utils
import numpy as np
from tensorforce.environments import Environment
import pyswmm.toolkitapi as tkai
from tensorforce.agents import Agent

test_env = dl_utils.custom_tensorflow_env(model_name = 'theta', threshold = 0.2, scaling = 100)
route_step = test_env.swmm_env.env.sim._model.getSimAnalysisSetting(tkai.SimulationParameters.RouteStep.value)

t_env = Environment.create(environment = test_env)

t_ag = Agent.create(agent = 'vpg', environment = t_env, max_episode_timesteps = 1e6, 
                    batch_size = 5)

#%% Train for 100 episodes
adjustmnt_tstep = 15 # minutes

episodes = 1
for ep in range(episodes):
    print(ep)
    states = t_env.reset()
    terminal = False
    reward = 0
    actions = [.5, .5]
    loop_counter = -1
    while not terminal:
        loop_counter += 1
#        print(loop_counter)
#        print(actions)
        states, terminal, reward = t_env.execute(actions=actions)
#        print('here')
#        print(terminal)
        if terminal:
#            print('finished')
            break
        if (loop_counter % (adjustmnt_tstep /(route_step / 60))) == 0 and loop_counter > 0:
            print('triggered')
            actions = t_ag.act(states=states)
            
            t_ag.observe(terminal=terminal, reward=reward)
#            print('---')
#            print(loop_counter)
#            print(states)

#%% plot results
df = dl_utils.create_df_of_outputs(test_env.swmm_env, route_step)
fig_name = '2_after_' + str(episodes) + '_episodes'
dl_utils.plt_key_states(fig_name, df, test_env.swmm_env)

#%% training
from tensorforce.execution import Runner

runner = Runner(
    agent=t_ag,
    environment=t_env,
#    max_episode_timesteps=1e6
)

runner.run(num_episodes=200)

runner.run(num_episodes=100, evaluation=True)

runner.close()



