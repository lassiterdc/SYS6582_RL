import dl_utils
import numpy as np
import pyswmm.toolkitapi as tkai
import pandas as pd
from tensorforce.agents import Agent
from tensorforce.environments import Environment
import time


#%% hyperparameters
# environment hyperparameters
threshold = 1.75
scaling = 1
action_penalty = 0 # anything other than 0 will just add to penalty if threshold is exceeded. It's already penalized.
advance_seconds = 15 * 60 # how often to take action and evaluate controls

# agent hyperparameters
agent = 'tensorforce'
max_ep_tstep = None
update = dict(unit="timesteps", batch_size="4")
optimizer = dict(optimizer = "adam", learning_rate = 1e-3)
objective = 'policy_gradient'
reward_estimation = dict(horizon = 5)
summarizer = dict(directory='_summaries', summaries = 'all')
saver = dict(directory = '_model', frequency = 10, unit = "episodes")

#%% test baseline
baseline_model_name = 'theta_undvlpd_test'
baseline_config = 'theta_baseline'

baseline_env = dl_utils.swmm_baseline_env(baseline_model_name, baseline_config)
route_step = baseline_env.env.sim._model.getSimAnalysisSetting(tkai.SimulationParameters.RouteStep.value)

done = False
while not done:
    done = baseline_env.step()

baseline_df = dl_utils.create_df_of_outputs(baseline_env, route_step, set_idx_to_hours = False)

#%% test setup
model_name = 'theta_test'
config_name = 'theta'

test_env = dl_utils.custom_tensorflow_env(model_name = model_name,
                                           config_name = config_name,
                                           threshold = threshold, 
                                           scaling = scaling,
                                           action_penalty = action_penalty,
                                           baseline_df = baseline_df.copy(),
                                           advance_seconds = advance_seconds)

route_step = test_env.swmm_env.env.sim._model.getSimAnalysisSetting(tkai.SimulationParameters.RouteStep.value)

tst_env = Environment.create(environment = test_env)

tst_ag = Agent.load(directory='_model-numpy', format='numpy', filename = 'agent', environment = tst_env)
#%% testing
from tensorforce.execution import Runner
num_episodes=1
runner = Runner(
    agent=tst_ag,
    environment=tst_env,
)
runner.run(num_episodes=num_episodes)


tst_ag.close()
tst_env.close()
runner.close()

df = dl_utils.create_df_of_outputs(test_env.swmm_env, route_step)


time = df.index
df = df.set_index([('episode', 0), time])
df.index.rename(['episode', 'time'], inplace = True)


idx = pd.IndexSlice
quants = [0.1, .3 , .5, .7, .9, 1]
eps = np.arange(1, num_episodes+1, 1)

q_cnt = -1
for q in quants:
    q_cnt += 1
    ep = np.quantile(eps, q, interpolation = 'nearest')
    df_temp = df.copy().loc[idx[ep, :], :]
    fig_name = '2_testing_' + str(ep) + '_episodes'
    dl_utils.plt_key_states(fig_name, df_temp.droplevel('episode'), test_env.swmm_env)


#%% plotting uncontrolled scenario
ucntrld_env = dl_utils.swmm_env(model_name = model_name, config_name = config_name,
                                threshold = threshold, scaling = scaling,
                                action_penalty = action_penalty,
                                baseline_df = baseline_df.copy(),
                                advance_seconds = advance_seconds)

route_step = ucntrld_env.env.sim._model.getSimAnalysisSetting(tkai.SimulationParameters.RouteStep.value)

done = False
while not done:
    done, reward = ucntrld_env.step(actions = [1, 1])

fig_name = '1_uncontrolled_test'

df = dl_utils.create_df_of_outputs(ucntrld_env, route_step)
dl_utils.plt_key_states(fig_name, df, ucntrld_env)