import numpy as np
from tensorforce.environments import Environment
from tensorforce.agents import Agent
from pyswmm import Simulation, Nodes, Links
import pyswmm.toolkitapi as tkai
import random

# nodes module: https://pyswmm.readthedocs.io/en/stable/reference/nodes.html
# links module: https://pyswmm.readthedocs.io/en/stable/reference/links.html
with Simulation('theta.inp') as sim:
    random.seed(10)
    step_size = 30 # seconds
    setting_adjust_freq = 60 # minutes
    nodes = Nodes(sim)
    links = Links(sim)
    
    p1 = nodes['P1']
    p2 = nodes['P2']
    l8 = links['8']
    l2 = links['2']
    
    # set up state lists
    ## orifice setting (frac open)
    p1_setting = list()
    p2_setting = list()
    ## storage depth
    p1_depth = list()
    p2_depth = list()
    ## storage flooding
    p1_fld = list()
    p2_fld = list()
    ## downstream link flowrate
    l8_flow = list()
    
    # possible settings
    actions = [0, 0.25, 0.5, 1]
    
    step_cnt = -1
    for step in sim:
        step_cnt += 1
        # states
        # orifice settings
        p1_setting.append(sim._model.getLinkResult('1', tkai.LinkResults.setting.value))
        p2_setting.append(sim._model.getLinkResult('2', tkai.LinkResults.setting.value))
        
        p1_depth.append(p1.depth)
        p2_depth.append(p2.depth)
        
        p1_fld.append(p1.flooding)
        p2_fld.append(p1.flooding)
        
        l8_flow.append(l8.flow)
        
        if step_cnt*step_size % (setting_adjust_freq*60) == 0: # only adjust orifice settings every 5 min
            s_p1 = random.choice(actions)
            s_p2 = random.choice(actions)
            sim._model.setLinkSetting('1', s_p1) # setting for p1
            sim._model.setLinkSetting('2', s_p2) # setting for p2
        
        
        stats_p1 = p1.statistics
        stats_p2 = p2.statistics
        stats_l8 = l8.conduit_statistics
         # action
        pass


pf_l8 = stats_l8['peak_flow']
fld_p1 = stats_p1['flooding_volume']
fld_p2 = stats_p2['flooding_volume']
   
#%%
import matplotlib.pyplot as plt
time_h = np.arange(0, step_cnt+1, 1) * step_size / 60 / 60 

fig, axs = plt.subplots(4, 1, sharex = True,
                        figsize = (16, 9))

# settings
s1 = np.asarray(p1_setting)
s2 = np.asarray(p2_setting)

axs[0].plot(time_h, s1, label = 'P1')
axs[0].plot(time_h, s2, label = 'P2')
axs[0].set(ylabel="Orifice settings (frac open)", title='')
axs[0].legend()

# depths
d1 = np.asarray(p1_depth)
d2 = np.asarray(p2_depth)

axs[1].plot(time_h, d1, label = 'P1')
axs[1].plot(time_h, d2, label = 'P2')
axs[1].set(ylabel="Storage Depth (m)", title='')
axs[1].legend()
# flooding
fld1 = np.asarray(p1_fld)
fld2 = np.asarray(p2_fld)

axs[2].plot(time_h, fld1, label = 'P1')
axs[2].plot(time_h, fld2, label = 'P2')
axs[2].set(ylabel="Flood flow (cms)", title='')
axs[2].legend()
# flow
f18 = np.asarray(l8_flow)

axs[3].plot(time_h, f18, label = 'Link 8')
axs[3].set(xlabel='Time (h)', ylabel="Flow (cms)", title='')
axs[3].legend()

plt.tight_layout()


plt.savefig('plots/test.png', dpi = 300, transparent = False)
plt.show()

#%% creating environment
class pyswmm_env(Environment):

    def __init__(self):
        super().__init__()

    def states(self):
        return dict(type='float', shape=(7,))
    # flood volume in P1 and P2
    # depth in P1 and P2
    # amount that P1 is open and amount that P2 is open
    # link 8 flowrate

    def actions(self):
        return dict(type='int', num_values=4)

    # Optional: should only be defined if environment has a natural fixed
    # maximum episode length; otherwise specify maximum number of training
    # timesteps via Environment.create(..., max_episode_timesteps=???)
    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    # Optional additional steps to close environment
    def close(self):
        super().close()

    def reset(self):
        state = np.random.random(size=(8,))
        return state

    def execute(self, actions):
        next_state = np.random.random(size=(8,))
        terminal = np.random.random() < 0.5
        reward = np.random.random()
        return next_state, terminal, reward

#%% creating environment - thermostat example
class pyswmm_env(Environment):
    def __init__(self):
        self.tau = 3.0
        self.current_temp = np.random.random(size=(1,))
        super().__init__()


# J: would the states be the depth in P1 and P2?
    def states(self):
        return dict(type='float', shape=(1,), min_value=0.0, max_value=1.0)

# J: would the actions be the position of the valves at P1 and P2? Would we need to figure
# J: out how to program the lid for this section, given two valve positions?
# need to add a min and a max value to capture everything between 0 and 1
    def actions(self):
        """Action 0 means no heater, temperature approaches 0.0.  Action 1 means
        the heater is on and the room temperature approaches 1.0.
        """
        return dict(type='int', num_values=2)


    #(original comment) Optional, should only be defined if environment has a natural maximum
    #(original comment) episode length
    # I can change the frequency (maybe 15 minutes, maybe something else)
    def max_episode_timesteps(self):
        return super().max_episode_timesteps()


### J: Not sure how this works with the swmm model.
    # Optional
    def close(self):
        super().close()

### J: Not sure how this works with the swmm model.
    def reset(self):
        """Reset state.
        """
        # state = np.random.random(size=(1,))
        self.timestep = 0
        self.current_temp = np.random.random(size=(1,))
        return self.current_temp

# J: The response should be the total flow in Pipe8, correct? We might not need this piece
    def response(self, action):
        """Respond to an action.  When the action is 1, the temperature
        exponentially decays approaches 1.0.  When the action is 0,
        the current temperature decays towards 0.0.
        """
        return action + (self.current_temp - action) * math.exp(-1.0 / self.tau)

# J: The reward is 0 if the flow in Pipe8 is betwee, X1 and X2, or else we could have
# J: it negatively increase as the flow increases/decreases beyond the boundary. 
    def reward_compute(self):
        """ The reward here is 0 if the current temp is between 0.4 and 0.6,
        else it is distance the temp is away from the 0.4 or 0.6 boundary.
        
        Return the value within the numpy array, not the numpy array.
        """
        delta = abs(self.current_temp - 0.5)
        if delta < 0.1:
            return 0.0
        else:
            return -delta[0] + 0.1

# check this with the step function in Ben's code
    def execute(self, actions):
        ## Check the action is either 0 or 1 -- heater on or off.
        assert actions == 0 or actions == 1

        ## Increment timestamp
        self.timestep += 1
        
        ## Update the current_temp
        self.current_temp = self.response(actions)
        
        ## Compute the reward
        reward = self.reward_compute()

        ## The only way to go terminal is to exceed max_episode_timestamp.
        ## terminal == False means episode is not done
        ## terminal == True means it is done.
        terminal = False
        
        return self.current_temp, terminal, reward
