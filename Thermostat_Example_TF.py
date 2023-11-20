# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 12:18:40 2021

@author: JakeNelson
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 13:11:49 2021

@author: JakeNelson
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import pyswmm.toolkitapi as tkai
from pyswmm import Simulation, Nodes, Links, Subcatchments, LidControls

## Compute the response for a given action and current temperature
# def respond(action, current_temp, tau):
#     return action + (current_temp - action) * math.exp(-1.0/tau)

# ## Actions of a series of on, then off
# sAction = pd.Series(np.array([1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]))
# sResponse = np.zeros(sAction.size)

# ## Update the response with the response to the action
# for i in range(sAction.size):
#     ## Get last response
#     if i == 0:
#         last_response = 0
#     else:
#         last_response = sResponse[i - 1]
#     sResponse[i] = respond(sAction[i], last_response, 3.0)

# ## Assemble and plot
# df = pd.DataFrame(list(zip(sAction, sResponse)), columns=['action', 'response'])
# df.plot()

# ## goal and reward
# def reward(temp):
#         delta = abs(temp - 0.5)
#         if delta < 0.1:
#             return 0.0
#         else:
#             return -delta + 0.1

# temps = [x * 0.01 for x in range(100)]
# rewards = [reward(x) for x in temps]

# fig=plt.figure(figsize=(12, 4))

# plt.scatter(temps, rewards)
# plt.xlabel('Temperature')
# plt.ylabel('Reward')
# plt.title('Reward vs. Temperature')

###-----------------------------------------------------------------------------
## Imports
from tensorforce.environments import Environment
from tensorforce.agents import Agent




###-----------------------------------------------------------------------------
### Environment definition
class SWMM_env(Environment):
    """This class defines a simple thermostat environment.  It is a room with
    a heater, and when the heater is on, the room temperature will approach
    the max heater temperature (usually 1.0), and when off, the room will
    decay to a temperature of 0.0.  The exponential constant that determines
    how fast it approaches these temperatures over timesteps is tau.
    """
   # J: current_temp could be changed to current_flow which is Links(sim)["8"].flow*35.3147
   # J: this is where we define everything and start the swmm model
    def __init__(self, inp_file): # the max depth of the ponds is 2 meters. This I believe is also an intial setting
        ## Some initializations.  Will eventually parameterize this in the constructor.
        self.modelfile = inp_file
        self.sim = Simulation(self.modelfile)
        
        self.controlstep = 900 #in seconds
        self.sim.step_advance(self.controlstep)
        node_object = Nodes(self.sim)
        self.P1 = node_object["P1"]
        self.P2 = node_object["P2"]
        #self.depth = depth
        # self.P1.depth = self.depth
        # self.P2.depth = self.depth
        
        # init link objects - includes downstream link and both orifices
        link_object = Links(self.sim)
        self.L8 = link_object["8"]
        #self.flow = flow
        #self.L8.flow = self.L8.flow
        self.O1 = link_object["1"]
        self.O2 = link_object["2"]
        
        
        self.sim.start()
        if self.sim.current_time == self.sim.start_time:
            #initially sets the orifices at half way open
            self.O1.target_setting = 0.5
            self.O2.target_setting = 0.5
        sim_len = self.sim.end_time - self.sim.start_time
        self.T = int(sim_len.total_seconds()/self.controlstep)
        self.state = np.concatenate([np.asarray([self.P1.depth, self.P2.depth, self.L8.flow, 
                        self.P1.flooding, self.P2.flooding, self.O1.current_setting, self.O2.current_setting])])
        super().__init__()


# J: I think this should consider the 6 states referenced in the self.state line above. 
    # def states(self):
    #     return dict(type='float', shape=(6,))
    
    def states(self):
        return dict(type='float', shape=(7,))

# J: would the actions be the position of the valves at P1 and P2? Would we need to figure
# J: out how to program the lid for this section, given two valve positions?
# need to add a min and a max value to capture everything between 0 and 1
    def actions(self):
        """Action 0 means no heater, temperature approaches 0.0.  Action 1 means
        the heater is on and the room temperature approaches 1.0.
        """
        return dict(type='float', num_values=2)


    #(original comment) Optional, should only be defined if environment has a natural maximum
    #(original comment) episode length
    # I can change the frequency (maybe 15 minutes, maybe something else)
    # def max_episode_timesteps(self):
    #     return super().max_episode_timesteps()


### J: Not sure how this works with the swmm model.
    # Optional
    def close(self):
        self.sim.report()
        self.sim.close()

### J: Not sure how this works with the swmm model.
    def reset(self):
        """Reset state.
        """
        # state = np.random.random(size=(1,))
        
        self.sim.close()
        self.sim = Simulation(self.modelfile)
        self.sim.step_advance(self.controlstep)
        node_object = Nodes(self.sim)
        self.P1 = node_object["P1"]
        self.P2 = node_object["P2"]
        self.P1.depth = self.depth
        self.P2.depth = self.depth
        
        link_object = Links(self.sim)
        self.L8 = link_object["8"]
        self.L8.flow = self.flow
        self.sim.start()
        
        if self.sim.current_time == self.sim.start_time:
            self.L8.target_setting = 0
            
        self.state = np.concatenate([np.asarray([self.P1.depth, self.P2.depth, self.L8.flow, 
                        self.P1.flooding, self.P2.flooding, self.O1.current_setting, self.O2.current_setting])])
        
        return self.state

# J: The response should be the total flow in Pipe8, correct? We might not need this piece
    # def response(self, action):
    #     """Respond to an action.  When the action is 1, the temperature
    #     exponentially decays approaches 1.0.  When the action is 0,
    #     the current temperature decays towards 0.0.
    #     """
    #     return action + (self.current_temp - action) * math.exp(-1.0 / self.tau)

# J: The reward is 0 if the flow in Pipe8 is betwee, X1 and X2, or else we could have
# J: it negatively increase as the flow increases/decreases beyond the boundary. 
    def reward_compute(self):
        """ The reward here is 0 if the current temp is between 0.4 and 0.6,
        else it is distance the temp is away from the 0.4 or 0.6 boundary.
        
        Return the value within the numpy array, not the numpy array.
        """
        
        # the reward should take into consideration flow being outside of desired range; should it also consider flooding?
        # reward needs to be based on the state - need to pass in new state information to calculate reward
        delta = abs(self.L8.flow- 0.25)
        if delta < 0.1:
            return 0.0
        else:
            return -delta[0] + 0.1

# check this with the step function in Ben's code
    def execute(self, actions):
        ## Check the action is either 0 or 1 -- heater on or off.
        
        self.O1.target_setting = actions[0]
        self.O2.target_setting = actions[1]
        
        self.sim.__next__()
        self.state = np.concatenate([np.asarray([self.P1.depth,self.P2.depth,self.P1.flooding,
                                                 self.P2.flooding,self.L8.current_setting])])

        # ## Increment timestamp
        # self.timestep += 1
        
        # ## Update the current_temp
        # self.current_temp = self.response(actions)
        
        ## Compute the reward
        reward = self.reward_compute()

        ## The only way to go terminal is to exceed max_episode_timestamp.
        ## terminal == False means episode is not done
        ## terminal == True means it is done.
        terminal = False
        
        return self.state, terminal, reward 
    

###-----------------------------------------------------------------------------
### Create the environment
###   - Tell it the environment class
###   - Set the max timestamps that can happen per episode
# removed timestamps as the model should take care of that.

environment = environment = Environment.create(
    environment=SWMM_env('theta.inp'))

# agent setup
agent = Agent.create(
    agent='tensorforce', environment=environment, update=64,
    optimizer=dict(optimizer='adam', learning_rate=1e-3),
    objective='policy_gradient', reward_estimation=dict(horizon=1)
)

# Check untrained agent performance 
### Initialize
environment.reset()

## Creation of the environment via Environment.create() creates
## a wrapper class around the original Environment defined here.
## That wrapper mainly keeps track of the number of timesteps.
## In order to alter the attributes of your instance of the original
## class, like to set the initial temp to a custom value, like here,
## you need to access the `environment` member of this wrapped class.
## That is why you see the way to set the current_temp like below.
# environment.current_temp = np.array([0.5])
# states = environment.current_temp

internals = agent.initial_internals()
terminal = False

### Run an episode
temp = [environment.state[2]]
while not terminal:
    actions, internals = agent.act(states= environment.state, internals=internals, independent=True)
    states, terminal, reward = environment.execute(actions=actions)
    #temp += [states[0]]
    
### Plot the run
plt.figure(figsize=(12, 4))
ax=plt.subplot()
ax.set_ylim([0.0, 1.0])
plt.plot(range(len(temp)), temp)
plt.hlines(y=0.4, xmin=0, xmax=99, color='r')
plt.hlines(y=0.6, xmin=0, xmax=99, color='r')
plt.xlabel('Timestep')
plt.ylabel('Temperature')
plt.title('Temperature vs. Timestep')
plt.show()

# Train the agent
# Train for 200 episodes
for _ in range(200):
    states = environment.reset()
    terminal = False
    while not terminal:
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)
        
# Check the trained agent performance
        

### Initialize
environment.reset()

## Creation of the environment via Environment.create() creates
## a wrapper class around the original Environment defined here.
## That wrapper mainly keeps track of the number of timesteps.
## In order to alter the attributes of your instance of the original
## class, like to set the initial temp to a custom value, like here,
## you need to access the `environment` member of this wrapped class.
## That is why you see the way to set the current_temp like below.
environment.current_temp = np.array([1.0])
states = environment.state

internals = agent.initial_internals()
terminal = False

### Run an episode
#temp = [environment.current_temp[0]]
while not terminal:
    actions, internals = agent.act(states=states, internals=internals, independent=True)
    states, terminal, reward = environment.execute(actions=actions)
    #temp += [states[0]]

### Plot the run
plt.figure(figsize=(12, 4))
ax=plt.subplot()
ax.set_ylim([0.0, 1.0])
plt.plot(range(len(temp)), temp)
plt.hlines(y=0.4, xmin=0, xmax=99, color='r')
plt.hlines(y=0.6, xmin=0, xmax=99, color='r')
plt.xlabel('Timestep')
plt.ylabel('Temperature')
plt.title('Temperature vs. Timestep')
plt.show()