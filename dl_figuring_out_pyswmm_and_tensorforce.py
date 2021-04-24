#%% import libraries
import numpy as np
from tensorforce.environments import Environment
from tensorforce.agents import Agent
from pyswmm import Simulation, Nodes, Links
import pyswmm.toolkitapi as tkai
import random

#%% run simulation
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
   
#%% plot simulation results
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

#%% work - blank class to test env creation
class test:
    def __init__(self):
        pass
#%% environment class pasted and modified from pystorm
import numpy as np
from pyswmm.simulation import Simulation
import pyswmm.toolkitapi as tkai

class environment:
    r"""Environment for controlling the swmm simulation

    This class acts as an interface between swmm's simulation
    engine and computational components. This class's methods are defined
    as getters and setters for generic stormwater attributes. So that, if need be, this
    class can be updated with a different simulation engine, keeping rest of the
    workflow stable.


    Attributes
    ----------
    config : dict
        dictionary with swmm_ipunt and, action and state space `(ID, attribute)`
    ctrl : boolean
        if true, config has to be a dict, else config needs to be the path to the input file

    Methods
    ----------
    step
        steps the simulation forward by a time step and returns the new state
    initial_state
        returns the initial state in the stormwater network
    terminate
        closes the swmm simulation
    reset
        closes the swmm simulaton and start a new one with the predefined config file.
    """

    def __init__(self, config): # got rid of ctrl and binary thing
        self.config = config
        self.sim = Simulation(self.config["swmm_input"])
        self.sim.start()

        # methods
        self.methods = {
            "depthN": self._getNodeDepth,
            "depthL": self._getLinkDepth,
            "volumeN": self._getNodeVolume,
            "volumeL": self._getLinkVolume,
            "flow": self._getLinkFlow,
            "flooding": self._getNodeFlooding,
            "inflow": self._getNodeInflow,
        }

    def _state(self):
        r"""
        Query the stormwater network states based on the config file.
        """
        state = []
        for _temp in self.config["states"]:
            ID = _temp[0]
            attribute = _temp[1]
            state.append(self.methods[attribute](ID))

        state = np.asarray(state)


    def step(self, actions=None):
        r"""
        Implements the control action and forwards
        the simulation by a step.

        Parameters:
        ----------
        actions : list or array
            actions to take as an array (1 x n)
        
        n = size of action space

        Returns:
        -------
        new_state : array
            next state
        done : boolean
            event termination indicator
        """
        if actions is not None:
            # implement the actions
            for asset, valve_position in zip(self.config["action_space"], actions):
                self._setValvePosition(asset, valve_position)

        # take the step !
        time = self.sim._model.swmm_step()
        done = False if time > 0 else True
        return done

    def reset(self):
        r"""
        Resets the simulation and returns the initial state

        Returns
        -------
        initial_state : array
            initial state in the network
        """
        self.terminate()

        # Start the next simulation
        self.sim._model.swmm_open()
        self.sim._model.swmm_start()

        # get the state
        state = self._state()
        return state

    def terminate(self):
        r"""
        Terminates the simulation
        """
        self.sim._model.swmm_end()
        self.sim._model.swmm_close()

    def initial_state(self):
        r"""
        Get the initial state in the stormwater network

        Returns
        -------
        initial_state : array
            initial state in the network
        """

        return self._state()

    # ------ Node Parameters  ----------------------------------------------
    def _getNodeDepth(self, ID):
        return self.sim._model.getNodeResult(ID, tkai.NodeResults.newDepth.value)

    def _getNodeFlooding(self, ID):
        return self.sim._model.getNodeResult(ID, tkai.NodeResults.overflow.value)

    def _getNodeLosses(self, ID):
        return self.sim._model.getNodeResult(ID, tkai.NodeResults.losses.value)
    
    def _getNodeVolume(self, ID):
        return self.sim._model.getNodeResult(ID, tkai.NodeResults.newVolume.value)

    def _getNodeInflow(self, ID):
        return self.sim._model.getNodeResult(ID, tkai.NodeResults.totalinflow.value)

    def _setInflow(self, ID, value):
        return self.sim._model.setNodeInflow(ID, value)

    # ------ Valve modifications -------------------------------------------
    def _getValvePosition(self, ID):
        return self.sim._model.getLinkResult(ID, tkai.LinkResults.setting.value)

    def _setValvePosition(self, ID, valve):
        self.sim._model.setLinkSetting(ID, valve)

    # ------ Link modifications --------------------------------------------
    def _getLinkDepth(self, ID):
        return self.sim._model.getLinkResult(ID, tkai.LinkResults.newDepth.value)

    def _getLinkVolume(self, ID):
        return self.sim._model.getLinkResult(ID, tkai.LinkResults.newVolume.value)

    def _getLinkFlow(self, ID):
        return self.sim._model.getLinkResult(ID, tkai.LinkResults.newFlow.value)
#%% pasted scenario theta class creation from pystorm
#from pystorms.environment import environment
#from pystorms.networks import load_network
#from pystorms.config import load_config
#from pystorms.scenarios import scenario
#from pystorms.utilities import threshold
import yaml

class swmm_env:
    """
    Model name is the model .inp name WITHOUT the .inp extension
    """
    def __init__(self, model_name, threshold):
        # Network configuration
        self.config = yaml.load(open("config/" + model_name + ".yaml", "r"), yaml.FullLoader)
        self.config["swmm_input"] = model_name + '.inp'

        self.threshold = threshold

        # Create the environment based on the physical parameters
        self.env = environment(self.config, ctrl=True)

        # Create an object for storing the data points
        self.data_log = {"performance_measure": [], "flow": {}, "flooding": {}}

        # Data logger for storing _performance data
        for ID, attribute in self.config["performance_targets"]:
            self.data_log[attribute][ID] = []

    def step(self, actions=None, log=True):
        # Implement the actions and take a step forward
        done = self.env.step(actions)

        # Log the flows in the networks
        if log:
            self._logger()

        # Estimate the performance
        __performance = 0.0

        for ID, attribute in self.config["performance_targets"]:
            if attribute == "flooding":
                __flood = self.env.methods[attribute](ID)
                if __flood > 0.0:
                    __performance += 10 ** 6
            if attribute == "flow":
                __flow = self.env.methods[attribute](ID)
                __performance = threshold(
                    value=__flow, target=self.threshold, scaling=10.0
                )

        # Record the _performance
        self.data_log["performance_measure"].append(__performance)

        # Terminate the simulation
        if done:
            self.env.terminate()

        return done


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
