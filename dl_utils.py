import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pyswmm.simulation import Simulation
import pyswmm.toolkitapi as tkai
import abc
#import numpy as np
from pystorms.utilities import perf_metrics
import yaml
from tensorforce.environments import Environment

def plt_key_states(fig_name, df, env):
    idx = pd.IndexSlice
    
    fig, axs = plt.subplots(4, 1, sharex = True,
                        figsize = (16, 9))
    
    for att in df.T.index.levels[0]:
    
        if att == 'setting':
            df.loc[:, idx[att, :]].plot(ax=axs[0])
            axs[0].set(ylabel="Frac Open", title='')
            axs[0].legend()
        
        if att == 'depthN':
            df.loc[:, idx[att, :]].plot(ax=axs[1])
            axs[1].set(ylabel="Storage Depth (m)", title='')
            axs[1].legend()
    
#        if att == 'flooding':
#            df.loc[:, idx[att, :]].plot(ax=axs[2])
#            axs[2].set(ylabel="Flood flow (cms)", title='')
#            axs[2].legend()
    
        if att == 'flow':
            df.loc[:, idx[att, :]].plot(ax=axs[2])
            if env.baseline:
                env.baseline_df.loc[:, idx['baseline_flow', :]].plot(ax=axs[2],
                                   color = 'aqua')
            
            axs[2].axhline(env.threshold, c = 'k', ls = '--', label = 'threshold')
            axs[2].set(ylabel="Flow (cms)", title='')
            axs[2].legend()
            
        if att == 'rewards':
            df.loc[:, idx[att, :]].plot(ax=axs[3], legend = False)
            axs[3].set(xlabel='Time (h)', ylabel="Rewards", title='')

    
    plt.tight_layout()
    plt.savefig('plots/' + fig_name + ' .png', dpi = 300, transparent = False)
#    plt.show()
    

def create_df_of_outputs(env, route_step, set_idx_to_hours = True):
    df = pd.DataFrame()
    for key, val in env.data_log.items():
        df_temp = pd.DataFrame(val)
        idx = pd.MultiIndex.from_product([[key], list(df_temp.columns.values)])
        df_temp = df_temp.T.set_index(idx).T
        df = df.join(df_temp, how = 'right')
    
    
    idx = pd.IndexSlice
    midx = pd.MultiIndex.from_tuples(df.T.index.values, names = ['attribute', 'id'])
    df = df.T.set_index(midx).T
    df.index.rename('step', inplace = True)
    if set_idx_to_hours:
        df = df.set_index(df.index.values * route_step / 3600)
        df.index.rename('time_hrs', inplace = True)
        
    df = df.set_index(('time', 0), drop = True)
    df.index.rename('time', inplace = True)
    
    if env.baseline:
        env.baseline_df.rename(columns = {"flow":'baseline_flow'}, inplace = True)
    
    return df


class environment:
    """
    DL - this code borrows heavily from the pystorms library. Only slight modifications were made
    in order to better suit our project.
    """
    
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

    def __init__(self, config, advance_seconds = None): # got rid of ctrl and binary thing
        self.config = config
        self.sim = Simulation(self.config["swmm_input"])
        self.sim.start()
        self.advance_seconds = advance_seconds

        # methods
        self.methods = {
            "depthN": self._getNodeDepth,
            "depthL": self._getLinkDepth,
            "volumeN": self._getNodeVolume,
            "volumeL": self._getLinkVolume,
            "flow": self._getLinkFlow,
            "flooding": self._getNodeFlooding,
            "inflow": self._getNodeInflow,
            "setting": self._getValvePosition # orifice setting
        }

    def _state(self):
        r"""
        Query the stormwater network states based on the config file.
        """
        state = []          
        for _temp in self.config["performance_targets"]:
            ID = _temp[0]
            attribute = _temp[1]
            state.append(self.methods[attribute](ID))
            
        for _temp in self.config["states"]:
            ID = _temp[0]
            attribute = _temp[1]
            state.append(self.methods[attribute](ID))

        state = np.asarray(state)
        
        return state


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
#            print(actions)
#            print(self.config)
#            print(self.initial_state())
            for asset, valve_position in zip(self.config["action_space"], actions):
                self._setValvePosition(asset, valve_position)

        if self.advance_seconds is None:

            time = self.sim._model.swmm_step()

        else:
            prev = self.sim._model.getCurrentSimulationTime()
            time = self.sim._model.swmm_stride((self.advance_seconds))
            elapsed = self.sim._model.getCurrentSimulationTime() - prev
            while elapsed > np.timedelta64((self.advance_seconds)*2, 's'):
                print('skipped ahead')
                print(prev)
                print(self.sim._model.getCurrentSimulationTime())
                self.reset_sim()
                prev = self.sim._model.getCurrentSimulationTime()
                time = self.sim._model.swmm_stride((self.advance_seconds))
                elapsed = self.sim._model.getCurrentSimulationTime() - prev
            
#        time = self.sim._model.swmm_step()
        done = False if time > 0 else True
#        if done:
#            print('Episode completed')
        return done

#    def reset_sim(self):
##        print('reset')
#        r"""
#        Resets the simulation and returns the initial state
#
#        Returns
#        -------
#        initial_state : array
#            initial state in the network
#        """
##        self.end_and_close()
##        self.sim._model.swmm_open()
##        self.sim._model.swmm_start()
#        self.sim.terminate_simulation()
##        self.sim._model.swmm_end()
#        self.sim.close()
#        self.sim.start()
#        
#        state = self._state()
#
#        return state

    def end_and_close(self):
#        print('end and close')
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
    
    
    

class scenario(abc.ABC):
    """
    DL - this code borrows heavily from the pystorms library. Only slight modifications were made
    in order to better suit our project.
    """
    @abc.abstractmethod
    # Specific to the scenario
    def step(self, actions=None, log=True):
        pass

    def _logger(self):
        for attribute in self.data_log.keys():
            if attribute not in ["performance_measure", "simulation_time"]:
                for element in self.data_log[attribute].keys():
                    self.data_log[attribute][element].append(
                        self.env.methods[attribute](element)
                    )

    def state(self):
        return self.env._state()

    def performance(self, metric="cumulative"):
        return perf_metrics(self.data_log["performance_measure"], metric)

    def save(self, path=None):
        if path is None:
            path = "{0}/data_{1}.npy".format("./", self.config["name"])
        return np.save(path, self.data_log)
    

class swmm_baseline_env(scenario):
    def __init__(self, baseline_model_name, baseline_config):
        self.config = yaml.load(open("config/" + baseline_config + ".yaml", "r"), yaml.FullLoader)
        self.config["swmm_input"] = baseline_model_name + '.inp'

        self.env = environment(self.config)
        self.tstep = self.env.sim._model.getCurrentSimulationTime()

        self.data_log = {"flow": {}, "time":[]}
            
        for ID, attribute in self.config["states"]:
            self.data_log["flow"][ID] = []
            
        self.baseline = False
            
    def logger(self):
        for attribute in self.data_log.keys():
            if attribute not in ["rewards", "simulation_time", "time"]:
                for element in self.data_log[attribute].keys():
                    self.data_log[attribute][element].append(
                        self.env.methods[attribute](element))
                    
    def step(self, log=True):
        done = self.env.step()
        self.tstep = self.env.sim._model.getCurrentSimulationTime()
        if log:
            self.logger()
        self.data_log['time'].append(self.tstep)
        return done
    
class swmm_env(scenario):
    """
    Model name is the model .inp name WITHOUT the .inp extension
    """
    def __init__(self, model_name, config_name, threshold, scaling, action_penalty = 0,
                 baseline_df = None, advance_seconds = None):
        # Network configuration
        self.config = yaml.load(open("config/" + config_name + ".yaml", "r"), yaml.FullLoader)
        self.config["swmm_input"] = model_name + '.inp'
        self.scaling = scaling
        self.threshold = threshold
        self.action_penalty = action_penalty
        self.advance_seconds = advance_seconds

        # Create the environment based on the physical parameters
        self.env = environment(self.config, advance_seconds = advance_seconds)
        self.tstep = self.env.sim._model.getCurrentSimulationTime()

        # Create an object for storing the data points
        self.data_log = {"rewards": [], "flow": {}, "flooding": {},
                         'depthN':{}, 'setting':{}, "time":[],
                         'episode':[]}

        # Data logger for storing _performance data
        for ID, attribute in self.config["performance_targets"]:
            self.data_log[attribute][ID] = []
            
        for ID, attribute in self.config["states"]:
            self.data_log[attribute][ID] = []
        
        self.baseline = False
        if baseline_df is not None:
            self.baseline = True
            self.baseline_df = baseline_df
            
    def reset_swmm_env(self):
#        self.env.end_and_close()
        self.env = environment(self.config, advance_seconds = self.advance_seconds)
        self.tstep = self.env.sim._model.getCurrentSimulationTime()
        
        state = self.env._state()
        return state
        
        
            
    def logger(self):
        for attribute in self.data_log.keys():
            if attribute not in ["rewards", "simulation_time", "time", "episode"]:
                for element in self.data_log[attribute].keys(): # ID
                    self.data_log[attribute][element].append(
                        self.env.methods[attribute](element))
                    

    def step(self, actions=None, log=True, episode = 0):
        prev_actions = []
        for asset in self.config["action_space"]:
                prev_actions.append(self.env._getValvePosition(asset))

        
#            self.prev_ac_tstep = self.tstep
        done = self.env.step(actions)
        


        # Log the flows in the networks
        if log:
            self.logger()
            
        
            
        self.tstep = self.env.sim._model.getCurrentSimulationTime()
        self.data_log['time'].append(self.tstep)
        self.data_log['episode'].append(episode)

        reward = 0
        # penalize for taking action
        for ac, prev_ac in zip(actions, prev_actions):
            if abs(ac - prev_ac) > 0.01:
                reward += self.action_penalty
        
        idx = pd.IndexSlice
        
        for ID, attribute in self.config["performance_targets"]:
            if attribute == "flooding":
                __flood = self.env.methods[attribute](ID)
                if __flood > 0.0:
                    reward += __flood * -1 * self.scaling
            if attribute == "flow":
                __flow = self.env.methods[attribute](ID)
                if self.baseline: 
                    try:
                        flow_bsln = self.baseline_df.copy().loc[self.tstep, idx[attribute, ID]]
                        # don't penalize if natural conditions exceed threshold
                        if flow_bsln > self.threshold: 
                            flow_bsln = self.threshold
                        reward -= abs(flow_bsln - __flow) ** 2
                    except:
                        pass

                        
#                        if __flow > 0:
#                            print("Basline: " + str(flow_blsn))
#                            print('Modified: ' + str(__flow))
                    
                
                if __flow <= self.threshold:
                    reward += 0.0
                else:
                    reward += (__flow - self.threshold) * self.scaling * -1
        
#        print(reward)
        self.data_log["rewards"].append(reward)
            
#        if done:
#            print('In swmm_env, made it to last step of sim.')
        return done, reward

class custom_tensorflow_env(Environment):

    def __init__(self, model_name, config_name, threshold, scaling = 1,
                 action_penalty = 0, baseline_df = None, advance_seconds = None):
        
        
        super().__init__()
        self.swmm_env = swmm_env(model_name, config_name, threshold, scaling,
                                 action_penalty, baseline_df, advance_seconds)
        
        self.ep = 0
#        print('starting episode ' + str(self.ep))
    

    def states(self):
        state_count = 0 # include states and performance targets
        for ID, attribute in self.swmm_env.config["performance_targets"]:
            state_count += 1
            
        for ID, attribute in self.swmm_env.config["states"]:
            state_count += 1
        
        return dict(type='float', shape=(state_count,))
    # flood volume in P1 and P2
    # depth in P1 and P2
    # amount that P1 is open and amount that P2 is open
    # link 8 flowrate

    def actions(self):
        action_count = 0
        for ID in self.swmm_env.config["action_space"]:
            action_count += 1
        return dict(type='float', shape=(action_count,), min_value = 0, max_value = 1)

    # Optional: should only be defined if environment has a natural fixed
    # maximum episode length; otherwise specify maximum number of training
    # timesteps via Environment.create(..., max_episode_timesteps=???)
    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    # Optional additional steps to close environment
    def close(self):
#        try:
#            self.swmm_env.env_bsln.end_and_close()
#        except:
#            pass
        
        self.swmm_env.env.end_and_close()



    def reset(self):
        self.ep += 1
        print('starting episode ' + str(self.ep))
        
        return self.swmm_env.reset_swmm_env()
    

    def execute(self, actions):
#        print('executing...')
        prev_state = self.swmm_env.env._state()
        
        terminal, reward = self.swmm_env.step(actions, episode = self.ep) # this automatically steps forward the baseline
        if terminal:
            next_state = prev_state
        else:
            next_state = self.swmm_env.env._state()
        
        return next_state, terminal, reward
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    