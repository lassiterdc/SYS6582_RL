# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 13:45:15 2021

@author: JakeNelson
"""

import pystorms
import pandas as pd
import pyswmm.toolkitapi as tkai
import matplotlib.pyplot as plt
import numpy as np
from pyswmm import Simulation, Nodes, Links, Subcatchments, LidControls

# def getlinkdepth(LinkID):
#     return env.env.sim._model.getLinkResult(LinkID,tkai.LinkResults.newDepth.value)

# def getflow(NodeID):
#     return env.env.sim._model.getNodeResult(NodeID,tkai.NodeResults.outflow.value)


# def getlinkdepth(LinkID):
#     return env.env.sim._model.getLinkResult(LinkID,tkai.LinkResults.newDepth.value)

# def getflow(NodeID):
#     return env.env.sim._model.getNodeResult(NodeID,tkai.NodeResults.outflow.value)
###################################################################################################################
# def _state(self):
#     r"""
#     Query the stormwater network states based on the config file.
#     """
#     if self.ctrl:
#         state = []
#         for _temp in self.config["states"]:
#             ID = _temp[0]
#             attribute = _temp[1]

#             if attribute == "pollutantN" or attribute == "pollutantL":
#                 pollutant_index = _temp[2]
#                 state.append(self.methods[attribute](ID, pollutant_index))
#             else:
#                 state.append(self.methods[attribute](ID))

#         state = np.asarray(state)
#         return state
#     else:
#         raise NameError("State config not defined !")

####################################################################################################################
def controller(state, target, MAX=2.0):
    fd = state / MAX
    avg_fd = np.mean(fd)
    potential = fd - avg_fd  # [<0, 0, <1]

    for i in range(0, 2):
        if potential[i] < -0.001:
            potential[i] = 0.0
        elif potential[i] < 0.001 and potential[i] > -0.001:
            potential[i] = avg_fd

    if sum(potential) > 0.0:
        potential = potential / sum(potential)

    actions = np.zeros(2)
    if state[0] > 0.00:
        flow0 = target * potential[0]
        actions[0] = min(1.0, flow0 / (1.00 * np.sqrt(2.0 * 9.81 * state[0])))
    if state[1] > 0.00:
        flow1 = target * potential[1]
        actions[1] = min(1.0, flow1 / (1.00 * np.sqrt(2.0 * 9.81 * state[1])))
    return actions

#######################################################################################################################

# def respond(action, current_flow):
#     return 0/(1.00 * np.sqrt(2.0 * 9.81 * state[0])

######################################################################################################################


time = []
nodedepthP1 = []
nodedepthP2 = []
nodeflowO = []
Pipe8flow = []
i = 0
for i in range(12575):
    time.append(i)
    i = i+1

with Simulation('C:/Users/JakeNelson/anaconda3/Lib/site-packages/pystorms/networks/theta.inp') as sim:
    Pipe8 = Links(sim)["8"]
    for step in sim:
        Pipe8flow.append(Pipe8.flow*35.3147)
        nodedepthP1.append(Nodes(sim)["P1"].depth*3.28084)
        nodedepthP2.append(Nodes(sim)["P2"].depth*3.28084)
        
        # done = False
        # while not done:
        #    state = [nodedepthP1,nodedepthP2]
        #    actions = controller(state, 0.50)
        #    done = sim.step(actions)
        

       
        
        
        
    
   # # call out the two storage nodes
   #  P1 = Nodes(env)["P1"]
   #  P2 = Nodes(env)["P2"]
   #  O = Nodes(env)["O"] #outfall
        
       


        # done = False
        # while not done:
        #     for s in state:
        #         state += state
        #         #print(state)
        #         actions = controller(state, 0.50)
        #         #print(actions)
        #         mydata.append(state)
        #         done = env.step(actions) #actions gives it the dimensions (an array of 2 places)
        
        
        
# nodedepthP1.append(P1.depth*3.28084)
# nodedepthP2.append(P2.depth*3.28084)
# nodeflowO.append(O.peak_flowrate*0.264172)
    #add controls
    #save to lists
    #use the report file as much as possible to read the results
    #incremental total flood volume can't be obtained from report file - must be saved separatly during simulation.
    #
pass

flow = {"timestep":time,'Pipe8flow': Pipe8flow, "nodedepthP1":nodedepthP1, "nodedepthP2":nodedepthP2 }
pipeflowdf = pd.DataFrame(flow, columns=['timestep','Pipe8flow','nodedepthP1','nodedepthP2'])

pipeflowdf.plot(x='timestep', y='nodedepthP1',kind = 'line')
pipeflowdf.plot(x='timestep', y='nodedepthP2',kind = 'line')
pipeflowdf.plot(x='timestep', y='Pipe8flow',kind = 'line')

# data = pd.DataFrame(data, columns=["depthP1", "depthP2","depthL7","depthL9"])
# data.plot()

# plt.plot(env_controlled.data_log["flow"]["8"], label="Controlled")
# #plt.plot(env_controlled.data_log["flow"]["8"], label="Uncontrolled")
# plt.ylabel("Outflows")
# plt.legend()
#data = pd.DataFrame(data, columns=["depthP1", "depthP2", "volumeP1", "volumeP2","depthL7","depthL9"])

    

# env = pystorms.scenarios.theta()
# done = False
# while not done:
#     done = env.step(np.ones(2))
# print("Uncontrolled Performance : {}".format(env.performance()))

# plt.plot(env.data_log["flow"]["8"])
# plt.ylabel("Outflows")
######################################################################################################################    
# Initalize scenario
#env = pystorms.scenarios.theta()


# Update state vector - this is where the data is 
# env_controlled.env.config["states"].append(("P1", "volumeN")) # you get to choose the name of the "P1". 
# env_controlled.env.config["states"].append(("P2", "volumeN"))
##################################################################################################################################
# env.config["states"].append(("7", "depthL"))
# env.config["states"].append(("9", "depthL"))
# env.config["states"].append(("8", "outflow"))

######################################################################################################################################
# adding precip to states - the "1" is the actual ID from the swmm code in the theta simulation. We looked at the actual swmm model that theta is running. went to C:\Users\JakeNelson\anaconda3\Lib\site-packages\pystorms\networks 


# Update the methods dict
#env_controlled.env.methods["volumeN"] = getNodeVolume
# adding the precip to the methods by calling getprecip function above
# methods is a dictionary that has "precip" as a key, that gets you that function getprecip. 
#env.env.methods["precip"] = getprecip
#########################################################################################################################
# env.env.methods["depthL"] = getlinkdepth
# env.env.methods["outflow"] = getflow

#########################################################################################################################
#to find state information: env.env.config["states"] which gives node id's and structure




#env.env.config["states"].append(("1", "precip"))





# empty list; we know the default values are depth at p1 and p2, so those are listed first, then we append volumeN values for p1 and p2, and then we add precip at the same time steps
# mydata = []


# done = False
# while not done:
#     state = env.state()
#     #print(state)
#     actions = controller(state, 0.50)
#     #print(actions)
#     mydata.append(state)
#     done = env.step(actions) #actions gives it the dimensions (an array of 2 places)


# # plt.plot(mydata[],label="depthP1")
# # plt.plot(env.data_log["flow"]["8"], label="flow")
# # plt.plot(env.data_log["flow"]["8"], label="Uncontrolled")
# # plt.ylabel("Outflows")
# # plt.legend()

# # plt.plot(state)


# mydata = pd.DataFrame(mydata, columns=["depthP1", "depthP2","depthL7","depthL9","flow8"])
# mydata.plot()















# data = []
# while not done:
#     state = env_controlled.state()
#     actions = controller(state, 0.50)
#     done = env_controlled.step([1, 1])
#     data.append(state)
#     done = env_controlled.step(actions)

# # while not done:
# #     state = env_controlled.state()
# #     actions = controller(state, 0.50)
# #     print(actions)
# #     done = env_controlled.step(actions)

# data = pd.DataFrame(data, columns=["depthP1", "depthP2","depthL7","depthL9"])
# data.plot()

# plt.plot(env_controlled.data_log["flow"]["8"], label="Controlled")
# #plt.plot(env_controlled.data_log["flow"]["8"], label="Uncontrolled")
# plt.ylabel("Outflows")
# plt.legend()
#data = pd.DataFrame(data, columns=["depthP1", "depthP2", "volumeP1", "volumeP2","depthL7","depthL9"])

# add precip states to dataframe
#raindata = pd.DataFrame(data, columns=["1", "precip"])

# data.plot()
# plt.show()

# for step in env:
#     print P1.rainfall



# self refers to the class in pyswmm (RainGage)