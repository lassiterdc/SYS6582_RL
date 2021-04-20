from tensorforce.environments import Environment
from tensorforce.agents import Agent
from pyswmm import Simulation, Nodes, Links
import pyswmm

# nodes module: https://pyswmm.readthedocs.io/en/stable/reference/nodes.html
# links module: https://pyswmm.readthedocs.io/en/stable/reference/links.html
#%%
with Simulation('theta.inp') as sim:
    sim.step_advance(300) # every 5 minutes
    nodes = Nodes(sim)
    links = Links(sim)
    
    p1 = nodes['P1']
    p2 = nodes['P2']
    l8 = links['8']
    
    
    
    for step in sim:
        stats_p1 = p1.statistics
        stats_p2 = p2.statistics
        stats_l8 = l8.conduit_statistics
        pass
#%%
        
pf_l8 = stats_l8['peak_flow']
fld_p1 = stats_p1['flooding_volume']
fld_p2 = stats_p2['flooding_volume']


         

