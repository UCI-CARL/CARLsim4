import numpy
from .carlsim import *
from pyNN import common
from . import simulator
from .standardmodels.synapses import StaticSynapse, STDPMechanism

from pyNN.connectors import AllToAllConnector,OneToOneConnector,FixedProbabilityConnector        

class Connection(common.Connection):
    def __init__(self, group_id1, group_id2, connection_type, syn_wt_type, delay):
        self.group_id1 = group_id1
        self.group_id2 = group_id2
        self.connection_type = connection_type
        self.syn_wt_type = syn_wt_type
        self.delay = delay

class Projection(common.Projection):
    __doc__ = common.Projection.__doc__
    _simulator = simulator
    _static_synapse_class = StaticSynapse
    _ConnectionMonitor = None

    def __init__(self, presynaptic_population, postsynaptic_population,
                 connector, synapse_type=None, source=None, receptor_type=None,
                 space=None, label=None):
        common.Projection.__init__(self, presynaptic_population, postsynaptic_population,
                                   connector, synapse_type, source, receptor_type,
                                   space, label)
        
        nameMapping = {'OneToOneConnector': 'one-to-one', 'AllToAllConnector': "full", "FixedProbabilityConnector": "random", "FromListConnector": "FromListConnector"} #Doublecheck mapping for random
        if (connector.__class__.__name__ not in nameMapping.keys()):
            print("This connection type is currently unsupported by PyCARLsim.")
            raise NotImplemented  ##TODO implement custom exception

        #Move connection operation into sub-classes
        prob = 0.0 if not isinstance(connector, FixedProbabilityConnector) else connector.p_connect 
        plasticity = SYN_PLASTIC if isinstance(synapse_type, STDPMechanism) else SYN_FIXED  

        if isinstance(synapse_type.parameter_space['delay'], float):
            print("CARLsim does not support floating point delays. Casting to int.") #Unit conversion issues here
       
        print(synapse_type) 
        if(isinstance(synapse_type, StaticSynapse)):
            weight, delay = synapse_type.parameter_space['weight'], int(synapse_type.parameter_space['delay'])
        else:
            weight, delay = synapse_type.parameter_space['weight'].base_value, int(synapse_type.parameter_space['delay'].base_value)
        
        maxWt = weight if plasticity is SYN_FIXED else 10*weight
    
        self.connId = simulator.state.network.connect(presynaptic_population.carlsim_group, postsynaptic_population.carlsim_group, nameMapping[connector.__class__.__name__], RangeWeight(0,weight,maxWt),prob,RangeDelay(delay),RadiusRF(int(-1)), plasticity)

        if plasticity == SYN_PLASTIC: #test this!!
            simulator.state.network.setESTDP(self.post.carlsim_group, True, STANDARD, ExpCurve(synapse_type.timing_dependence.parameter_space['A_plus'].base_value, synapse_type.timing_dependence.parameter_space['tau_minus'].base_value, synapse_type.timing_dependence.parameter_space['A_minus'].base_value\
            ,synapse_type.timing_dependence.parameter_space['tau_minus'].base_value))   
        simulator.state.connections.append(self)

    def get(self, attribute_names, format, gather=True, with_address=True, multiple_synapses='sum'):
        return self._ConnectionMonitor.takeSnapshot()
    def _setConnectionMonitor(self, fName):
        self._ConnectionMonitor = simulator.state.network.setConnectionMonitor(self.pre.carlsim_group, self.post.carlsim_group, fName)
    def printWeights(self):
        self._ConnectionMonitor._print()
    def printWeightsSparse(self):
        self._ConnectionMonitor.printSparse()
