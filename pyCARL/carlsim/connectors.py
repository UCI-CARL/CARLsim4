"""
Connection method classes for the neuron module

:copyright: Copyright 2006-2016 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""
import .carlsim

from pyNN.neuron import simulator
from pyNN.connectors import AllToAllConnector, \
                            OneToOneConnector, \
                            FixedProbabilityConnector, \
                            DistanceDependentProbabilityConnector, \
                            DisplacementDependentProbabilityConnector, \
                            IndexBasedProbabilityConnector, \
                            FromListConnector, \
                            FromFileConnector, \
                            FixedNumberPreConnector, \
                            FixedNumberPostConnector, \
                            SmallWorldConnector, \
                            CSAConnector, \
                            CloneConnector, \
                            ArrayConnector, \
                            FixedTotalNumberConnector



class AllToAllConnector(AllToAllConnector):

    def connect(self, projection, plasticity = SYN_STATIC, ):
	simulator.state.network.connect(presynaptic_population.carlsim_group, postsynaptic_population.carlsim_group, "full")	
