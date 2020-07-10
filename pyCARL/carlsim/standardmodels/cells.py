from copy import deepcopy
from pyNN.standardmodels import cells, build_translations
from ..import simulator
from ..carlsim import *
import numpy as np


class Izhikevich(cells.Izhikevich):
    __doc__ = cells.Izhikevich.__doc__
    def __init__(self, neuronType, a, b, c, d):

        if neuronType=='EXCITATORY_NEURON':
            self.type = EXCITATORY_NEURON
        elif neuronType=='INHIBITORY_NEURON':
            self.type = INHIBITORY_NEURON
        else: 
            print("Neuron type not supported by pyCARL")
    
        self.parameter_space = {'a': a, 'b': b, 'c': c, 'd': d}

    translations = build_translations(
        ('a',        'a'),
        ('b',        'b'),
        ('c',        'c'),
        ('d',        'd'),
        ('i_offset', 'I_e', 1000.0),
    )


class SpikeSourceArray(cells.SpikeSourceArray):
    __doc__ = cells.SpikeSourceArray.__doc__
    neuronType = -1
    def __init__(self, neuronType, spike_times):

        if neuronType=='EXCITATORY_NEURON':
            self.type = EXCITATORY_NEURON
        elif neuronType=='INHIBITORY_NEURON':
            self.type = INHIBITORY_NEURON
        else: 
            print("Neuron type not supported by pyCARL")

        for x in spike_times:
            if (not isinstance(x, int)):
                print("Spike times cannot be sub-millisecond precision")
                raise ValueError

        if isinstance(spike_times, np.ndarray):
            self.spike_times = spike_times.tolist()
        else:
            self.spike_times = spike_times

    

class SpikeSourcePoisson(cells.SpikeSourcePoisson):
    pars = cells.SpikeSourcePoisson.default_parameters
    
    def __init__(self, neuronType, rate = pars['rate']):
        self.rate = rate
        if neuronType=='EXCITATORY_NEURON':
            self.type = EXCITATORY_NEURON
        elif neuronType=='INHIBITORY_NEURON':
            self.type = INHIBITORY_NEURON
        else: 
            print("Neuron type not supported by pyCARL")


        #if (self.pars['duration'] != duration or self.pars['start'] != start):
        #print ("CARLsim does not support setting duration or start time for poisson objects. These parameters will be ignored")

