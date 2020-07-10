import numpy
from pyNN import common
from pyNN.standardmodels import StandardCellType, cells
from pyNN.parameters import ParameterSpace, simplify
from . import simulator
from .recording import Recorder
from .carlsim import *

#synapse_type = ''

class Assembly(common.Assembly):
    __doc__ = common.Assembly.__doc__
    _simulator = simulator

class Population(common.Population):
    __doc__ = common.Population.__doc__
    _simulator = simulator
    _recorder_class = Recorder
    _assembly_class = Assembly

    def _create_cells(self):
        id_range = numpy.arange(simulator.state.id_counter,
                                simulator.state.id_counter + self.size)
        self.all_cells = numpy.array([simulator.ID(id) for id in id_range],
                                     dtype=simulator.ID)
        self._mask_local = numpy.ones((self.size,), bool)

        for id in self.all_cells:
            id.parent = self
        simulator.state.id_counter += self.size
        
        if (isinstance(self.size, int)):
            shape = self.size
        elif (len(self.size) == 2):
            shape = Grid3D(self.size[0],self.size[1], 1)
        elif (len(self.size) == 3):
            shape = Grid3D(self.size[0],self.size[1],self.size[2])
        else:
            print("really? How do you expect me to build a neural network in more than three dimensions?")
            raise ValueError

        if isinstance(self.celltype, cells.SpikeSourceArray):
            self.carlsim_group = simulator.state.network.createSpikeGeneratorGroup(str(self.label), shape, self.celltype.type) 
            spikeGen = SpikeGeneratorFromVector(self.celltype.spike_times)
            simulator.state.network.setSpikeGenerator(self.carlsim_group, spikeGen)
            simulator.state.rateObjects.append(spikeGen) #Need this so spikeGen doesn't go out of scope. Need a better solution
             
        if isinstance(self.celltype, cells.Izhikevich):
            self.carlsim_group = simulator.state.network.createGroup(str(self.label), shape, self.celltype.type)
            parameters = self.celltype.parameter_space
            simulator.state.network.setNeuronParameters(self.carlsim_group, parameters['a'], parameters['b'],
                                                            parameters['c'], parameters['d'])
        if isinstance(self.celltype, cells.SpikeSourcePoisson):
            self.carlsim_group = simulator.state.network.createSpikeGeneratorGroup(str(self.label), shape, self.celltype.type) 
            self._simulator.state.poissonObjects.append((self.carlsim_group, self.celltype.rate, numpy.prod(shape)))
            self._simulator.state.groupIDs.append(self.carlsim_group)

    def _set_initial_value_array(self, variable, initial_value):
        """
        Empty method to suppress setting initial value
        Carlsim does not need initial value setting (handled internally)
        :param variable:
        :param initial_value:
        :return:
        """
        pass

    def _get_view(self, selector, label=None):
        pass

    def _get_parameters(self, parameter_space):
        pass
    
    def _set_parameters(self, parameter_space):
        pass

class PopulationView(common.PopulationView):
    _assembly_class = Assembly
    _simulator = simulator
