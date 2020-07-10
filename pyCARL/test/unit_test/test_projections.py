import unittest
import sys
import logging
from importlib import import_module



class TestProjections(unittest.TestCase):
    def setUp(self):
        self.sim = import_module("pyNN.carlsim")
        self.sim.setup(timestep=0.01, min_delay=1.0, netName = "test_projections", simMode = 0, logMode = 3, ithGPUs = 0, randSeed = 42)
        self.size = 2
        self.sizeN = self.size
        

    def test_AlltoAllConnector(self):
        log = logging.getLogger("All-to-All")
        log.debug("Testing All-to-All Connector")
        cellType = self.sim.Izhikevich(neuronType="EXCITATORY_NEURON",a=0.02, b=0.2, c=-65, d=8)
        g0 = self.sim.Population(self.size, cellType)
        g1 = self.sim.Population(self.size, cellType)

        syn = self.sim.StaticSynapse(weight=1, delay=1)
        c0 = self.sim.Projection(g0, g1, self.sim.AllToAllConnector(), synapse_type = syn)

        self.sim.state.setupNetwork()
        
        self.assertEqual(self.sim.state.network.getNumSynapticConnections(c0.connId), self.sizeN*self.sizeN)
        log.debug("All-to-All Connector test: Success")
    


class TestPopulations(unittest.TestCase):
    def setUp(self):
        self.sim = import_module("pyNN.carlsim")
        self.sim.setup(timestep=0.01, min_delay=1.0, netName = "test_populations", simMode = 0, logMode = 3, ithGPUs = 0, randSeed = 42)

    #assign silly values and expect the program to fail

    def test_createSpikeGeneretorDeath(self):
        log = logging.getLogger("Spike Source")
        log.debug("Testing Spike Source Population")

        with self.assertRaises(Exception, msg="SS1: Negative 1 was supposed to raise an exception"):
            nNeur = -1
            spikeSource = self.sim.SpikeSourceArray(neuronType='EXCITATORY_NEURON', spike_times=[1,2,3,4,5,6,7,8])
            self.sim.Population(nNeur, spikeSource)

        with self.assertRaises(Exception, msg="SS2: The given x value was supposed to raise an expception"):
            nNeur = (-1,1,1)
            spikeSource = self.sim.SpikeSourceArray(neuronType='EXCITATORY_NEURON', spike_times=[1,2,3,4,5,6,7,8])
            self.sim.Population(nNeur, spikeSource)
        with self.assertRaises(Exception, msg="SS3: The given y value was supposed to raise an exception"):
            nNeur = (1,-1,1)
            spikeSource = self.sim.SpikeSourceArray(neuronType='EXCITATORY_NEURON', spike_times=[1,2,3,4,5,6,7,8])
            self.sim.Population(nNeur, spikeSource)
        with self.assertRaises(Exception, msg="SS4: The given z value was supposed to raise an exception"):
            nNeur = (1,1,-1)
            spikeSource = self.sim.SpikeSourceArray(neuronType='EXCITATORY_NEURON', spike_times=[1,2,3,4,5,6,7,8])
            self.sim.Population(nNeur, spikeSource)


        log.debug("Spike Source Population test: Success")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    #unittest.TextTestRunner().run(TestPopulations())
    unittest.main()
