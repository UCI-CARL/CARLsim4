import unittest
import sys
import logging
from importlib import import_module

# unittest is used for writing the test application
# sys and logging modules are for logging test info onto the terminal

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
        
        
    
    #set silly values
    #for now unable to write tests for neuron type
    def test_createGroup(self):
        log = logging.getLogger("TestLog")
        log.debug("Testing Izhikevich population")
        with self.assertRaises(Exception, msg= "IZ1: The negative value was supposed to raise an exception"):
            self.sim.Population(-10, self.sim.Izhikevich("EXCITATORY_NEURON",a=0.02, b=0.2, c=-65, d=6))
        
        with self.assertRaises(Exception, msg="IZ2: The negative x value was supposed to raise an exception"):
            self.sim.Population((-10,1,1), self.sim.Izhikevich("EXCITATORY_NEURON",a=0.02, b=0.2, c=-65, d=6))

        with self.assertRaises(Exception, msg="IZ3: The negative y value was supposed to raise an exception"):
            self.sim.Population((10,-1,1), self.sim.Izhikevich("EXCITATORY_NEURON",a=0.02, b=0.2, c=-65, d=6))

        with self.assertRaises(Exception, msg="IZ4: The negative z value was supposed to raise an exception"):
            self.sim.Population((10,1,-1), self.sim.Izhikevich("EXCITATORY_NEURON",a=0.02, b=0.2, c=-65, d=6))

        with self.assertRaises(Exception, msg="IZ5: The a paramater cannot be a string"):
            self.sim.Population(10, self.sim.Izhikevich("EXCITATORY_NEURON",a="a", b=0.2, c=-65, d=6))

        with self.assertRaises(Exception, msg="IZ6: The b parameter cannot be a string"):
            self.sim.Population(10, self.sim.Izhikevich("EXCITATORY_NEURON",a=0.02, b="a", c=-65, d=6))

        with self.assertRaises(Exception, msg="IZ7: The c parameter cannot be a string"):
            self.sim.Population(10, self.sim.Izhikevich("EXCITATORY_NEURON",a=0.02, b=0.2, c="a", d=6))

        with self.assertRaises(Exception, msg="IZ8: The d parameter cannot be a string"):
            self.sim.Population(10, self.sim.Izhikevich("EXCITATORY_NEURON",a=0.02, b=0.2, c=-65, d="a"))
    
        log.debug("Izhikevich Population tests: Success")

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    #unittest.TextTestRunner().run(TestPopulations())
    unittest.main()
