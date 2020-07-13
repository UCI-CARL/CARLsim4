import unittest
import pyNN.carlsim as sim
import sys
import logging

class TestStaticSynapse(unittest.TestCase):
    
    def setUp(self):
        sim.setup(timestep = 0.01, min_delay = 1, netName = "StaticSynapseTest", simMode = 0, logMode = 3, ithGPUs = 0, randSeed = 42)
        self.spikeType = sim.SpikeSourcePoisson(neuronType='EXCITATORY_NEURON', rate = 40)
        self.inputGroup = sim.Population(5, self.spikeType)
        self.outputGroup = sim.Population(3, sim.Izhikevich(a=0.02, b=0.2, c=-65, d=6, neuronType = "EXCITATORY_NEURON"))
        self.log = logging.getLogger("StaticSynapseLog")
        self.log.debug("Finished setup")

    def test_StaticSynapse(self):

        self.log.debug("Starting StaticSynapse with AlltoAllConnector test")

        #with self.assertRaises(Exception, msg = "Weight can't be negative"):
        #    connection = sim.Projection(self.inputGroup, self.outputGroup, sim.AllToAllConnector(), sim.StaticSynapse(weight = -3.0, delay = 4))

        #with self.assertRaises(Exception, msg = "Delay can't be less than one"):
        #    connection = sim.Projection(self.inputGroup, self.outputGroup, sim.AllToAllConnector(), sim.StaticSynapse(weight = 3.0, delay = -4))

        self.log.debug("Finished raising exceptions")

        connection = sim.Projection(self.inputGroup, self.outputGroup, sim.AllToAllConnector(), sim.StaticSynapse(weight = 3.0, delay = 4))

        sim.state.setupNetwork()

        weights = connection.get("weights",format = "list")
        
        for row in weights:
            for weight in row:
                self.assertEqual(weight, 3.0)
        
        #TODO add delay checking here
        self.log.debug("StaticSynapse with AlltoAllConnector test passed!")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    unittest.main()    
