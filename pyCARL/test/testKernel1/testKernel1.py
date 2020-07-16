
import time
t = time.time()
from numpy import arange
from pyNN.utility import get_simulator
from pyNN.carlsim import *

# Configure the application (i.e) configure the additional

# simualator parameters
sim, options = get_simulator(("netName", "String for name of simulation"),("--gpuMode", "Enable GPU_MODE (CPU_MODE by default)", {"action":"store_true"}), ("logMode", "Enter logger mode (USER by default)", {"default":"USER"}), ("ithGPUs", "Number of GPUs"), ("randSeed", "Random seed"))

##################################################################
# Utility section (move to different utility class later)
##################################################################
# Create file scope vars for options
netName = None
simMode = None

logMode = None
ithGPUs = None
randSeed = None

# Validate and assign appropriate options
netName = options.netName

if options.gpuMode:
    simMode = sim.GPU_MODE
else:
    simMode = sim.CPU_MODE

if options.logMode == "USER":
    logMode = sim.USER
elif options.logMode == "DEVELOPER":
    logMode = sim.DEVELOPER

ithGPUs = int(options.ithGPUs)

if (simMode == sim.CPU_MODE and int(ithGPUs) > 0 ):
    print("Simulation set to CPU_MODE - overriding numGPUs to 0")

ithGPUs = 0

randSeed = int(options.randSeed)
##################################################################
# Start of application code
##################################################################

sim.setup(timestep=0.01, min_delay=1.0, netName = netName, simMode = simMode, logMode = logMode, ithGPUs = ithGPUs, randSeed = randSeed)

numNeurons = 1 

# define the neuron groups
inputCellType = sim.SpikeSourcePoisson(neuronType="EXCITATORY_NEURON", rate=50)
spike_source = sim.Population(numNeurons, inputCellType)

izhikevichCellType = sim.Izhikevich(neuronType="EXCITATORY_NEURON", a=0.02, b=0.2, c=-65, d=6)
neuron_group1 = sim.Population(numNeurons, izhikevichCellType)


# connect the neuron groups
connection = sim.Projection(spike_source, neuron_group1, sim.AllToAllConnector(), sim.StaticSynapse(weight=3.0, delay=4.0))

sim.state.network.setConductances(False)

# function has to be called before any record function is called. 
neuron_group1.record('spikes')

# run the simulation for 100ms
sim.run(100)
sim.state.network.setExternalCurrent(1, 70)
sim.run(900)

