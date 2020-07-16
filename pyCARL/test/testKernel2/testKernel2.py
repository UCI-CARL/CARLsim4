from numpy import arange
from pyNN.utility import get_simulator
from pyNN.carlsim import *
import time
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

nExc = 800
nInh = 200

# connection propability
pConn = 100.0/ (nExc+nInh)

##################################################################
# Define the neuron groups.
##################################################################

# create a spike generator group.
inputCellType = sim.SpikeSourceArray(neuronType="EXCITATORY_NEURON", spike_times=[5, 10, 15, 20])
spike_source = sim.Population(nExc, inputCellType, label='input')

# create neuron groups. 
izhikevichCellType1 = sim.Izhikevich("EXCITATORY_NEURON", a=0.02, b=0.2, c=-65, d=8)
neuron_group1 = sim.Population(nExc, izhikevichCellType1, label='exc')

# create neuron groups. 
izhikevichCellType2 = sim.Izhikevich("INHIBITORY_NEURON", a=0.1, b=0.2, c=-65, d=2)
neuron_group2 = sim.Population(nInh, izhikevichCellType1, label='inh')

##################################################################
# Define the connections.
##################################################################

# connect input to exc - "one-to-one"
connection = sim.Projection(spike_source, neuron_group1, sim.OneToOneConnector(), sim.StaticSynapse(weight=3.0, delay=4.0))

# connect input to exc - "random connector"
connection = sim.Projection(neuron_group1, neuron_group1, sim.FixedProbabilityConnector(pConn), sim.StaticSynapse(weight=3.0, delay=4.0))

# connect input to exc - "random connector"
connection = sim.Projection(neuron_group1, neuron_group2, sim.FixedProbabilityConnector(pConn), sim.StaticSynapse(weight=3.0, delay=4.0))

# connect input to exc - "random connector"
connection = sim.Projection(neuron_group2, neuron_group1, sim.FixedProbabilityConnector(pConn), sim.StaticSynapse(weight=3.0, delay=4.0))

sim.state.network.setConductances(False)

##################################################################
# Setup Network (function native to CARLsim) 
# Record functions can only be called after the setupNetwork 
# function is called.
##################################################################

# function has to be called before any record function is called. 
neuron_group1.record('spikes')

# run the simulation for 100ms
sim.run(1000)

# start the recording of the groups

