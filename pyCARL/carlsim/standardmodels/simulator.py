from .carlsim import *
import logging
from pyNN import common
from pyNN.common.control import DEFAULT_MAX_DELAY, DEFAULT_TIMESTEP, DEFAULT_MIN_DELAY

name = "carlsim"
logger = logging.getLogger("PyNN")

class ID(int, common.IDMixin):

    def __init__(self, n):
        """Create an ID object with numerical value `n`."""
        int.__init__(n)
        common.IDMixin.__init__(self)

class State(common.control.BaseState):
    groupIDs = []
    connections = []
    poissonObjects = []
    rateObjects = [] #Keep a reference to all created SpikeGen objects so they don't go out of scope mid-run
    def __init__(self):
        self.initsuccess = False
        common.control.BaseState.__init__(self)
        self.network        = None
        self.netName        = None
        self.simMode        = None
        self.logMode        = None
        self.ithGPUs        = None
        self.randSeed       = None
        self.id_counter         = 0
        self.num_processes      = 1
        self.spikeMonitors = []
        self.connMonitors = []
        self.recordingGroups = []
        self._isSetup = False
        self.clear()

    def set_params_and_init(self, extra_params):
        
        for key in extra_params:
            if key == "netName":
                self.netName = extra_params[key]
            elif key == "simMode":
                self.simMode = extra_params[key]
            elif key == "logMode":
                self.logMode = extra_params[key]
            elif key == "ithGPUs":
                self.ithGPUs = extra_params[key]
            elif key == "randSeed":
                self.randSeed = extra_params[key]
        
        self.network = CARLsim(self.netName, self.simMode, 
                                self.logMode, self.ithGPUs,
                                self.randSeed)
        
        self.initsuccess = True

    def clear(self):
        self.id_counter = 0

    def reset(self):
        pass

    def run(self, simtime):
            #simtime in PyNN is provided in milliseconds
        if (not self._isSetup):
            self.setupNetwork()
            self._isSetup = True

        self._startRecordingSpikes()
        nSec = simtime/1000
        nMsec = simtime%1000
        self.network.runNetwork(int(nSec), int(nMsec))
     
    def run_until(self, tstop):
        # get the present sim time compute the time to run until
        if (not self._isSetup):
            self.setupNetwork()
            self._isSetup = True

        self._startRecordingSpikes()
        time = int(self.network.getSimTimeMsec())
        runtime=0
        if tstop > time:
            runtime = tstop - time
        nSec = runtime/1000
        nMsec = runtime%1000
        self.network.runNetwork(int(nSec), int(nMsec))
        
    def setupNetwork(self): 
        self.network.setupNetwork()
        self._setupSpikeMonitors()
        self._setupConnMonitors()
        self._setupPoissonObjects()

    def clear(self):
        pass
    
    def setHomeostasis(self, neuronGroupId, enableHomeostasis, alpha, T):
        self.network.setHomeostasis(neuronGroupId, enableHomeostasis, alpha, T)

    def setHomeoBaseFiringRate(self, neuronGroupId, R_target, std):
        self.network.setHomeoBaseFiringRate(neuronGroupId, R_target, std)           

    # cannot be implemented with CARLsim
    def reset(self):
        pass    

    def _get_dt(self):
        pass    

    def _set_dt(self, timestep):
        pass    

    def _get_min_delay(self):
        pass
    
    def _set_min_delay(self, delay):
        pass 
    
    def _setupSpikeMonitors(self):
        for gid in self.groupIDs:
            mon = self.network.setSpikeMonitor(gid, "NULL")
            self.spikeMonitors.append(mon)
    
    def _setupConnMonitors(self):
        for conn in self.connections:
            conn._setConnectionMonitor("NULL")
    
    def _startRecordingSpikes(self):
        for mon in self.spikeMonitors:
            if (mon.isRecording()):
                mon.startRecording()
    
    def _setupPoissonObjects(self):
        for tup in self.poissonObjects:
            (gid, rate, size) = tup #see populations.py for definition of this tuple
            P = PoissonRate(int(size), bool(0))
            P.setRates(rate)
            self.network.setSpikeRate(gid, P) 
            self.rateObjects.append(P) #This is needed so the PoissonRate objects don't get garbage collected.

############################################################################
# Params to suppress recording errors - not used otherwise
############################################################################
    dt = DEFAULT_TIMESTEP
    @property
    def t(self):
        return 0
############################################################################
state = State()
