import numpy
from pyNN import recording
from pyNN.carlsim import simulator

class Recorder(recording.Recorder): 
    _simulator = simulator    
   
    def __init__(self, population, file=None):
        super(Recorder, self).__init__(population, file=file)
        self.event_output_files = []    
        self.displays = []
        self.output_files = []
        self.spikeMonitor = None

    def _record(self, variable, to_file = None, sampling_interval=None):
        if variable == 'spikes':
            self._simulator.state.recordingGroups.append(self.population.carlsim_group)
            #print(self._simulator.state.spikeIds)
    def start_recording(self):
        self.spikeMonitor.startRecording()
    def stop_recording(self):
        self.spikeMonitor.stopRecording()
