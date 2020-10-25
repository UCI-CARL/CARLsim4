% A simple demo of the NetworkMonitor functionality

% add Offline Analysis Toolbox to path
initOAT;

% init NetworkMonitor
NM = NetworkMonitor('../results/sim_hello world.dat');

% plot network activity
disp('NetworkMonitor.plot')
disp('-------------------')
disp('Press ''p'' to pause.')
disp('Press ''q'' to quit.')
disp(['Press ''s'' to enter stepping mode; then press the ' ...
	'right arrow key to step'])
disp('one frame forward, press the left arrow key to step one frame back.')
disp('Press ''s'' again to leave stepping mode.')
disp(' ')
NM.plot;


% init ConnectionMonitor
CM = ConnectionMonitor('input','output','../results/');

% visualize receptive fields (little 2D Gaussian kernels for each post-neuron)
disp('ConnectionMonitor.plot(''receptivefield'')')
disp('-------------------')
disp('Press ''p'' to pause.')
disp('Press ''q'' to quit.')
disp(['Press ''s'' to enter stepping mode; then press the ' ...
	'right arrow key to step'])
disp('one frame forward, press the left arrow key to step one frame back.')
disp('Press ''s'' again to leave stepping mode.')
CM.plot('receptivefield')

% analogously, the response field of each pre-neuron can be visualized:
% CM.plot('responsefield')