% A simple demo of the NetworkMonitor functionality

% add Offline Analysis Toolbox to path
initOAT;

% init NetworkMonitor
NM = NetworkMonitor('../results/sim_random.dat');

% plot network activity
disp('NetworkMonitor.plot')
disp('-------------------')
disp('Press ''p'' to pause.')
disp('Press ''q'' to quit.')
disp(['Press ''s'' to enter stepping mode; then press the right arrow ' ...
	'key to step one frame forward, press the left arrow key to step ' ...
	'one frame back.'])
NM.plot;