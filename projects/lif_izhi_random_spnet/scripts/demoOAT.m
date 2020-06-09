% OAT demo

% first, init OAT by adding path
initOAT

% second, open a NetworkMonitor on the simulation file
% and plot the activity of the network
NM = NetworkMonitor('../results/sim_spnet.dat')
NM.plot

% third, observe weight changes in the network
CM0 = ConnectionMonitor('exc','exc','../results')
CM0.plot('histogram')

CM1 = ConnectionMonitor('inh','exc','../results')
CM1.plot('histogram')