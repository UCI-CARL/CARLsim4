% OAT demo

% first, init OAT by adding path
initOAT

% second, open a NetworkMonitor on the simulation file
% and plot the activity of the network
NM = NetworkMonitor('../results/sim_spnet.dat');
NM.plot

nM = NeuronMonitor('exc','../results/');
nM.plot
