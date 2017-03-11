% Initialize the OAT: add paths etc.
initOAT

% Point a NetworkMonitor to the simulation file
NM = NetworkMonitor('../results/sim_me.dat')

% First, let's have a look at the population activity as a heat map.
% Set the plot type for all groups, and plot in 500 ms increments
% (every stimulus direction has 10 frames of 50 ms duration each).
NM.setGroupPlotType(-1, 'heatmap')
NM.plot(-1, 500)

% Second, let's have a look at the population activity as a flow
% field. Since there are eight neurons coding for eight different
% directions of motion at each pixel location, we can use population
% vector decoding to find the net vector of motion at every pixel
% location.
NM.setGroupPlotType(-1, 'flowfield')
NM.plot(-1, 500)