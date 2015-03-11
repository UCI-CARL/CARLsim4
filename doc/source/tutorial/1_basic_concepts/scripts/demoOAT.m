% OAT demo

% init OAT by adding path
initOAT

% then open a GroupMonitor on the group's
% spike file and plot the activity
GM = GroupMonitor('output','../results')
GM.plot