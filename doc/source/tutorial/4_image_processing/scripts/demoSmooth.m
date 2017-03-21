% OAT demo

%% INITIALIZATION

% init Offline Analysis Toolbox and Visual Stimulus Toolbox
initOAT


%% CREATING AN IMAGE FILE

% Create a raw stimulus file from an image file using the Visual Stimulus
% Toolbox. CARLsim is able to read raw stimulus files using the Visual
% Stimulus bindings.
createStimFromImage('../input/carl.jpg', '../input/carl.dat', ...
    [256 256], 'gray')

% Run the CARLsim 'smooth' network using the .dat file created above...
% $ make smooth
% ./smooth


%% PLOTTING ACTIVITY

% Plot network activity
clear NM
NM = NetworkMonitor('../results/sim_smooth.dat')
NM.setGroupPlotType(-1, 'heatmap')
NM.plot