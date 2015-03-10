% Tutorial 3 Plasticity OAT Scripts

% First run initOAT script to set the correct path
initOAT;

% Read the spike files
SR = SpikeReader('../results/spk_output.dat');
% Bin the data into 1s (1000 ms) intervals
spkData = SR.readSpikes(1000);
% Generate time data
time=linspace(1,1000,1000);
% generate target firing rate data
targetFR(1:1000)=35;
figure(1);
hold on;
% plot the average output neuron firing rate in blue
plot(time,spkData,'blue');
% plot the target firing rate in red
plot(time,targetFR,'red');
% make labels and title
xlabel('Time (sec)');
ylabel('Average Firing Rate (Hz)');
title('Average Firing Rate of Output Neuron vs. Time');

% Read the 
CR = ConnectionReader('../results/conn_input_output.dat');
[allTimestamps, allWeights] = CR.readWeights();
x1=linspace(1, 100, 100);
figure(2);
hold on;
plot(x1,allWeights(1,:),'red');
plot(x1,allWeights(2,:),'blue');
xlabel('Neuron ID');
ylabel('Synaptic Weight Strength');
title('Synaptic Weight Strength vs. Neuron ID');