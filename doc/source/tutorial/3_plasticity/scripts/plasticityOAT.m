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

% Read the synaptic weights using connectionReader
CR = ConnectionReader('../results/conn_input_output.dat');

% readWeights() returns time stamps and weights.

% allTimestamps will contain the times of the individual weight snapshots
% (in ms). We took a snapshot at the beginning of the simulation and at the
% end, after 1000 seconds have elapsed, so we should have 2 values: 0 and
% 1000000.

% allWeights is of dimension numSnapshots X numSynapsesPossible.
% numSnapshots is the number of the snapshots taken while
% numSynapsesPossible is the total number of synapses possible in the
% synaptic connection. For instance, if you have 30% connectivity between
% two groups of size N1 and N2, the number of possible synapses would be
% N1 X N2.
[allTimestamps, allWeights] = CR.readWeights();

% Generate x-axis of neuron ids
x1=linspace(1, 100, 100);

% output everything to figure 2
figure(2);
hold on;

% plot the initial weights in red (first row of data).
plot(x1,allWeights(1,:),'red');

% plot the final weights in blue (second row of data).S\
plot(x1,allWeights(2,:),'blue');
xlabel('Neuron ID');
ylabel('Synaptic Weight Strength');
title('Synaptic Weight Strength vs. Neuron ID');