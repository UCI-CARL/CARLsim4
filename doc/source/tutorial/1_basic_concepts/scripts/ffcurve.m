% initialize OAT
initOAT

% open SpikeReader object on spike file
SR = SpikeReader('../results/spk_output.dat');

% read # spikes binned into 1000ms bins
spk = SR.readSpikes(1000);

% plot input-output (FF) curve
plot(10:10:100, spk, 'bo-', 'LineWidth',2)
xlabel('input rate (Hz)')
ylabel('output rate (Hz)')