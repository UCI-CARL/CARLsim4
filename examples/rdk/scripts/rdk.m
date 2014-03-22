ss = zeros(8,10,8);

s = readSpikes('../../results/rdk/spkPFC.dat',3300);
if (size(s,1) < 80) s(80,8) = 0; end

ss = squeeze(mean(reshape(s,8,10,[],8),3));
figure, plot(100:-10:10,diff(squeeze(ss(3,:,[5 1]+2)),[],2)*1000); xlabel('Coherence'); ylabel('PFC right - left (Hz)'); title('Motion Selectivity in PFC');
