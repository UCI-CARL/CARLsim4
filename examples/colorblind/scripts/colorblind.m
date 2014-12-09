% Version 12/8/14
addpath ../../../tools/offline_analysis_toolbox

list={'R','G','B','C','M','Y'};
nrX=64;
fid=fopen('../videos/colorblind.dat','r');
vid=fread(fid,[3*nrX*nrX inf],'uint8');
fclose(fid);

inputs = reshape(permute(reshape(vid(:,1:6)/255,3,nrX,nrX,6),[3 2 4 1 ]),[nrX,nrX*6,3]);

frameDurMs=100;

resp = zeros(0,nrX*6);
for j=1:length(list)
    SR = SpikeReader(['../results/spkV4' list{j} '.dat']);
	spkData = SR.readSpikes(frameDurMs);    
    resp = [resp; ...
		reshape(permute(reshape(spkData(1:6,:),6,nrX,nrX),[3 2 1]), ...
		[nrX nrX*6])];
	clear SR;
end
figure, imagesc([inputs; repmat(resp/max(resp(:)), [1 1 3])]);
axis image off;
