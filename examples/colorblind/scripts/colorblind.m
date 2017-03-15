list={'R','G','B','C','M','Y'};
nrX=64;
fid=fopen('../../videos/colorblind.dat','r');
vid=fread(fid,[3*nrX*nrX inf],'uint8');
fclose(fid);
inputs = reshape(permute(reshape(vid(:,1:6)/255,3,nrX,nrX,6),[3 2 4 1 ]),[nrX,nrX*6,3]);

FrameDur=100;

resp = zeros(0,nrX*6);

for j=1:length(list)
    s = readSpikes(['../../results/colorblind/spkV4' list{j} '.dat'],FrameDur);
    
    resp = [resp; reshape(permute(reshape(s(1:6,:),6,nrX,nrX),[3 2 1]),[nrX nrX*6])];
end
figure, imagesc([inputs; repmat(resp/max(resp(:)), [1 1 3])]);
axis image off;
