FrameDur=3300;
nrX=32;
s = readSpikes('../../results/orientation/spkV4o.dat',FrameDur);

if ~isequal(size(s),[4,nrX*nrX*4]), s(4,nrX*nrX*4) = 0; end

figure, imagesc(reshape(permute(reshape(s([2 1 4 3],:),[4 nrX nrX 4]),[2 1 3 4]),[4*nrX 4*nrX])/3300*1000);
set(gca,'XTick',nrX/2:nrX:4*nrX);
set(gca,'XTickLabel',{'RD','H','LD','V'});
set(gca,'YTickLabel','');
colorbar;