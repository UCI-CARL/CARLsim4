clear;
ME=MotionEnergySim('MTfixedHalfRect/MTpatternTest');

xDirection=(0:11)*2*pi/12;          % all data points
nrF=50;                             % # of frames per data point
normHz = ME.simFrameDur/1000*nrF;   % denominator for normalizing to Hz

%     497;500;505
nid=466 ; % 464. 528
% plot V1 grating and plaid
% we're plotting the first data point twice to close the loop in the tuning
% curve (for aesthetics only)
V1=ME.getFiringRates('V1ME',ME.simFrameDur*nrF)/normHz;
CDS=ME.getFiringRates('MT1CDS',ME.simFrameDur*nrF)/normHz;
CDS(max(1,end),8*32*32)=0;
PDS=ME.getFiringRates('MT1PDS',ME.simFrameDur*nrF)/normHz;
PDS(max(1,end),8*32*32)=0;

subplot(242)
V1tmp = reshape(V1(1:20,16*1024+1:17*1024),20,32,32);
V1tmp = V1tmp(:,5:27,5:27); % ignore border

V1mu = mean(reshape(V1tmp,20,[]),2);
V1std = std(reshape(V1tmp,20,[]),0,2);

polar([0 1e-3],[60 60]);hold on;
polar([xDirection xDirection(1)], [V1mu;V1mu(1)]', 'k.-');
polar([xDirection xDirection(1)], [V1mu+V1std;V1mu(1)+V1std(1)]', 'g.-');
polar([xDirection xDirection(1)], [V1mu-V1std;V1mu(1)-V1std(1)]', 'r.-');
hold off;
freezeColors;

subplot(246);
V1tmp = reshape(V1(21:40,16*1024+1:17*1024),20,32,32);
V1tmp = V1tmp(:,5:27,5:27); % ignore border

V1mu = mean(reshape(V1tmp,20,[]),2);
V1std = std(reshape(V1tmp,20,[]),0,2);

polar([0 1e-3],[60 60]);hold on;
polar([xDirection xDirection(1)], [V1mu;V1mu(1)]', 'k.-');
polar([xDirection xDirection(1)], [V1mu+V1std;V1mu(1)+V1std(1)]', 'g.-');
polar([xDirection xDirection(1)], [V1mu-V1std;V1mu(1)-V1std(1)]', 'r.-');
hold off;
freezeColors;


% plot MT CDS grating and plaid
subplot(243)
CDStmp = reshape(CDS(1:20,2*1024+1:3*1024),20,32,32);
CDStmp = CDStmp(:,5:27,5:27); % ignore border

CDSmu = mean(reshape(CDStmp,20,[]),2);
CDSstd = std(reshape(CDStmp,20,[]),0,2);

polar([0 1e-3],[60 60]);hold on;
polar([xDirection xDirection(1)], [CDSmu;CDSmu(1)]', 'k.-');
polar([xDirection xDirection(1)], [CDSmu+CDSstd;CDSmu(1)+CDSstd(1)]', 'g.-');
polar([xDirection xDirection(1)], [CDSmu-CDSstd;CDSmu(1)-CDSstd(1)]', 'r.-');
hold off;
freezeColors;

subplot(247)
CDStmp = reshape(CDS(21:40,2*1024+1:3*1024),20,32,32);
CDStmp = CDStmp(:,5:27,5:27); % ignore border

CDSmu = mean(reshape(CDStmp,20,[]),2);
CDSstd = std(reshape(CDStmp,20,[]),0,2);

polar([0 1e-3],[60 60]);hold on;
polar([xDirection xDirection(1)], [CDSmu;CDSmu(1)]', 'k.-');
polar([xDirection xDirection(1)], [CDSmu+CDSstd;CDSmu(1)+CDSstd(1)]', 'g.-');
polar([xDirection xDirection(1)], [CDSmu-CDSstd;CDSmu(1)-CDSstd(1)]', 'r.-');
hold off;
freezeColors;


% plot MT PDS grating and plaid
subplot(244)
PDStmp = reshape(PDS(1:20,2*1024+1:3*1024),20,32,32);
PDStmp = PDStmp(:,5:27,5:27); % ignore border

PDSmu = mean(reshape(PDStmp,20,[]),2);
PDSstd = std(reshape(PDStmp,20,[]),0,2);

polar([0 1e-3],[60 60]);hold on;
polar([xDirection xDirection(1)], [PDSmu;PDSmu(1)]', 'k.-');
polar([xDirection xDirection(1)], [PDSmu+PDSstd;PDSmu(1)+PDSstd(1)]', 'g.-');
polar([xDirection xDirection(1)], [PDSmu-PDSstd;PDSmu(1)-PDSstd(1)]', 'r.-');
hold off;
freezeColors;

subplot(248)
PDStmp = reshape(PDS(21:40,2*1024+1:3*1024),20,32,32);
PDStmp = PDStmp(:,5:27,5:27); % ignore border

PDSmu = mean(reshape(PDStmp,20,[]),2);
PDSstd = std(reshape(PDStmp,20,[]),0,2);

polar([0 1e-3],[60 60]);hold on;
polar([xDirection xDirection(1)], [PDSmu;PDSmu(1)]', 'k.-');
polar([xDirection xDirection(1)], [PDSmu+PDSstd;PDSmu(1)+PDSstd(1)]', 'g.-');
polar([xDirection xDirection(1)], [PDSmu-PDSstd;PDSmu(1)-PDSstd(1)]', 'r.-');
hold off;
freezeColors;

