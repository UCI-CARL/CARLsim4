function [] = plotGratingPlaid()
% [] = plotGratingPlaid()
%
% PLOTGRATINGPLAID generates polar plots of MT direction tuning for
% a sinusoidal grating and a plaid stimulus drifting upwards, where the
% angle in the polar plot denotes motion direction and the radius is the
% firing rate in spikes per second.
% This script reproduces Fig. 4 of Beyeler, Richert, Dutt, and Krichmar
% (2014).
% A suitable stimulus can be produces using scripts/v1MTLIP/makeBarSpeed.m
%
% Author: Michael Beyeler <mbeyeler@uci.edu>
% Ver 2/5/14

nPnt = 24; % number of stim directions (data points)
nrF=50; % number of frames per stim direction
frameDur=50; % number of ms each frame is presented

xDirection = (0:nPnt-1)*2*pi/nPnt; % stim direction

V1 = readSpikes('../../results/v1MTLIP/spkV1ME.dat',frameDur*nrF);
CDS = readSpikes('../../results/v1MTLIP/spkMT1CDS.dat',frameDur*nrF);
PDS = readSpikes('../../results/v1MTLIP/spkMT1PDS.dat',frameDur*nrF);

% convert to Hz
toHz = frameDur*nrF/1000;

data{1} = V1(:,16*1024+1:17*1024)/toHz;
data{2} = CDS(:,2*1024+1:3*1024)/toHz;
data{3} = PDS(:,2*1024+1:3*1024)/toHz;

for d=1:3
    for i=1:2
        subplot(2,3,(i-1)*3+d);
        tmpData = data{d};
        tmpData = reshape(tmpData((i-1)*nPnt+1:i*nPnt,:),nPnt,32,32);
        tmpData = tmpData(:,5:27,5:27); % ignore border
        
        muData = mean(reshape(tmpData,nPnt,[]),2);
        stdData = std(reshape(tmpData,nPnt,[]),0,2);
        
        maxR = 120;
        if d~=1 || i~=1, maxR=60; end
        polar([0 1e-3],[maxR maxR]);
        hold on;
%         polar([xDirection xDirection(1)], [predicted;predicted(1)]', 'k-.');
        polar([xDirection xDirection(1)], [muData;muData(1)]', 'k.-');
        polar([xDirection xDirection(1)], [muData+stdData;muData(1)+stdData(1)]', 'g.-');
        polar([xDirection xDirection(1)], [muData-stdData;muData(1)-stdData(1)]', 'r.-');
        hold off;
    end
end

end