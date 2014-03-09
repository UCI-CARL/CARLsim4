function [] = plotGratingContrast()
% [] = plotGratingContrast()
%
% PLOTGRATINGCONTRAST produces the contrast sensitivity function of model
% V1 simple cells (blue), plotted against electrophysiological data from
% (Movshon & Newsome, 1996) (red). Each data point is a V1 mean response to
% a drifting grating, averaged over both one second of stimulus
% presentation and all neurons in the subpopulation. Vertical bars are the
% standard deviation on the population average.
% This script reproduces Fig. 3 of Beyeler, Richert, Dutt, and Krichmar
% (2014).
% A suitable stimulus can be created using
% scripts/v1MTLIP/makeGratingContrast.m. However, in order to analyze V1
% simple cell responses, one needs to comment out the V1 complex cell code
% in v1ColorME.cu.
%
% Author: Michael Beyeler <mbeyeler@uci.edu>
% Ver: 2/5/14

xDirection = 0; % drift direction
nrF = 50; % number of frames per contrast
frameDur = 50; % frame duration (ms)
normHz = frameDur/1000*nrF; % norm to Hz
nDataPoints=20; % number of data points

% create IDs of neurons far from image border
nid=[];
for i=10:20
for j=10:20
nid=[nid sub2ind([32 32],i,j)];
end
end

stimContrast = linspace(log2(0.02), log2(1), nDataPoints);
stimContrast = 2.^stimContrast;

spk = readSpikes('../../results/v1MTLIP/spkV1ME.dat',frameDur)/normHz;
spk(end,28*3*1024)=0;
spk=reshape(spk,size(spk,1),32*32,28,3);

figure;
for i=25 % choose some V1 subpopulation that gives strong responses
    plot([0.015 0.02 0.03 0.06 0.08 0.1:0.1:0.5],[3.5 7 11 32 55 65 90 92 91 100],'r','LineWidth',2);
    hold on;
    errorbar(stimContrast,mean(squeeze(spk(:,nid,i,1)),2)',std(squeeze(spk(:,nid,i,1)),0,2)');
    hold off;
    axis([0.01 1.01 0 110])
end

end