function [] = plotSpeed()
% [] = plotSpeed()
%
% PLOTSPEED plots MT CDS speed tuning curves. In this experiment, a
% single bar is drifting over the entire visual field either to the right
% (preferred direction) and left (anti-preferred) direction at different
% speeds. MT CDS cells are either "band-pass", "low-pass" or "high-pass".
% This script reproduces Fig. 6 in (Beyeler, Dutt, and Krichmar, 2013;
% publication to be named later).
%
% A suitable stimulus can be created using scripts/v1MTLIP/makeBarSpeed.m,
% which can then be plugged into the CARLsim example model v1MTLIP.
%
% Author: Michael Beyeler <mbeyeler@uci.edu>
% Ver 07/23/13


%% LOAD PARAMS

% each speed/direction scene has a different number of frames. the
% structure whichStage contains, for each frame, an experiment number. this
% makes it easy to average the neuronal response over a single scene
load ../../videos/mkBarSpeed_ctrst0.2_32x32x7520.mat

vidLen=7520;
nrX=32;
nrY=32;
frameDur=50;

% here are the speeds
xSpeed(1,:) = [0.3125  0.5441  0.9473  1.6494  2.8717  5.0000]; % band-pass
xSpeed(2,:) = [0.0375  0.0653  0.1137  0.1979  0.3446  0.6000]; % low-pass
xSpeed(3,:) = [1.0000  1.5849  2.5119  3.9811  6.3096 10.0000]; % high-pass


%% PLOT RESULTS

% plot all 3x3 responses
% the diagonal should contain the tuning curves from the S&H paper
thisNeuron = 7079; % 528    ; %nrX*nrX*6 + round(nrX*nrX/2);
for mt=1:3
    disp(['MT' num2str(mt) 'CDS'])
    spk = readSpikes(['../../Results/v1MTLIP/spkMT' num2str(mt) 'CDS.dat'],frameDur);
    spk(vidLen,8*nrX*nrY)=0; % grow to right size
    
    % normalize to Hz
    spk = spk/frameDur*1000;
    
    % reshape and remove border pixels
    spk = reshape(spk,vidLen,nrX,nrY,8);
    spk = reshape(spk(:,10:22,10:22,1),vidLen,[]);
    
    % for each stage (direction and speed), compute mean and standard
    % deviation of population response
    muData=zeros(1,max(unique(whichStage)));
    stdData=zeros(size(muData));
    for i=1:max(unique(whichStage))
        % first subpopulation, only this stage, ignore border
        muData(i) = mean(reshape(spk(whichStage==i,:),1,[]));
        stdData(i) = std(reshape(spk(whichStage==i,:),1,[]));
    end
    
    subplot(1,3,mt)
    errorbar([xSpeed(mt,:);xSpeed(mt,:)]', ...
        [muData(12*(mt-1)+1:12*(mt-1)+6);muData(12*(mt-1)+7:12*(mt-1)+12)]', ...
        [stdData(12*(mt-1)+1:12*(mt-1)+6);stdData(12*(mt-1)+7:12*(mt-1)+12)]',...
        '.-','LineWidth',2,'MarkerSize',25);
    set(gca,'xscale','log');
    
    %    semilogx(xSpeed(mt,:),[x(12*(mt-1)+1:12*(mt-1)+6);x(12*(mt-1)+7:12*(mt-1)+12)]','.-','LineWidth',2,'MarkerSize',25);
    title(['MT' num2str(mt)],'FontSize',14);
    xlabel('speed (pixels/frame)','FontSize',14);
    ylabel('activity (spikes/sec)','FontSize',14);
    set(gca,'FontSize',14)
end
legend('preferred direction','anti-preferred direction');

end