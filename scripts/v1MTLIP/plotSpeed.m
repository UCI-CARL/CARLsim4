function [] = plotSpeed()
% [] = plotSpeed()
%
% PLOTSPEED plots MT CDS speed tuning curves. In this experiment, a
% single bar is drifting over the entire visual field either to the right
% (preferred direction) and left (anti-preferred) direction at different
% speeds. MT CDS cells are either "band-pass", "low-pass" or "high-pass".
% This script reproduces Fig. 6 of Beyeler, Richert, Dutt, and Krichmar
% (2014).
%
% A suitable stimulus can be created using scripts/v1MTLIP/makeBarSpeed.m,
% which can then be plugged into the CARLsim example model v1MTLIP.
%
% Author: Michael Beyeler <mbeyeler@uci.edu>
% Ver 2/5/14


%% LOAD PARAMS

% NOTE: the following params need to be changed for stimulus sizes other
% than 32x32

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

% pick a neuron in the first subpopulation (selective to rightward motion)
% near the center of the visual field
thisNeuron = sub2ind([nrX nrY 8],nrX/2,nrY/2+1);
for mt=1:3
    disp(['MT' num2str(mt) 'CDS'])
    spk = readSpikes(['../../results/v1MTLIP/spkMT' num2str(mt) 'CDS.dat'],frameDur);
    spk(vidLen,8*nrX*nrY)=0; % grow to right size
    
    % normalize to Hz
    spk = spk/frameDur*1000;
    
    % average neuron's response over all relevant image frames (where
    % whichStage==i)
    x=zeros(1,max(unique(whichStage)));
    for i=1:max(unique(whichStage))
        x(i)=mean(spk(whichStage==i,thisNeuron));
    end
    
    subplot(3,1,mt)
    semilogx(xSpeed(mt,:),[x(12*(mt-1)+1:12*(mt-1)+6);x(12*(mt-1)+7:12*(mt-1)+12)]','.-','LineWidth',2,'MarkerSize',25);
    title(['MT' num2str(mt)],'FontSize',14);
    xlabel('speed (pixels/frame)','FontSize',14);
    ylabel('activity (spikes/sec)','FontSize',14);
    set(gca,'FontSize',14)
    legend('preferred direction','anti-preferred direction');
end

end
