function [] = plotRDKdecision()
% [] = plotRDKdecision()
%
% PLOTRDKDECISION plots the results of the random dot kinematogram (RDK)
% two-alternative forced-choice (2AFC) experiment. This script generates
% both the psychometric and the chronometric function. Simulation data are
% compared to psychophysical data from (Roitman & Shadlen, 2002).
% The RDK stimulus was constructed out of approximately 150 dots (15 % dot
% density, maximum stimulus contrast) on a 32x32 input movie
% This script reproduces Fig. 7 of Beyeler, Richert, Dutt, and Krichmar
% (2014).
% A suitable stimulus can be created using scripts/v1MTLIP/makeRDK.m
%
% Author: Michael Beyeler <mbeyeler@uci.edu>
% Ver 2/5/14

% NOTE: 64-bit MATLAB may be required to run this script (in order to
% handle objects > 4GB)

frameDur=1000; % frame duration
nDir=8; % number of directions
nRep=10; % number of trials
cohVec=[0.1 1 5:5:20 30 40 50]; % coherence levels
nDataPoints=length(cohVec); % number of data points
spkPerNeur = 10; % spikes per neuron needed to reach a decision

s = readSpikes('../../results/v1MTLIP/spkLIP.dat',1);
s(nDir*nRep*nDataPoints*frameDur,1) = 0; % grow it to right size...
s=reshape(s,frameDur,nDir,nRep,nDataPoints,[],8);
nNeur = size(s,4);

correct = zeros(nRep*nDir,nDataPoints);
times = zeros(nRep*nDir,nDataPoints);

measureAtSpike = round(nNeur*spkPerNeur); % # spikes needed for decision
for i=1:nDir
    for r=1:nRep
        for c=1:nDataPoints
            % find the pool to first emit measureAtSpike number of spikes
            
            % we are searching a 2-D matrix <#frames x #neurons>, so find will
            % return all spike times ti and neuron ids id
            [ti,id]=find(squeeze(s(:,i,r,c,:,mod(i-1,8)+1)));
            
            % ti has to be sorted; append some values in case there weren't
            % enough spikes
            spikeTimesPref = [sort(ti); inf(measureAtSpike,1)];
            
            % for antiPref, count the spikes in the group coding for the
            % opposite direction: mod(i-1+4,8) instead of mod(i-1,8)
            [ti,id]=find(squeeze(s(:,i,r,c,:,mod(i-1+4,8)+1)));
            spikeTimesAntiPref = [sort(ti); inf(measureAtSpike,1)];
            
            % correct is true if pref group reached threshold before
            % antipref
            correct((r-1)*nDir+i,c) = spikeTimesPref(measureAtSpike)<spikeTimesAntiPref(measureAtSpike);
            times((r-1)*nDir+i,c) = min(spikeTimesPref(measureAtSpike),spikeTimesAntiPref(measureAtSpike));
            
            if times((r-1)*nDir+i,c)>frameDur
                % TODO: if it takes too long to make a decision, this data
                % point should be excluded from the mean calculation
                % (see below)
                % NOTE: this did not happen in Fig. 7
                correct((r-1)*nDir+i,c)=0;
                times((r-1)*nDir+i,c)=frameDur;
                disp(['(dir=' num2str(i) ',smpl=' num2str(c) ') no decision reached '...
                    'within ' num2str(frameDur) ' ms']);
            end
        end
    end
end
clear s;

subplot(121)
errorbar(cohVec,mean(correct)*100,100*sqrt(mean(correct).*(1-mean(correct))/size(correct,1)),'.-','LineWidth',2,'MarkerSize',30);
hold on
% from Fig.3A in Roitman & Shadlen, 2002 (eyeballed)
plot([3.2 6.4 12.8 25.6 51.2],[68 77 99 100 100],'r','LineWidth',5);
xlabel('Motion strength (% coherence)','FontSize',14);
ylabel('Accuracy (% correct)','FontSize',14);
axis([0.1 51 39 101]);
set(gca,'FontSize',14);
set(gca,'XScale','log');


subplot(122)
errorbar(cohVec,mean(times),std(times),'.-','LineWidth',2,'MarkerSize',30);
hold on
% T2 choices (outside RF) from Roitman & Shadlen, 2002 (Table 2)
plot([0 3.2 6.4 12.8 25.6 51.2],[827.6 818.3 758.3 678.1 551.1 436.9],'r','LineWidth',5);
xlabel('Motion strength (% coherence)','FontSize',14);
ylabel('Reaction time (ms)','FontSize',14);
axis([-1 51 -20 1020]);
set(gca,'FontSize',14);

end
