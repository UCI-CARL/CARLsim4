function [] = plotGratingPlaidCorrelation()
% [] = plotGratingPlaidCorrelation()
%
% This script computes the pattern index for all MT CDS cells (blue open
% circles) and all MT PDS cells (red crosses), and plots them as a Fisher
% Z-score. The blue solid lines are the classification region boundaries,
% with a criterion of 1.28.
% This script reproduces Fig. 5 of Beyeler, Richert, Dutt, and Krichmar
% (2014).
% A suitable stimulus can be created using
% scripts/v1MTLIP/makeTuningGrating.m and scripts/v1MTLIP/makeTuningPlaid.m
%
% Author: Michael Beyeler <mbeyeler@uci.edu>
% Ver 07/28/13


%% LOAD PARAMS %%

nrX=32;
nrY=32;
nPnt = 24;   % number of stim directions (data points)
nrF=50;      % number of frames per stim direction
frameDur=50; % number of ms each frame is presented
nB = 5;      % margin; border pix to remove

% load spike files
CDS = readSpikes('../../results/v1MTLIP/spkMT1CDS.dat',frameDur*nrF);
PDS = readSpikes('../../results/v1MTLIP/spkMT1PDS.dat',frameDur*nrF);

% remove border pix
CDS = reshape(CDS(:,2*nrX*nrY+1:3*nrX*nrY),[],nrX,nrY);
CDS = CDS(:,nB+1:nrX-nB,nB+1:nrY-nB);
PDS = reshape(PDS(:,2*nrX*nrY+1:3*nrX*nrY),[],nrX,nrY);
PDS = PDS(:,nB+1:nrX-nB,nB+1:nrY-nB);


%% COMPUTE CDS INDEX %%
clear Rp Rc Zp Zc;
for i=1:(nrX-2*nB)
    for j=1:(nrY-2*nB)
        % rc: simple correlation between component prediction and data
        % component prediction: CDS response to two gratings, superimposed
        pred_c = [CDS(6:24,i,j);CDS(1:5,i,j)] + [CDS(21:24,i,j);CDS(1:20,i,j)];
        % data: what we observed in 25:48
        rc=corr(pred_c,CDS(25:48,i,j));
        
        % rp: simple correlation between pattern prediction and data
        % pattern prediction: same as CDS response to grating
        pred_p = CDS(1:24,i,j);
        % data: what we observed
        rp=corr(pred_p,CDS(25:48,i,j));
        
        % rpc: simple correlation between predictions
        rpc=corr(pred_c,pred_p);
        
        Rp((i-1)*(nrX-2*nB)+j) = (rp-rc*rpc)/sqrt((1-rc^2)*(1-rpc^2));
        Rc((i-1)*(nrX-2*nB)+j) = (rc-rp*rpc)/sqrt((1-rp^2)*(1-rpc^2));
        
        Zp((i-1)*(nrX-2*nB)+j) = atanh(Rp((i-1)*(nrX-2*nB)+j))/sqrt(1/(nPnt-3));
        Zc((i-1)*(nrX-2*nB)+j) = atanh(Rc((i-1)*(nrX-2*nB)+j))/sqrt(1/(nPnt-3));
        
        % instead of atanh one can also use these ones:
        % Zp((i-1)*23+j) = 0.5*log((1+Rp((i-1)*23+j))/(1-Rp((i-1)*23+j)))/sqrt(1/(24-3));
        % Zc((i-1)*23+j) = 0.5*log((1+Rc((i-1)*23+j))/(1-Rc((i-1)*23+j)))/sqrt(1/(24-3));
        
        
    end
end

plot(Zc,Zp,'ob');
hold on


%% COMPUTE PDS INDEX %%
clear Rp Rc Zp Zc;
for i=1:(nrX-2*nB)
    for j=1:(nrY-2*nB)
        % rc: simple correlation between component prediction and data
        % component prediction: CDS response to two gratings, superimposed
        pred_c = [CDS(6:24,i,j);CDS(1:5,i,j)] + [CDS(21:24,i,j);CDS(1:20,i,j)];
        % data: what we observed in 25:48
        rc=corr(pred_c,PDS(25:48,i,j));
        
        % rp: simple correlation between pattern prediction and data
        % pattern prediction: same as CDS response to grating
        pred_p = CDS(1:24,i,j);
        % data: what we observed
        rp=corr(pred_p,PDS(25:48,i,j));
        
        % rpc: simple correlation between predictions
        rpc=corr(pred_c,pred_p);
        
        Rp((i-1)*(nrX-2*nB)+j) = (rp-rc*rpc)/sqrt((1-rc^2)*(1-rpc^2));
        Rc((i-1)*(nrX-2*nB)+j) = (rc-rp*rpc)/sqrt((1-rp^2)*(1-rpc^2));
        
        % Zp((i-1)*23+j) = 0.5*log((1+Rp((i-1)*23+j))/(1-Rp((i-1)*23+j)))/sqrt(1/(24-3));
        % Zc((i-1)*23+j) = 0.5*log((1+Rc((i-1)*23+j))/(1-Rc((i-1)*23+j)))/sqrt(1/(24-3));
        
        Zp((i-1)*(nrX-2*nB)+j) = atanh(Rp((i-1)*(nrX-2*nB)+j))/sqrt(1/(nPnt-3));
        Zc((i-1)*(nrX-2*nB)+j) = atanh(Rc((i-1)*(nrX-2*nB)+j))/sqrt(1/(nPnt-3));
    end
end


%% PLOT IT %%
plot(Zc,Zp,'xr');
line([-12 0],[1.28 1.28]);line([0 20],[1.28 20+1.28])
line([1.28 1.28],[-12 0]);line([1.28 20+1.28],[0 20])
axis square
xlabel('Z-transformed component correlation (Z_c)');
ylabel('Z-transformed pattern correlation (Z_p)');
legend('component','pattern')

end