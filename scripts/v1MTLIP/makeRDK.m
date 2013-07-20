function I = makeRDK()
% makeRDK()
%
% Creates the RDK stimulus used to produce Fig. 7. 
% Dots are drifting in random directions, with a given percentage of dots
% drifting coherently into the same direction ("coherence level"). Dots
% move at a given speed. A trial consists of a particular combination of
% coherence level and coherent drift direction (shown for nrF frames).
% Repeat trials nrTr times.
%
% This function uses scripts from an open-source MATLAB package of
% Simoncelli & Heeger's Motion Energy model, obtained from
% http://www.cns.nyu.edu/~lcv/MTmodel/ version 1.0 (10/14/05).
% Authors: Timothy Saint (saint@cns.nyu.edu) and Eero P. Simoncelli
% (eero.simoncelli@nyu.edu)
%
% Author: Michael Beyeler
% Ver 07/20/2013

%% make RDK stimulus
nrX=32; % dimension X
nrY=32; % dimension Y
nrF=20; % number of frames per trial
nrTr=10; % number of trials

% RDK stimulus has the following properties
speed=1.0; % pixels/frame
density=0.15; % dot density
delayPeriod=0;

ii=0;

% for all coherence levels (fraction of dots drifting in coherent
% direction)
for coherence = [0.1 1 5:5:20 30 40 50]*0.01
    % for a number of trials
    for trial=1:nrTr
        % for all directions
        for dir=(0:7)*pi/4
            % create an RDK stimulus using the S&H Matlab toolbox
            s = mkDots([nrX nrY nrF], dir,  speed, density, coherence);
            
            % you can add a delay period in which a blank frame is shown
            if delayPeriod>0
                s(:,:,nrF+1:nrF+delayPeriod)=zeros(nrX,nrY,delayPeriod);
                stim(:,:,ii*(nrF+delayPeriod)+1:(ii+1)*(nrF+delayPeriod))=s;
            else
                stim(:,:,ii*nrF+1:(ii+1)*nrF)=s;
            end
            ii=ii+1;
        end
    end
end

% normalize contrast
stim = stim-min(stim(:));
stim = stim./max(stim(:));

% convert to RGB representation
for i=1:size(stim,3)
    I{i}(:,:,1) = stim(:,:,i)*255;
    I{i}(:,:,2) = stim(:,:,i)*255;
    I{i}(:,:,3) = stim(:,:,i)*255;
end
clear stim;

% write to RGB file
writeFramesToRgbFile(['mkRDK_' num2str(nrX) 'x' num2str(nrY) 'x' ...
    num2str(length(I)) '.dat'],I,false);

end
