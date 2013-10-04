function [stim,whichStage] = makeBarSpeed(stimContrast, writeFile)
% [stim,whichStage] = mkBarSpeed(stimContrast, writeFile)
% This function creates the bar stimulus for the speed tuning experiment.
% The stimulus contains a single bar, moving either in the preferred or
% anti-preferred direction at different speeds.
%
% Inputs:
%   stimContrast: stimulus contrast, e(0,1]. Default: 0.2.
%   writeFile:    true|false, will write STIM to file if set to true (using
%                 the function WRITEFRAMESTORGBFILE). Default: false.
% Outputs:
%   stim:         A cell array, each element is a frame consisting of
%                 <nrX x nrY x nrT>, where nrT varies for each stage (a
%                 configuration given by bar direction, bar speed, and bar
%                 width).
%   whichStage:   For each frame (for each element in stim), this vector
%                 stores a stage ID number. Find all frames that belong to
%                 stage ID 2? Simply do whichStage==2.
%
% This function uses scripts from an open-source MATLAB package of
% Simoncelli & Heeger's Motion Energy model, obtained from
% http://www.cns.nyu.edu/~lcv/MTmodel/ version 1.0 (10/14/05).
% Authors: Timothy Saint (saint@cns.nyu.edu) and Eero P. Simoncelli
% (eero.simoncelli@nyu.edu)
%
% Author: Michael Beyeler <mbeyeler@uci.edu>
% Ver 10/4/2013

addpath ../common

if nargin<2,writeFile=false;end
if nargin<1,stimContrast=0.2;end

if stimContrast<=0 || stimContrast>1
    error('stimContrast must be e(0,1]');
end


%% Make drifting bar stimulus for speed tuning experiment
% The stimulus contains a single bar, moving either in the preferred or
% anti-preferred direction at different speeds. We'll call each such
% configuration a stage of the experiment.
% The code is based on S&H's own Matlab script shTuneBarSpeed.m; adjusted
% to match our neuron population and stripped of all the overhead code.

% Problem: Because of the varying speeds, it will take a variable number of
% frames to let a bar drift across the entire visual field.
% Solution: Build up stim frame-by-frame. Count all frames in cntFr.
% Keep track of which stage (nested loop) each frame belongs to in
% whichExp. In other words, at the end of the experiment whichExp will be
% of size <1 x cntFr> and contain a stage ID for each frame. If you want to
% find all frames that belong to stage 2, just do whichExp==2.

nrX=32; nrY=32;               % canvas size
nrF=50;                       % number of frames per stage
neuronSpeeds = [1.5 0.125 9]; % preferred speed in pixels/frame for 3
                              % neuron populations
speedMinMax = [.3125 5; .0375 .6; 1 10];
widthBarEdge = [2 1 11];
nDataPoints = 6;
rfRadius = 20;
widthBar = 1;
timePadSz = 4; % pad the stimulus with zeros at the beginning and the end


cntFr=0;
cntExp=0;
whichStage=[];
stim={};

% for each pool of speed-selective neurons (i.e., MT1, MT2, and MT3)
for n=1:length(neuronSpeeds)
%     minSpeed = 0.25*neuronSpeeds(n);
%     maxSpeed = 4*neuronSpeeds(n);
    barDir = 0;
        minSpeed = speedMinMax(n,1);
        maxSpeed = speedMinMax(n,2);
    %     barDirection = neurons(n,1);
    
    xSpeed = linspace(log2(minSpeed), log2(maxSpeed), nDataPoints);
    xSpeed = 2.^xSpeed;
    
    % for the preferred and anti-preferred direction
    for dir=[barDir mod(barDir+pi,2*pi)]
        % for each data point
        for i=1:length(xSpeed)
            % update an experiment counter
            cntExp = cntExp+1;
            
            % generate the drifting bar stimulus
            nBarPasses = round(nrF .* xSpeed(i) ./ rfRadius);
            if nBarPasses == 0,nBarPasses = 1;end
            stimSize = [nrX nrY ceil(nBarPasses*2*rfRadius/xSpeed(i))];
            
            thisBar = mkBar(stimSize, dir, xSpeed(i), widthBar, ...
                widthBarEdge(n), 2*rfRadius, -rfRadius); %1*rfRadius
            thisBar = thisBar.*stimContrast;
            
            % NOTE: the original S&H shTuneBarSpeed.m has a low-pass filter
            % applied at this point, but the variable thisBarFiltered does
            % not seem to get used for generating thisStim...
            
            % Pad the stimulus with zeros at the beginning and end.
            thisStim = zeros(stimSize(1),stimSize(2),stimSize(3)+timePadSz);
            thisStim(:,:,timePadSz/2+1:end-timePadSz/2) = thisBar;

            % Problem: size(thisStim,3) varies from iteration to iteration
            % Idea: Add it frame by frame to I{}, keep track of which
            % experiment (nested loop) this frame belongs to in whichExp
            for j=1:size(thisStim,3)
                stim{cntFr+j}(:,:,1) = thisStim(:,:,j)*255;
                stim{cntFr+j}(:,:,2) = thisStim(:,:,j)*255;
                stim{cntFr+j}(:,:,3) = thisStim(:,:,j)*255;
                whichStage(end+1) = cntExp;
            end
            
            cntFr = cntFr + size(thisStim,3);            
        end
    end
end

% write stim to file if flag is set
if writeFile
    filename=['mkBarSpeed_ctrst' num2str(stimContrast) ...
        '_' num2str(nrX) 'x' num2str(nrY) 'x' num2str(length(stim))];
    writeFramesToRgbFile([filename '.dat'],stim,false);
    save([filename '.mat'],'whichStage');
end



%% The original untouched S&H functions used to create the stimulus

    % s = mkBar(stimSz, barDirection, barSpeed, barWidth, barEdgeWidth,
    %           pixelsBetweenBars, barStartingPosition);
    %
    % Make a drifting bar stimulus.
    %
    % Required arguments:
    % stimSz                the size of the entire stimulus, in [Y X T] coordinates
    % barDirection          the direction of the bar motion in radians with 0 = right
    % barSpeed              the speed of the bar motion in frames/second
    %
    % Optional arguments:
    % barWidth              the width of the center of the bar in pixels.
    %                       DEFAULT = 1
    % barEdgeWidth          the width of the cosine edges of the bar in pixels. An
    %                       edge of width barEdgeWidth will be tacked onto both
    %                       sides of the bar, so the total width will be
    %                       barWidth + 2*barEdgeWidth. DEFAULT = 2.
    % pixelsBetweenBars     the number of pixels between the bars in the stimulus
    %                       DEFAULT = stimSz(1)/4.
    % barStartingPosition   the starting position of the first bar in pixels.
    %                       The coordinate system is a line lying along the
    %                       direction of the bar motion and passing through the
    %                       center of the stimulus. The point 0 is the center
    %                       of the stimulus. DEAFULT = 0
    function driftingBar = mkBar(varargin)
        % The following arguments are optional and by default are 'default'
        barWidth = 'default';
        barEdgeWidth = 'default';
        pixelsBetweenBars = 'default';
        barStartingPosition = 'default';
        
        % parse the varargin
        stimSz = varargin{1};
        barDirection = varargin{2};
        barSpeed = varargin{3};
        if nargin >= 4;         barWidth = varargin{4};             end
        if nargin >= 5;         barEdgeWidth = varargin{5};         end
        if nargin >= 6;         pixelsBetweenBars = varargin{6};    end
        if nargin >= 7;         barStartingPosition = varargin{7};  end
        
        % Assign default values where appropriate
        if strcmp(barWidth, 'default');             barWidth = 5;                       end
        if strcmp(barEdgeWidth, 'default');         barEdgeWidth = 3;                   end
        if strcmp(pixelsBetweenBars, 'default');    pixelsBetweenBars = stimSz(1)/4;    end
        if strcmp(barStartingPosition, 'default');  barStartingPosition = 0;            end
        
        % Calculate a few things.
        gratingSf = 1/pixelsBetweenBars;
        gratingTf = gratingSf * barSpeed;
        
        
        % The plan of action: make a drifting grating with the right velocity and
        % orientation. The period of the sin wave will be the same as the
        % distance between the bars that the user wants. Then we make bars out of
        % the peaks of the sin wave.
        
        % Make the grating
        gratingPhase = barStartingPosition * gratingSf;
        driftingBar = 2*mkSin(stimSz, barDirection, gratingSf, gratingTf, 1, gratingPhase) - 1;
        
        % Find the thresholds
        barInnerThreshold = cos(2*pi*gratingSf*barWidth/2);
        barOuterThreshold = cos(2*pi*gratingSf*(barWidth/2 + barEdgeWidth));
        
        % There are three regions: where the stimulus should be one (the centers of
        % the bars), where it should be zero (outside the bars), and where it
        % should be somehwere in between (edges of the bars). Find them
        wOne = find(driftingBar >= barInnerThreshold);
        wEdge = find(driftingBar < barInnerThreshold & ...
            driftingBar > barOuterThreshold);
        wZero = find(driftingBar <= barOuterThreshold);
        
        % Set the regions to the appropriate level
        driftingBar(wOne) = 1;
        driftingBar(wZero) = 0;
        
        % keyboard
        driftingBar(wEdge) = acos(driftingBar(wEdge));      % now it ranges from 0 to 2*pi
        driftingBar(wEdge) = driftingBar(wEdge)/(2*pi*gratingSf);       % now it ranges from 0 to 1 spatial period
        driftingBar(wEdge) = (pi/2)*(driftingBar(wEdge) - barWidth/2)/(barEdgeWidth);
        driftingBar(wEdge) = cos(driftingBar(wEdge));
    end


    % s = mkWin(stimSz, windowSz, anglePref, angleWidth, windowEdgeWidth, windowPosition);
    %
    % Make a cosine edged wedge in the Fourier domain.
    %
    % Required arguments:
    % stimSz            The size of the reponse in [Y X T] coordinates
    % windowSz          A 2-vector specifying size of the window. The first element
    %                   is required, and specifies the radius of the window. The
    %                   second element is optional and specifies the duration of
    %                   the window. If the duration isn't supplied, the window will
    %                   last for the entire length of the stimulus.
    % anglePref         the preferred direction of motion.
    % angleWidth        the azimuthal width of the wedge.
    %
    % Optional arguments:
    % windowEdgeWidth   A 2-vector specifying the width of the smooth cosine
    %                   edge of the window. The first element is the width of
    %                   the spatial edge, the second element is the width of
    %                   the temporal edge. DEFAULT = 1/4 of the window size.
    %                   The window edges are added to the window, so if the
    %                   radius of the window is 5 and the edge is 3, the total
    %                   radius of the window will be 8.
    % windowPosition    The center point of the window. DEFAULT = the center of
    %                   the stimulus.
    function s =  mkWin(varargin)
        % the following arguments are optional and by default are 'default'
        windowEdgeWidth = 'default';
        windowPosition = 'default';
        
        % Parse the varargin
        stimSz = varargin{1};
        windowSz = varargin{2};
        anglePref = varargin{3};
        angleWidth = varargin{4};
        if nargin >= 5;     windowEdgeWidth = varargin{5};  end
        if nargin >= 6;     windowPosition = varargin{6};   end
        
        % Assign default values where appropriate
        if strcmp(windowEdgeWidth, 'default');  windowEdgeWidth = windowSz./4;          end;
        if strcmp(windowPosition, 'default');   windowPosition = floor(stimSz./2)+1;    end;
        
        % If user hasn't supplied a duration for the window, make the window last
        % as long as the stimulus.
        windowRadius = windowSz(1);
        if length(windowSz) == 1;
            windowDuration = stimSz(3);
        else
            windowDuration = windowSz(2);
        end
        
        windowSpatialEdgeWidth = windowEdgeWidth(1);
        if length(windowEdgeWidth) == 1
            windowTimeEdgeWidth = windowDuration/4;
        else
            windowTimeEdgeWidth = windowEdgeWidth(2);
        end
        
        angleWidth = .5*angleWidth;
        
        
        % Now make the window. We'll construct it in two pieces: spatial and
        % temporal.
        [xSpatial, ySpatial] = meshgrid([1:stimSz(1)]-windowPosition(1), ...
            [1:stimSz(2)]-windowPosition(2));
        rSpatial = sqrt(xSpatial.^2 + ySpatial.^2);
        wOne = find(rSpatial <= windowRadius);
        wEdge = find(rSpatial > windowRadius & ...
            rSpatial < windowRadius + windowSpatialEdgeWidth);
        wZero = find(rSpatial >= windowRadius + windowSpatialEdgeWidth);
        
        spatialWindow = zeros(stimSz(1), stimSz(2));
        spatialWindow(wOne) = 1;
        spatialWindow(wEdge) = cos((pi/2)*(rSpatial(wEdge) - windowRadius)/windowSpatialEdgeWidth);
        spatialWindow(wZero) = 0;
        
        
        % Now make the wedges
        thetaSpatial = atan3(ySpatial, xSpatial);
        
        thetaSpatialTop = mod(thetaSpatial+anglePref+pi, 2*pi);
        wOne = find(thetaSpatialTop >= pi-angleWidth & thetaSpatialTop <= pi + angleWidth);
        wedgeWindowTop = zeros(stimSz(1), stimSz(2));
        wedgeWindowTop(wOne) = 1;
        
        thetaSpatialBottom = mod(thetaSpatial+anglePref, 2*pi);
        wOne = find(thetaSpatialBottom >= pi-angleWidth & thetaSpatialBottom <= pi + angleWidth);
        wedgeWindowBottom = zeros(stimSz(1), stimSz(2));
        wedgeWindowBottom(wOne) = 1;
        
        tmp = zeros(stimSz);
        tmp(:,:,1:floor(stimSz(3)/2)+1) = repmat(spatialWindow.*wedgeWindowBottom, [1 1 floor(stimSz(3)/2)+1]);
        tmp(:,:,floor(stimSz(3)/2)+1:stimSz(3)) = ...
            repmat(spatialWindow.*wedgeWindowTop, [1, 1, stimSz(3)-floor(stimSz(3)/2)]);
        spatialWindow = tmp;
        
        % Now make the temporal window
        xTemporal = abs([1:stimSz(3)]-windowPosition(3));
        wOne = find(xTemporal <= windowDuration/2);
        wEdge = find(xTemporal > windowDuration/2 & ...
            xTemporal < windowDuration/2 + windowTimeEdgeWidth);
        wZero = find(xTemporal >= windowDuration/2 + windowTimeEdgeWidth);
        
        temporalWindow = zeros(1, 1, stimSz(3));
        temporalWindow(wOne) = 1;
        temporalWindow(wEdge) = cos((pi/2)*(xTemporal(wEdge) - windowDuration/2)/windowTimeEdgeWidth);
        temporalWindow(wZero) = 0;
        temporalWindow = repmat(temporalWindow, [stimSz(1) stimSz(2), 1]);
        
        % Take the pointwise product of the two windows to get the final result
        s = spatialWindow .* temporalWindow;
    end
end