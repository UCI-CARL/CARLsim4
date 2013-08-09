function stim = makeTuningPlaid(nrX, nrY, nrF, nPnts, plaidContrast,...
    plaidAngle, gratingSf, gratingTf)
% stim = makeTuningPlaid(nrX, nrY, nrF, nPnts, plaidContrast, plaidAngle,
%                   gratingSf, gratingTf)
%
% MAKETUNINGPLAID creates a plaid stimulus consisting of two superimposed
% sinusoidal gratings. The stimulus is both returned and written to file,
% such that it can be plugged into CARLsim example model v1MTLIP.
% Use default parameter values to reproduce the stimulus used in CARLsim
% example v1MTLIP.
% Input arguments:
%   nrX           number of neurons in direction X
%   nrY           number of neurons in direction Y
%   nrF           number of frames per stimulus drift direction
%   nPnts         number of data points (samples 360deg drift direction)
%   plaidContrast contrast of the plaid (between 0 and 1)
%   plaidAngle    separating angle of the two gratings
%   gratingSf     spatial frequency component of the grating
%   gratingTf     temporal frequency component of the grating
%
% Output argument:
%   stim          cell array of frames, each frame has dim nrX x nrY
%
%
% This function uses scripts from an open-source MATLAB package of
% Simoncelli & Heeger's Motion Energy model, obtained from
% http://www.cns.nyu.edu/~lcv/MTmodel/ version 1.0 (10/14/05).
% Authors: Timothy Saint (saint@cns.nyu.edu) and Eero P. Simoncelli
% (eero.simoncelli@nyu.edu)
%
% Author: Michael Beyeler <mbeyeler@uci.edu>
% Ver 07/20/2013

%% SET PARAMS %%

if nargin<1,nrX=32;end
if nargin<2,nrY=32;end
if nargin<3,nrF=50;end
if nargin<4,nPnts=24;end
if nargin<5,plaidContrast=0.3;end
if nargin<6,plaidAngle=2.0944;end
if nargin<7,gratingSf=0.1205;end
if nargin<8,gratingTf=0.1808;end

xDirection = (0:nPnts-1)*2*pi/nPnts;


%% CREATE PLAID STIMULUS USING S&H MODEL
frames=[];
for i=1:nPnts
    s=mkPlaid([nrX nrY nrF], xDirection(i), gratingSf, gratingTf, plaidAngle, plaidContrast);
    frames(:,:,(i-1)*nrF+1:i*nrF)=s;
end

frames = frames-min(frames(:));
frames = frames./max(frames(:));


%% APPEND ALL FRAMES AND WRITE TO FILE
for i=1:size(frames,3)
    stim{i}(:,:,1) = frames(:,:,i)*255;
    stim{i}(:,:,2) = frames(:,:,i)*255;
    stim{i}(:,:,3) = frames(:,:,i)*255;
end

writeFramesToRgbFile(['mkPlaid_' num2str(nrX) 'x' num2str(nrY) 'x' ...
    num2str(length(stim)) '.dat'],stim,false);



%% USE S&H MODEL TO CREATE STIMULUS


% s = mkPlaid(stimSz, plaidDirection, gratingSf, gratingTf, plaidAngle,
%            plaidContrast)
%
% mkPlaid returns a drifting plaid stimulus
%
% Required arguments:
% stimSz            the size of the stimulus in [Y X T] coordinates
% plaidDirection    the direction of motion
% gratingSf         the spatial frequency of the plaid's grating components
% gratingTf         the temporal frequency of the plaid's grating components
%
% Optional arguments:
% plaidAngle        the angle between the grating components in radians.
%                   DEFAULT = (2/3)*pi (120 degrees).
% plaidContrast     the overall contrast of the plaid.

function s = mkSin(varargin)

% The following arguments are optional and by default are 'default'
angle = 'default';
contrast = 'default';

% parse the varargin
                    stimSz = varargin{1};
                    direction = varargin{2};
                    gratSf = varargin{3};
                    gratTf = varargin{4};
if nargin >= 5      angle = varargin{5};       end
if nargin >= 6      contrast = varargin{6};    end

% assign default values where appropriate
if strcmp(angle, 'default');       angle = (2/3)*pi;      end
if strcmp(contrast, 'default');    contrast = 1;          end

firstDirection = mod(direction + angle/2, 2*pi);
secondDirection = mod(direction - angle/2, 2*pi);
firstGrating =  mkSin(stimSz, firstDirection, gratSf, gratTf, ...
                      contrast/2);
secondGrating = mkSin(stimSz, secondDirection, gratSf, gratTf, ...
                      contrast/2);
s = firstGrating + secondGrating;
end

end
