function stim = makeTuningGrating(nrX, nrY, nrF, nPnts, gratingContrast,...
    gratingSf, gratingTf)
% stim = makeTuningGrating(nrX, nrY, nrF, nPnts, gratingContrast, 
%                   gratingSf, gratingTf)
%
% MAKETUNINGGRATING creates a sinusoidal grating stimulus. The stimulus is
% both returned and written to file, such that it can be plugged into
% CARLsim example model v1MTLIP.
% Use default parameter values to reproduce the stimulus used in CARLsim
% example v1MTLIP.
% Input arguments:
%   nrX             number of neurons in direction X
%   nrY             number of neurons in direction Y
%   nrF             number of frames per stimulus drift direction
%   nPnts           number of data points (samples 360deg drift direction)
%   gratingContrast contrast of the plaid (between 0 and 1)
%   gratingSf       spatial frequency component of the grating
%   gratingTf       temporal frequency component of the grating
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
% Ver 10/4/2013

addpath ../common

%% SET PARAMS
if nargin<1,nrX=32;end
if nargin<2,nrY=32;end
if nargin<3,nrF=50;end
if nargin<4,nPnts=24;end
if nargin<5,gratingContrast=0.3;end
if nargin<6,gratingSf=0.1205;end
if nargin<7,gratingTf=0.1808;end

xDirection = (0:nPnts-1)*2*pi/nPnts;

%% CREATE STIMULUS USING S&H MODEL
stim=[];
for i=1:nPnts
    s=mkSin([nrX nrY nrF],xDirection(i),gratingSf,gratingTf,gratingContrast);
    stim(:,:,(i-1)*nrF+1:i*nrF)=s;
end

stim = stim-min(stim(:));
stim = stim./max(stim(:));


%% APPEND ALL FRAMES AND WRITE TO FILE
for i=1:size(stim,3)
    I{i}(:,:,1) = stim(:,:,i)*255;
    I{i}(:,:,2) = stim(:,:,i)*255;
    I{i}(:,:,3) = stim(:,:,i)*255;
end

writeFramesToRgbFile(['mkGrid_' num2str(nrX) 'x' num2str(nrY) 'x' ...
    num2str(length(I)) '.dat'],I,false);


%% USE S&H MODEL TO CREATE STIMULUS

% S =  mkSin(stimSz, sinDirection, sinSf, sinTf, sinContrast, sinPhase)    
% 
% mkSing returns a drifting grating. This grating will have a mean value of
% .5 and a contrast of sinContrast.
%
% Required arguments:
% stimSz            the size of the stimulus in [y, x, t] coordinates.
% sinDirection      the direction, in radians (0 = rightward), of motion
% sinSf             spatial frequency in units of cycles/pixel
% sinTf             temporal frequency in units of cycles/frame
%
% Optional arguments:
% sinContrast       grating contrast. DEFAULT = 1.
% sinPhase          initial phase in periods. DEFAULT = 0.

function [res] = mkSin(varargin)

% the following variables are optional and by default are 'default'
sinContrast = 'default';
sinPhase = 'default';

% parse varargin
                    stimSz = varargin{1};
                    sinDirection = varargin{2};
                    sinSf = abs(varargin{3});
                    sinTf = abs(varargin{4});
if nargin >= 5;     sinContrast = varargin{5};      end
if nargin >= 6;     sinPhase = varargin{6};         end

% assign default values
if strcmp(sinContrast, 'default');      sinContrast = 1;            end
if strcmp(sinPhase, 'default');         sinPhase = 0;               end

% Error message if arguments are incorrectly formatted
if length(stimSz) ~= 3
    error('stimSz must be a 3-vector');
end

% Make a coordinate system
y = [1:stimSz(1)] - (floor(stimSz(1)/2)+1);
x = [1:stimSz(2)] - (floor(stimSz(2)/2)+1);
t = [0:stimSz(3)-1];

[y, x, t] = ndgrid(x, y, t);
y = -y;

res = cos(2*pi*sinSf*cos(sinDirection)*x + ...
          2*pi*sinSf*sin(sinDirection)*y - ...
          2*pi*sinTf*t + ...
          2*pi*sinPhase);
          
res = sinContrast.*res./2 + .5;
end

end