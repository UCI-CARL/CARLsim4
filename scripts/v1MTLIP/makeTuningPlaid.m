function I = makeTuningPlaid()
% I = makeTuningPlaid()
%
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

%% create plaid stimulus

nrX=32;
nrY=32;
nrF=50;

nDataPoints=24;

% mt2sin([0 1.5])
gratingSf=0.1205; % 0.2051;
gratingTf=0.1808; % 0.0718;
plaidAngle=2.0944;
plaidContrast=0.3;

xDirection = (0:nDataPoints-1)*2*pi/nDataPoints;
% xDirection = linspace(0, 2.*pi, nDataPoints);

stim=[];
for i=1:nDataPoints
    s=mkPlaid([nrX nrY nrF], xDirection(i), gratingSf, gratingTf, plaidAngle, plaidContrast);
    stim(:,:,(i-1)*nrF+1:i*nrF)=s;
end

stim = stim-min(stim(:));
stim = stim./max(stim(:));


for i=1:size(stim,3)
    I{i}(:,:,1) = stim(:,:,i)*255;
    I{i}(:,:,2) = stim(:,:,i)*255;
    I{i}(:,:,3) = stim(:,:,i)*255;
end

writeFramesToRgbFile(['mkPlaid_' num2str(nrX) 'x' num2str(nrY) 'x' ...
    num2str(length(I)) '.dat'],I,false);



%% use S&H plaid stimulus blaster


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
plaidAngle = 'default';
plaidContrast = 'default';

% parse the varargin
                    stimSz = varargin{1};
                    plaidDirection = varargin{2};
                    gratingSf = varargin{3};
                    gratingTf = varargin{4};
if nargin >= 5      plaidAngle = varargin{5};       end
if nargin >= 6      plaidContrast = varargin{6};    end

% assign default values where appropriate
if strcmp(plaidAngle, 'default');       plaidAngle = (2/3)*pi;      end
if strcmp(plaidContrast, 'default');    plaidContrast = 1;          end

firstDirection = mod(plaidDirection + plaidAngle/2, 2*pi);
secondDirection = mod(plaidDirection - plaidAngle/2, 2*pi);
firstGrating =  mkSin(stimSz, firstDirection, gratingSf, gratingTf, ...
                      plaidContrast/2);
secondGrating = mkSin(stimSz, secondDirection, gratingSf, gratingTf, ...
                      plaidContrast/2);
s = firstGrating + secondGrating;
end

end
