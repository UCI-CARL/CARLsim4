classdef InputStimulus < handle
    % IS = InputStimulus(varargin) creates a new instance of class
    % InputStimulus.
    %
    % IS = InputStimulus(width,height,mode) initializes an empty InputStimulus
    % object of canvas size width*height and a given image mode. width and
    % height will be rounded to the nearest integer (number of pixels). Image
    % mode can be either 'gray' (grayscale) or 'rgb'.
    %
    % IS = InputStimulus(width,height) initializes an empty InputStimulus
    % object of canvas size width*height in grayscale mode.
    %
    % IS = InputStimulus(width,height,'rgb') initializes an empty InputStimulus
    % object of canvas size width*height in rgb mode.
    %
    % IS = InputStimulus(fileName) loads an InputStimulus object from file.
    % fileName must be a valid relative/absolute path to a binary file that has
    % been saved using method InputStimulus.saveToFile.
    %
    % Version 8/14/14
    % Author: Michael Beyeler <mbeyeler@uci.edu>
    %
    % This class uses some scripts from a MATLAB package of Simoncelli &
    % Heeger's Motion Energy model, obtained from
    % http://www.cns.nyu.edu/~lcv/MTmodel/, version 1.0 (10/14/05).
    % Authors: Timothy Saint <saint@cns.nyu.edu> and Eero P. Simoncelli
    % <eero.simoncelli@nyu.edu>
    
    %% PROPERTIES
    properties (SetAccess = private)
        width;                      % stimulus width (pixels)
        height;                     % stimulus height (pixels)
        length;                     % stimulus length (number of frames)
        stim;                       % 3-D matrix width-by-height-by-length
        channels;                   % number of channels (gray=1, RGB=3)
        mode;                       % image mode (gray, RGB)
        
        supportedImageModes;        % list of supported image modes
        supportedNoiseTypes;        % list of supported noise types
        supportedStimulusTypes;     % list of supported stimulus types
    end
    
    properties (SetAccess = private, GetAccess = private)
        needToLoad;
        loadFile;
        defaultSaveName;
        plotAbortPlotting;
    end
    properties(Constant, GetAccess = private)
        fileSignature = 304698591;  % some unique identifier of IS binary files
    end
    
    
    %% PUBLIC METHODS
    methods
        % Use varargin to mimick function overloading
        function obj = InputStimulus(varargin)
            obj.width = -1;
            obj.height = -1;
            obj.length = 0;
            obj.stim = [];
            obj.channels = -1;
            obj.loadFile = '';
            obj.needToLoad = false;
            obj.defaultSaveName = 'inp'; % will be expanded when creating stim
            obj.mode = -1;
            obj.plotAbortPlotting;
            
            % the following image types are supported
            obj.supportedImageModes = {'gray'};
            
            % the following noise types are supported
            obj.supportedNoiseTypes = {'gaussian', 'localvar', 'poisson', ...
                'salt & pepper', 'speckle'};
            
            % the following stimulus types are supported
            obj.supportedStimulusTypes = {'grating','plaid','rdk'};
            
            isValidInput=false; % flag to indicate whether input is valid
            if numel(varargin)<1 || numel(varargin)>3
                error('Illegal amount of input arguments')
            end
            
            if numel(varargin)==1
                % only 1 input argument
                if ischar(varargin{1})
                    % input argument is string => load stimulus from file
                    obj.needToLoad = true;
                    obj.loadFile = varargin{1};
                    isValidInput = true;
                end
            elseif numel(varargin)>=2
                % 2 arguments
                if isnumeric(varargin{1}) && isnumeric(varargin{2})
                    obj.width = round(varargin{1});
                    obj.height = round(varargin{2});
                    isValidInput = true;
                end
                if numel(varargin)>=3 && ischar(varargin{3})
                    switch lower(varargin{3})
                        case 'gray'
                            obj.mode = 'gray';
                            obj.channels = 1;
                            isValidInput = true;
                        case 'rgb'
                            obj.mode = 'rgb';
                            obj.channels = 3;
                            isValidInput = true;
                            error('RGB not yet supported')
                        otherwise
                            error(['Unknown image mode "' varargin{3} '". ' ...
                                'Currently supported are: ' ...
                                this.supportedImageModes])
                    end
                else
                    obj.mode = 'gray';
                    obj.channels = 1; % default
                    isValidInput = true;
                end
            end
            
            if ~isValidInput
                error('Invalid input.');
            end
        end
        
        function addApertureToExistingFrames(this, frames, maskType, maskSize, ...
                maskCenter, bgGrayVal)
            % IS.addApertureToExistingFrames(frames, type, size, ptCenter,
            % bgGrayVal) adds an aperture window of a given MASKTYPE and
            % MASKSIZE to all specified FRAMES. The aperture will be drawn
            % around center pixel location MASKCENTER. At pixel positions that
            % lie within the aperture, the existing stimulus will be preserved.
            % The grayscale value of all pixel positions outside the aperture
            % will be set to BGGRAYVAL.
            %
            % FRAMES    - A vector of frame numbers to which the aperture shall
            %             be applied. Set it to -1 to select all frames. Default
            %             is all frames.
            %
            % MASKTYPE  - A string that specifies the aperture type. Default is
            %             'gaussian'.
            %             'gaussian'     - The aperture will be an ellipse whose
            %                              semi major-axis and semi minor-axis
            %                              will be equal to the width and height
            %                              specified by MASKSIZE, respectively.
            %             'rectangle'    - The aperture will be a rectangle
            %                              whose width and height are specified
            %                              by SIZE.
            %
            % MASKSIZE  - A 2-element vector specifying the width and height of
            %             the aperture, respectively. Default is
            %             [width/4,height/4].
            %
            % MASKCENTER- The pixel location of the center point of the
            %             aperture. Default is [width/2,height/2].
            %
            % BGGRAYVAL - The grayscale value of the background, in the range
            %             [0,255]. Default is 128.
            if nargin<6,bgGrayVal=128;end
            if nargin<5,maskCenter=[this.width/2 this.height/2];end
            if nargin<4,maskSize=[this.width/4 this.height/4];end
            if nargin<3,maskType='gaussian';end
            if nargin<2 || frames==-1,frames=1:this.length;end

            if ~isnumeric(frames) || ~isvector(frames)
                error('Frames must be a numeric vector')
            end
            if ~isnumeric(maskSize) || ~isvector(maskSize) || numel(maskSize)~=2
                error('Size must be a 2-element vector [width height]')
            end
            if ~isnumeric(maskCenter) || ~isvector(maskCenter) || numel(maskCenter)~=2
                error('Center point must be a 2-element vector [x y]')
            end
            if ~isnumeric(bgGrayVal) || bgGrayVal<0 || bgGrayVal>255
                error('Mean grayscale value must be in the range [0,255]')
            end

            % create aperture mask
            switch lower(maskType)
                case 'gaussian'
                    % find points within elliptic aperture
                    [X,Y] = meshgrid(1:this.width, 1:this.height);
                    mask = ((X-maskCenter(1))/maskSize(1)).^2 ...
                        + ((Y-maskCenter(2))/maskSize(2)).^2 <= 1;
                case 'rectangle'
                    [X,Y] = meshgrid(1:this.width, 1:this.height);
                    mask = abs(X-maskCenter(1))<=maskSize(1)/2 ...
                        & abs(Y-maskCenter(2))<=maskSize(2)/2;
                otherwise
                    error(['Unknown type "' maskType '"'])
            end
            for i=1:numel(frames)
                % get frame
                img = this.getFrames(frames(i));

                % apply mask and add background grayval
                img = mask'.*img + not(mask').*bgGrayVal/255;

                % store frame
                this.privSetFrame(frames(i),img);
            end
        end

        function addBar(this, length, barDirection, barSpeed, barWidth, ...
                barEdgeWidth, pixelsBetweenBars, barStartPos)
            % IS.addBar(length, barDirection, barSpeed, barWidth, barEdgeWidth,
            % pixelsBetweenBar, barStartPos) creates a drifting bar stimulus of
            % LENGTH frames drifting in BARDIRECTION at BARSPEED.
            %
            % This method is based on a script by Timothy Saint and Eero P.
            % Simoncelli at NYU. It was adapted for coordinate system [x,y,t].
            % For more information see beginning of this file.
            %
            % LENGTH            - The number of frames to create the stimulus
            %                     for. Default is 10.
            %
            % BARDIRECTION      - The direction of the bar motion in radians
            %                     (0=rightwards, pi/2=upwards; angle increases
            %                     counterclockwise). Default is 0.
            %
            % BARSPEED          - The speed of the bar in pixels/frame. Default
            %                     is 1.
            %
            % BARWIDTH          - The width of the center of the bar in pixels.
            %                     Default is 1.
            %
            % BAREDGEWIDTH      - The width of the cosine edges of the bar in
            %                     pixels. An edge of width BAREDGEWIDTH will be
            %                     tacked onto both sides of the bar, so the
            %                     total width will be BARWIDTH + 2*BAREDGEWIDTH.
            %                     Default is 3.
            %
            % PIXELSBETWEENBARS - The number of pixels between the bars in the
            %                     stimulus. Default is width.
            % BARSTARTPOS       - The starting position of the first bar in
            %                     pixels. The coordinate system is a line lying
            %                     along the direction of the bar motion and
            %                     passing through the center of the stimulus.
            %                     The point 0 is the center of the stimulus.
            %                     Default is 0.
            if nargin<8,barStartPos=0;end
            if nargin<7,pixelsBetweenBars=this.width;end
            if nargin<6,barEdgeWidth=3;end
            if nargin<5,barWidth=1;end
            if nargin<4,barSpeed=1;end
            if nargin<3,barDirection=0;end
            if nargin<2,length=10;end
            
            res = this.privMakeBar(length, barDirection, barSpeed, barWidth, ...
                barEdgeWidth, pixelsBetweenBars, barStartPos);

            % add frames to existing stimulus
            this.privAppendFrames(res);

            % update default save file name
            lastName = this.defaultSaveName(max(1,end-2):end);
            if ~strcmp(lastName,'Bar') % only save distinct names
                this.defaultSaveName = cat(2,this.defaultSaveName,'Bar');
            end
        end
            
        function addBlankFrames(this, length, grayVal)
            % IS.addBlankFrames(length) adds a total of LENGTH blank frames with
            % grayscale value GRAYVAL existing stimulus.
            %
            % LENGTH     - The number of blank frames to append. Default is 1.
            %
            % GRAYVAL    - The grayscale value of the background (must be in the
            %              range [0,255]. Default is 0 (black).
            if nargin<3,grayVal=0;end
            if nargin<2,length=1;end
            if ~isnumeric(grayVal) || numel(grayVal)~=1
                error('Grayscale value must be numeric')
            end
            if grayVal<0 || grayVal>255
                error('Grayscale value must be in the range [0,255]')
            end
            if ~isnumeric(length) || numel(length)~=1
                error('Length must be numeric')
            end
            
            % adjust grayscale range to [0,1]
            img = ones(this.width, this.height, length)*grayVal/255;
            
            % add frames to existing stimulus
            this.privAppendFrames(img);
        end
                
        function addDots(this, length, type, dotDirection, dotSpeed, ...
                dotDensity, dotCoherence, dotRadius, ptCenter, densityStyle, ...
                sampleFactor, interpMethod)
            % IS.addDots(length, type, dotDirection, dotSpeed, dotCoherence,
            % dotRadius, ptCenter, densityStyle, sampleFactor, interpMethod)
            % adds a field of drifting dots to the existing stimulus. The field
            % consists of roughly DOTDENSITY*imgWidth*imgHeight drifting dots,
            % of which a  fraction DOTCOHERENCE drifts coherently into a
            % DOTDIRECTION (either direction of motion or motion gradient).
            % Supported flow patterns include drifting into a certain direction,
            % expansion, contraction, rotations, and deformations.
            %
            % This method uses a script initially authored by Timothy Saint and
            % Eero P. Simoncelli at NYU. It was adapted for coordinate system
            % [x,y,t], and extended to feature patterns of expansion,
            % contraction, rotation, and deformation. For more information see
            % beginning of this file.
            %
            % LENGTH        - The number of frames you want to generate. Default
            %                 is 10.
            %
            % TYPE          - The type of motion stimulus. Currently supported
            %                 are:
            %                   'linear'      - The classic RDK stimulus, where
            %                                   a fraction DOTCOHERENCE of dots
            %                                   drift coherently into a
            %                                   direction DOTDIRECTION.
            %                   'rotExpContr' - Rotations, expansion,
            %                                   contraction, and mixtures of
            %                                   those types. These types lie
            %                                   along the spectrum of
            %                                   DOTDIRECTION as follows:
            %                                   0      expansion
            %                                   pi/2   counterclockwise rotation
            %                                   pi     contraction
            %                                   3*pi/2 clockwise rotation
            %                                   In between those points, the
            %                                   flow pattern acts as a mixture
            %                                   between expansion/rotation,
            %                                   contraction/rotation.
            %                   'deform'      - Deformations.
            %                 Default is 'linear'.
            %
            % DOTDIRECTION  - The direction in radians in which a fraction
            %                 DOTCOHERENCE of all dots drift (0=rightwards,
            %                 pi/2=upwards; angle increases counterclockwise).
            %                 Default is 0.
            %                 For TYPE 'rotExpContr' and 'deform', DOTDIRECTION
            %                 is the direction of the motion gradient (see
            %                 above).
            %
            % DOTSPEED      - The speed in pixels/frame at which a fraction
            %                 DOTCOHERENCE of all dots drift. Default is 1.
            %
            % DOTDENSITY    - The density of the dots (in the range [0,1]). This
            %                 will create roughly DOTDENSITY*imgWidth*imgHeight
            %                 dots. Default is 0.1.
            %
            % DOTCOHERENCE  - The fraction of dots that drift coherently in a
            %                 given direction of motion or motion gradient. The
            %                 remaining fraction moves randomly. Default is 1.
            %
            % DOTRADIUS     - The radius of the dots. If dotRadius<0, the dots
            %                 will be single pixels. If dotRadius>0, the dots
            %                 will be Gaussian blobs with sigma = 0.5*DOTRADIUS.
            %                 Default is -1.
            %
            % PTCENTER      - Only important when using motion gradients. Center
            %                 point of expansion, contraction, or rotation
            %                 (depending on stimulus TYPE). Default is [width/2
            %                 height/2].
            %
            % DENSITYSTYLE  - The number of dots is calculated by multiplying
            %                 the DOTDENSITY by the size of the image window. If
            %                 the dots are placed randomly, some dots will
            %                 overlap, so the actual dot density will be lower
            %                 than DOTDENSITY. If, however, the dots are placed
            %                 exactly so that they never overlap, all the dots
            %                 will move in register.
            %                 Both of these are problems that only occur when
            %                 using pixel dots rather than Gaussian dots.
            %                 DENSITYSTYLE can be either 'random' (for the first
            %                 problem) or 'exact' (for the second problem).
            %                 Default is 'random'.
            %
            % SAMPLEFACTOR  - Only important when using Gaussian dots. The dots
            %                 are made by calculating one large Gaussian dot,
            %                 and interpolating from that dot to make smaller
            %                 dots. This parameter specifies how much larger the
            %                 large dot is than the smaller dots (a factor).
            %                 Default is 10.
            %
            % INTERPMETHOD  - Only important when using Gaussian dots. This
            %                 parameter specifies the interpolation method used
            %                 in creating smaller dots. Choices: 'linear',
            %                 'cubic', 'nearest', 'v4'. Default is 'linear'.
            if nargin<12,interpMethod='linear';end
            if nargin<11,sampleFactor=10;end
            if nargin<10,densityStyle='random';end
            if nargin<9,ptCenter=[this.width/2 this.height/2];end
            if nargin<8,dotRadius=-1;end
            if nargin<7,dotCoherence=1;end
            if nargin<6,dotDensity=0.1;end
            if nargin<5,dotSpeed=1;end
            if nargin<4,dotDirection=0;end
            if nargin<3,type='linear';end
            if nargin<2,length=10;end
            
            % parse numeric input arguments (string arguments are easier to
            % catch in switch statements)
            errorStr=' must be a single numeric value';
            if ~isnumeric(length) || numel(length)>1 || length<=0
                error(['Length' errorStr ' > 0'])
            end
            if ~isnumeric(dotDirection) || numel(dotDirection)>1 ...
                    || dotDirection<0 || dotDirection>=2*pi
                error(['Dot direction' errorStr ' in the range [0,2*pi)'])
            end
            if ~isnumeric(dotSpeed) || numel(dotSpeed)>1 || dotSpeed<=0
                error(['Dot speed' errorStr ' > 0'])
            end
            if ~isnumeric(dotDensity) || numel(dotDensity)>1 ...
                    || dotDensity<=0 || dotDensity>1
                error(['Dot density' errorStr ' in the range [0,1]'])
            end
            if ~isnumeric(dotCoherence) || numel(dotCoherence)>1 ...
                    || dotCoherence<0 || dotCoherence>1
                error(['Dot coherence' errorStr ' in the range [0,1]'])
            end
            if ~isnumeric(dotRadius) || numel(dotRadius)>1
                error(['Dot radius' errorStr])
            end
            if ~isnumeric(ptCenter) || ~isvector(ptCenter) ...
                    || numel(ptCenter) ~=2 || ptCenter(1)<1 ...
                    || ptCenter(1)>this.width || ptCenter(2)<1 ...
                    || ptCenter(2)>this.height
                error(['Center point must be a 2-element vector whose ' ...
                    'elemnts are in the range [1:width, 1:height]'])
            end
            if ~isnumeric(sampleFactor) || numel(sampleFactor)>1 ...
                    || sampleFactor<=0
                error(['Sample factor' errorStr ' > 0'])
            end
        
            % if this object was created by loading a file, read the header
            % (loadHeaderOnly=true) to get stimulus dimensions
            this.privLoadStimIfNecessary(true);
            
            % use S&H script
            res = this.privMakeDots(length, type, dotDirection, dotSpeed, ...
                dotDensity, dotCoherence, dotRadius, ptCenter, densityStyle, ...
                sampleFactor, interpMethod);
            
            % append to existing frames
            this.privAppendFrames(res);
            
            % update default save file name
            switch lower(type)
                case 'linear'
                    svStr='RdkDrift';
                case 'rotexpcontr'
                    dirStr = {'Expand','RotCCW','Contract','RotCW'};
                    dominantDir = mod(round(dotDirection/(pi/2)),4);
                    svStr = ['Rdk' dirStr{dominantDir+1}];
                    if dominantDir ~= dotDirection/(pi/2)
                        svStr = [svStr 'Mixed'];
                    end
                case 'deform'
                    svStr='RdkDeform';
                otherwise
                    error(['Unknown stimulus type "' type '"'])
            end
            lastName = this.defaultSaveName(max(1,end-numel(svStr)+1):end);
            if ~strcmp(lastName,svStr) % only save distinc names
                this.defaultSaveName = cat(2,this.defaultSaveName,svStr);
            end
        end
        
        function addNoiseFrames(this, length, grayVal, type, varargin)
            % IS.addNoiseFrames(length, grayVal, type, varargin) adds a total
            % of LENGTH frames of a noise TYPE to the existing stimulus. The
            % background will have mean grayscale value GRAYVAL. TYPE is a
            % string that specifies a type of noise (generated using Matlab
            % function IMNOISE). Depending on the noise types, additional
            % arguments may be required (VARARGIN).
            %
            % LENGTH     - The number of noise frames to append. Default is 1.
            %
            % GRAYVAL    - The mean grayscale value of the background. Must be
            %              in the range [0,255]. Default is 128.
            %
            % TYPE       - A string that specifies any of the following noise
            %              types.
            %              'gaussian'      - Gaussian with constant mean (1st
            %                                additional argument) and variance
            %                                (2nd additional argument). Default
            %                                is zero mean noise with 0.01
            %                                variance.
            %              'localvar'      - Zero-mean, Gaussian white noise of
            %                                a certain local variance (1st
            %                                additional argument; must be of
            %                                size width-by-height).
            %              'poisson'       - Generates Poisson noise from the
            %                                data instead of adding additional
            %                                noise.
            %              'salt & pepper' - Salt and pepper noise of a certain
            %                                noise density d (1st additional
            %                                argument). Default is 0.05.
            %              'speckle'       - Multiplicative noise, using the
            %                                equation J=I+n*I, where n is
            %                                uniformly distributed random noise
            %                                with zero mean and variance v (1st
            %                                additional argument). Default for v
            %                                is 0.04.
            %
            % VARARGIN   - Additional arguments as required by the noise types
            %              described above.
            if nargin<4,type='gaussian';end
            if nargin<3,grayVal=128;end
            if nargin<2,length=1;end
            
            if ~isnumeric(length) || numel(length)~=1
                error('Length must be numeric')
            end
            if ~isnumeric(grayVal) || grayVal<0 || grayVal>255
                error('Mean grayscale value must be in the range [0,255]')
            end
            
            % scale grayscale value to [0,1]
            grayVal = grayVal / 255;
            
            % base mean value
            blank = ones(this.width, this.height)*grayVal;
            
            img=[];
            for i=1:length
                % for each frame, add noise
                noisy = this.privAddNoiseToFrame(blank,type,varargin{:});
                
                img = cat(3, img, noisy);
            end
            
            % add frames to existing stimulus
            this.privAppendFrames(img);
        end

        function addNoiseToExistingFrames(this, frames, type, varargin)
            % IS.addNoiseToExistingFrames(frames, type, varargin) adds noise of
            % a given type to all specified frames. FRAMES is a vector of frame
            % numbers. TYPE is a string that specifies any of the following
            % types of noise: 'gaussian', 'localvar', 'poisson', 'salt &
            % pepper', 'speckle'.
            %
            % FRAMES     - A vector of frame numbers to which noise shall be
            %              added. Default is all frames.
            %
            % TYPE       - A string that specifies any of the following noise
            %              types. Default is 'gaussian'.
            %              'gaussian'      - Gaussian with constant mean (1st
            %                                additional argument) and variance
            %                                (2nd additional argument). Default
            %                                is zero mean noise with 0.01
            %                                variance.
            %              'localvar'      - Zero-mean, Gaussian white noise of
            %                                a certain local variance (1st
            %                                additional argument; must be of
            %                                size width-by-height).
            %              'poisson'       - Generates Poisson noise from the
            %                                data instead of adding additional
            %                                noise.
            %              'salt & pepper' - Salt and pepper noise of a certain
            %                                noise density d (1st additional
            %                                argument). Default is 0.05.
            %              'speckle'       - Multiplicative noise, using the
            %                                equation J=I+n*I, where n is
            %                                uniformly distributed random noise
            %                                with zero mean and variance v (1st
            %                                additional argument). Default for v
            %                                is 0.04.
            %
            % VARARGIN   - Additional arguments as required by the noise types
            %              described above.
            if nargin<3,type='gaussian';end
            if nargin<2,frames=1:this.length;end
            if ~isvector(frames)
                error('Input argument FRAMES must be a vector')
            end
            
            for i=1:numel(frames)
                % extract specified frame
                img = this.getFrames(frames(i));
                
                % add noise
                noisy = this.privAddNoiseToFrame(img,type,varargin{:});
                
                % store frame
                this.privSetFrame(frames(i),noisy);
            end
        end
        
        function addPlaid(this, length, plaidDir, gratFreq, plaidAngle, ...
                plaidContrast)
            % IS.addPlaid(length, dir, freq, angle, contr, append) will add a
            % drifting plaid stimulus with mean intensity value 128 and a
            % contrast of value CONTR to your existing stimulus object.
            % The plaid stimulus is made of two sinusoidal gratings ("grating
            % components") separated by a specified ANGLE.
            %
            % This method is based on a script by Timothy Saint and Eero P.
            % Simoncelli at NYU. It was adapted for coordinate system [x,y,t].
            % For more information see beginning of this file.
            %
            % LENGTH        - The number of frames to create. Default is 10.
            %
            % PLAIDDIR      - The drifting direction of the stimulus in radians
            %                 (0=rightwards, pi/2=upwards; angle increases
            %                 counterclockwise). Default is 0.
            %
            % GRATFREQ      - 2-D vector of stimulus frequency for the grating
            %                 components. The first vector element is the
            %                 spatial frequency (cycles/pixels), whereas the
            %                 second vector element is the temporal frequency
            %                 (cycles/frame). Default is [0.1 0.1].
            %
            % PLAIDANGLE    - The angle between the grating components in
            %                 radians. Default is (2/3)*pi = 120 degrees.
            %
            % PLAIDCONTRAST - The grating contrast. Default is 1.
            if nargin<6,plaidContrast=1;end
            if nargin<5,plaidAngle=(2/3)*pi;end
            if nargin<4,gratFreq=[0.1 0.1];end
            if nargin<3,plaidDir=0;end
            if nargin<2,length=10;end
            
            if length<=0,error('Length must be positive'),end
            if plaidDir<0 || plaidDir>=2*pi
                error('Drifting direction must be in the range [0,2*pi)')
            end
            if numel(gratFreq)~=2
                error('Frequency must be [spatFreq tempFreq]')
            elseif gratFreq(1)<0 || gratFreq(2)<0
                error('Spatial and temporal frequency cannot be negative')
            end
            if plaidAngle<0 || plaidAngle>=2*pi
                error('Plaid angle must be in the range [0,2*pi)')
            end
            if plaidContrast<=0 || plaidContrast>1
                error('Contrast must be in the range (0,1]')
            end
            
            % if this object was created by loading a file, read the header
            % (loadHeaderOnly=true) to get stimulus dimensions
            this.privLoadStimIfNecessary(true);
            
            % use S&H script
            res = this.privMakePlaid(length, plaidDir, gratFreq, plaidAngle, ...
                plaidContrast);
            
            % append to existing frames
            this.privAppendFrames(res);
            
            % update default save file name
            lastName = this.defaultSaveName(max(1,end-4):end);
            if ~strcmp(lastName,'Plaid') % only save distinct names
                this.defaultSaveName = cat(2,this.defaultSaveName,'Plaid');
            end
        end
        
        function addSinGrating(this, length, sinDir, sinFreq, sinContrast, ...
                sinPhase)
            % IS.addSinGrating(length, sinDir, sinFreq, sinContrast, sinPhase)
            % will add a drifting sinusoidal grating with mean intensity value
            % 128 and a contrast of sinContrast to the existing stimulus.
            %
            % LENGTH       - The number of frames to create. Default is 10.
            %
            % DIR          - The drifting direction of the stimulus in radians
            %                (where 0=rightward). Default is 0.
            %
            % FREQ         - 2-D vector of stimulus frequency. The first
            %                component is the spatial frequency (cycles/pixels),
            %                whereas the second component is the temporal
            %                frequency (cycles/frame). Default is [0.1 0.1].
            %
            % CONTRAST     - The grating contrast. Default is 1.
            %
            % PHASE        - The initial phase of the grating in periods.
            %                Default is 0.
            %
            % This method is based on a script by Timothy Saint and Eero P.
            % Simoncelli at NYU. It was adapted for coordinate system [x,y,t].
            % For more information see beginning of this file.
            %
            if nargin<6,sinPhase=0;end
            if nargin<5,sinContrast=1;end
            if nargin<4,sinFreq=[0.1 0.1];end
            if nargin<3,sinDir=0;end
            if nargin<2,length=10;end
            
            if length<=0,error('Length must be positive'),end
            if sinDir<0 || sinDir>=2*pi
                error('Drifting direction must be in the range [0,2*pi)')
            end
            if numel(sinFreq)~=2
                error('Frequency must be [spatFreq tempFreq]')
            elseif sinFreq(1)<0 || sinFreq(2)<0
                error('Spatial and temporal frequency cannot be negative')
            end
            if sinContrast<=0 || sinContrast>1
                error('Contrast must be in the range (0,1]')
            end
            if sinPhase<0,error('The initial phase cannot be negative'),end
            
            % if this object was created by loading a file, read the header
            % (loadHeaderOnly=true) to get stimulus dimensions
            this.privLoadStimIfNecessary(true);
            
            % use S&H script
            res = this.privMakeSin(length, sinDir, sinFreq, sinContrast, sinPhase);
            
            % append to existing frames
            this.privAppendFrames(res);
            
            % update default save file name
            lastName = this.defaultSaveName(max(1,end-6):end);
            if ~strcmp(lastName,'Grating') % only save distinc names
                this.defaultSaveName = cat(2,this.defaultSaveName,'Grating');
            end
        end
        
        function clear(this)
            % IS.clear() clears all added frames. If the stimulus was loaded
            % from file, the object will be reset to the version from file.
            this.length = 0;
            this.stim = [];
            
            % if object was loaded from file, need to load again
            if ~isempty(this.loadFile)
                this.needToLoad = true;
            end
        end
        
        
        function displayFrames(this,frames,dispFrameNr)
            % img = IS.displayFrames(frames) displays the specified frames in
            % the current figure/axes.
            %
            % FRAMES       - A list of frame numbers. For example, requesting
            %                frames=[1 2 8] will return the first, second, and
            %                eighth frame in a width-by-height-by-3 matrix.
            %                Default: display all frames.
            % DISPFRAMENR  - A boolean flag that indicates whether to display
            %                the frame number. Default: true.
            this.privLoadStimIfNecessary(); % need to load stim first
            if nargin<3,dispFrameNr = true;end
            if nargin<2,frames = 1:this.length;end
            
            % reset abort flag, set up callback for key press events
            this.plotAbortPlotting = false;
            set(gcf,'KeyPressFcn',@this.privPauseOnKeyPressCallback)
                        
            % display frame in specified axes
            for i=frames
                if this.plotAbortPlotting
                    % user pressed button to quit plotting
                    this.plotAbortPlotting = false;
                    close;
                    return
                end
                
                colormap gray
                imagesc(permute(this.stim(:,:,i),[2 1 3]),[0 1])
                if dispFrameNr
                    text(2,this.height-1,num2str(i), ...
                        'FontSize',10,'BackgroundColor','white')
%                     text(2,this.height-1,num2str(find(frames==i)), ...
%                         'FontSize',10,'BackgroundColor','white')
                end
                axis image
                drawnow
                pause(0.1)
            end
        end
        
        function img = getFrames(this,frames)
            % img = IS.getFrames(frames) returns the specified frames of the
            % stimulus as a 3-D width-by-height-by-length(frames) matrix, where
            % the third argument is the length of the frames list.
            %
            % FRAMES    - A list of frame numbers. For example, requesting
            %             frames=[1 2 8] will return the first, second, and
            %             eighth frame in a width-by-height-by-3 matrix.
            %             Default: return all frames.
            if nargin<2,frames=1:this.length;end
            if ~isnumeric(frames) || ~isvector(frames)
                error('Must specify a vector of frames')
            end
            if sum(frames>this.length)>0 || sum(frames<1)>0
                error(['Specified frames must be in the range [1,' ...
                    num2str(this.length) ']'])
            end
            
            this.privLoadStimIfNecessary(); % need to load stim first
            
            % return frames
            img = this.stim(:,:,frames);
        end
		
        function recordMovie(this, fileName, frames, fps, winSize, bgColor)
            % IS.recordMovie(movieFile, frames, fps, winSize, bgColor) takes an
            % AVI movie of a list of frames using the VIDEOWRITER utility.
            %
            % FILENAME  - A string enclosed in single quotation marks that
            %             specifies the name of the file to create.
            %             Default: 'movie.avi'.
            % FRAMES    - A list of frame numbers. For example, requesting
            %             frames=[1 2 8] will return the first, second, and
            %             eighth frame in a width-by-height-by-3 matrix.
            %             Default: return all frames.
            % FPS       - Rate of playback for the video in frames per second.
            %             Default: 10.
            % WINSIZE   - A 2-element vector specifying the window size of the
            %             video as width x height in pixels. Set to [0 0] in
            %             order to automatically make the movie window fit to
            %             the size of the plot window. Default: [0 0].
            % BGCOLOR   - The background color of the video. Must be of type 
            %             ColorSpec (char such as 'w','b','k' or a 3-element 
            %             vector for RGB channels). Default: 'w'.
            if nargin<6,bgColor='w';end
            if nargin<5,winSize=[0 0];end
            if nargin<4,fps=10;end
            if nargin<3,frames=1:this.length;end
            if nargin<2,fileName='movie.avi';end
            
            if ~isnumeric(frames) || ~isvector(frames)
                error('Must specify a vector of frames')
            end
            if sum(frames>this.length)>0 || sum(frames<1)>0
                error(['Specified frames must be in the range [1,' ...
                    num2str(this.length) ']'])
            end
            if ~isnumeric(fps) || fps<=0
                error('Invalid frame rate. fps must be an integer > 0.')
            end
            
            set(gcf,'color',bgColor);
            if sum(winSize>0)==2
                set(gcf,'Position',[100 100 winSize]);
            end
            set(gcf,'PaperPositionMode','auto');
            
            % open video object
            vidObj = VideoWriter(fileName);
            vidObj.Quality = 100;
            vidObj.FrameRate = fps;
            open(vidObj);
            
            % display and record all frames
            for i=frames
                displayFrames(this,i);
                writeVideo(vidObj, getframe(gcf));
            end
            close(gcf)
            
            close(vidObj);
            disp(['created file "' fileName '"'])
        end

        function saveToFile(this, fileName)
            % IS.saveToFile(fileName) saves a InputStimulus object to fileName.
            % Later the stimulus can be loaded by creating a new object such as
            % IS = InputStimulus(fileName).
            %
            % FILENAME       - The name of the file to create (optional).
            %                  Default is to use a name consisting of a stimulus
            %                  description and the stimulus dimensions (to be
            %                  created in the current directory).
            %                  Make sure to have write access to the specified
            %                  file, and that the directory exists.
            if nargin<2
                fileName = [this.defaultSaveName '_' lower(this.mode) ...
                    '_' num2str(this.width) ...
                    'x' num2str(this.height) 'x' num2str(this.length) '.dat'];
            end
            fid=fopen(fileName,'w');
            if fid==-1
                error(['Could not open "' fileName '" with write permission.']);
            end
            if this.width<0 || this.height<0 || this.length<0
                error('Stimulus width/height/length not set.');
            end
            
            % check whether fwrite is successful
            wrErr = false;
            
            % start with file signature
            sign=this.fileSignature; % some random number
            cnt=fwrite(fid,sign,'int');           wrErr = wrErr | (cnt~=1);
            
            % include version number
            cnt=fwrite(fid,1.0,'float');          wrErr = wrErr | (cnt~=1);
            
            % include number of channels (1 for GRAY, 3 for RGB)
            cnt=fwrite(fid,this.channels,'int8'); wrErr = wrErr | (cnt~=1);
            
            % specify width, height, length
            cnt=fwrite(fid,this.width,'int');     wrErr = wrErr | (cnt~=1);
            cnt=fwrite(fid,this.height,'int');    wrErr = wrErr | (cnt~=1);
            cnt=fwrite(fid,this.length,'int');    wrErr = wrErr | (cnt~=1);
            
            % read stimulus
            cnt=fwrite(fid,this.stim*255,'uchar');
            wrErr = wrErr | (cnt~=this.width*this.height*this.length*this.channels);
            
            % if there has been an error along the way, inform user
            if wrErr
                error(['Error during writing to file "' fileName '"'])
            end
            
            fclose(fid);
            disp(['Successfully saved stimulus to file "' fileName '"'])
        end
    end
    
    %% PRIVATE METHODS
    methods (Hidden, Access = private)
        function noisy = privAddNoiseToFrame(this,frame,type,varargin)
            % Private method to add noise of a specific type to a single
            % stimulus frame
            if ~isnumeric(frame) || size(frame,3)~=1
                error('Cannot act on more than one frame at once')
            end
            if min(frame(:))<0 || max(frame(:))>1
                error('Grayscale values must be in the range [0,1]')
            end
            
            switch lower(type)
                case 'gaussian'
                    valMean = 0;
                    valVar = 0.01;
                    if numel(varargin)>=1
                        valMean=varargin{1};
                        if ~isnumeric(valMean)
                            error('Mean for Gaussian noise must be numeric')
                        end
                    end
                    if numel(varargin)>=2
                        valVar =varargin{2};
                        if ~isnumeric(valVar)
                            error('Variance for Gaussian noise must be numeric')
                        end
                    end
                    noisy = imnoise(frame,'gaussian',valMean,valVar);
                case 'localvar'
                    if numel(varargin)<1
                        error('Must specify local variance of image')
                    end
                    if size(varargin{1}) ~= size(frame)
                        error('Local variance must have same size as image')
                    end
                    noisy = imnoise(frame,'localvar',varargin{1});
                case 'poisson'
                    noisy = imnoise(frame,'poisson');
                case 'salt & pepper'
                    valD = 0.05;
                    if numel(varargin)>=1
                        valD=varargin{1};
                        if ~isnumeric(valD)
                            error(['Noise density for Salt & Pepper noise ' ...
                                'must be numeric'])
                        end
                    end
                    noisy = imnoise(frame,'salt & pepper',valD);
                case 'speckle'
                    valVar = 0.04;
                    if numel(varargin)>=1
                        valVar=varargin{1};
                        if ~isnumeric(valVar)
                            error('Variance for Speckle noise must be numeric')
                        end
                    end
                    noisy = imnoise(frame,'speckle',valVar);
                otherwise
                    error(['Unknown noise type "' type '". Currently ' ...
                        'supported are: ' this.supportedNoiseTypes])
            end
            
            % confine to [0,1]
            noisy = max(0, min(1,noisy));
        end
        
        function privAppendFrames(this,frames)
            % Private method to append a series of frames to the existing
            % stimulus.
            this.privLoadStimIfNecessary(); % need to load stim first
            this.stim = cat(3, this.stim, frames);
            
            % update attributes
            this.needToLoad = false; % stim is up-to-date
            this.length = size(this.stim,3);
        end
        
        function privLoadFromFile(this, fileName, loadHeaderOnly)
            % Private method to load a InputStimulus object from fileName.
            % This file must have been created using method
            % InputStimulus.saveToFile.
            % Make sure to have read access to the specified file.
            %
            % FILENAME       - relative or absolute path to a binary file
            %                  containing a InputStimulus object.
            % LOADHEADERONLY - A flag to indicate whether only the header should
            %                  be read. This is helpful if one only cares about
            %                  the stimulus dimensions and such. Default: false.
            if nargin<3,loadHeaderOnly=false;end
            fid=fopen(fileName,'r');
            if fid==-1
                error(['Could not open "' fileName '" with read permission.']);
            end
            
            % read signature
            sign = fread(fid,1,'int');
            if sign~=this.fileSignature
                error('Unknown file type')
            end
            
            % read version number
            version = fread(fid,1,'float');
            if (version ~= 1.0)
                error(['Unknown file version, must have Version 1.0 (Version ' ...
                    num2str(version) ' found)'])
            end
            
            % read number of channels
            this.channels = fread(fid,1,'int8');
            
            % read stimulus dimensions
            this.width = fread(fid,1,'int');
            this.height = fread(fid,1,'int');
            this.length = fread(fid,1,'int');
            
            % don't read stimulus if this flag is set
            if loadHeaderOnly
                return
            end
            
            % read stimulus
            this.stim = fread(fid,'uchar')/255;
            
            % make sure dimensions match up
            dim = this.width*this.height*this.length*this.channels;
            if size(this.stim,1) ~= dim
                error(['Error during reading of file "' fileName '". ' ...
                    'Expected width*height*length = ' ...
                    num2str(this.privSizeOf()) ...
                    'elements, found ' num2str(numel(this.stim))])
            end
            
            % reshape accordingly
            if this.channels==1
                % grayscale
                this.mode = 'gray';
                this.stim = reshape(this.stim, this.width, this.height, ...
                    this.length);
            else
                % RGB
                this.mode = 'rgb';
                error('Not yet supported')
            end
            disp(['Successfully loaded stimulus from file "' fileName '"'])
        end
        
        function privLoadStimIfNecessary(this, loadHeaderOnly)
            % Private method to load the stimulus file if necessary (which is
            % determined by the flag needToLoad)
            if nargin<2,loadHeaderOnly=false;end
            if this.needToLoad
                this.privLoadFromFile(this.loadFile,loadHeaderOnly);
                
                % if loaded the stimulus, set the following flag to false
                % but don't if only header was read
                if ~loadHeaderOnly
                    this.needToLoad = false;
                end
            end
        end
        
        function res = privMakeBar(this, length, dir, speed, barWidth, ...
                edgeWidth, pxBtwBars, barStartingPos)
            % Private method to create a drifting bar stimulus.
            %
            % This method was initially authored by Timothy Saint and Eero
            % P. Simoncelli at NYU. It was adapted for coordinate system
            % [x,y,t]. For more information see beginning of this file.
            %
            
            % spatial and temporal frequencies
            gratingSf = 1/pxBtwBars;
            gratingTf = gratingSf * speed;
            
            % The plan of action: make a drifting grating with the right
            % velocity and orientation. The period of the sin wave will be the
            % same as the distance between the bars that the user wants. Then we
            % make bars out of the peaks of the sin wave.
            
            % Make the grating
            gratingPhase = barStartingPos * gratingSf;
            res = 2*this.privMakeSin(length, dir, ...
                [gratingSf gratingTf], 1, gratingPhase) -1;
            
            % Find the thresholds
            barInnerThreshold = cos(2*pi*gratingSf*barWidth/2);
            barOuterThreshold = cos(2*pi*gratingSf*(barWidth/2 + edgeWidth));
            
            % There are three regions: where the stimulus should be one (the
            % centers of the bars), where it should be zero (outside the bars),
            % and where it should be somehwere in between (edges of the bars).
            % Find them
            wOne = res >= barInnerThreshold;
            wEdge = (res < barInnerThreshold) ...
                & (res > barOuterThreshold);
            wZero = res <= barOuterThreshold;
            
            % Set the regions to the appropriate level
            res(wOne) = 1;
            res(wZero) = 0;
            
            % adjust range to [0,2*pi)
            res(wEdge) = acos(res(wEdge));
            
            % adjust range to [0,1] spatial period
            res(wEdge) = res(wEdge)/(2*pi*gratingSf);
            res(wEdge) =  (pi/2)*(res(wEdge) ...
                                 - barWidth/2)/(edgeWidth);
            res(wEdge) = cos(res(wEdge));
        end
        
        function res = privMakeDots(this, length, type, dotDirection, ...
                dotSpeed, dotDensity, dotCoherence, dotRadius, ...
                ptCenter, densityStyle, sampleFactor, interpMethod)
            % Private method to create an RDK stimulus of a given TYPE.
            %
            % This method was initially authored by Timothy Saint and Eero
            % P. Simoncelli at NYU. It was adapted for coordinate system
            % [x,y,t]. For more information see beginning of this file.
            stimSz = [this.width this.height length];
            if dotDensity<0
                if dotRadius<0
                    dotDensity = 0.1;
                else
                    dotDensity = 0.3/(pi*dotRadius^2);
                end
            end
            
            % adjust drift direction to make up for flipped y-axis in imagesc
            dotDirection = mod(2*pi-dotDirection,2*pi);
            
            % make the large dot for sampling
            xLargeDot = linspace(-(3/2)*dotRadius, (3/2)*dotRadius, ...
                ceil(3*dotRadius*sampleFactor));
            sigmaLargeDot = dotRadius/2;
            largeDot = exp(-xLargeDot.^2./(2.*sigmaLargeDot^2));
            largeDot = largeDot'*largeDot;
            [xLargeDot, yLargeDot] = meshgrid(xLargeDot, xLargeDot);
            
            % There is a buffer area around the image so that we don't have to
            % worry about getting wraparounds exactly right.  This buffer is
            % twice the size of a dot diameter.
            if dotRadius > 0
                bufferSize = 2*ceil(dotRadius*3);
            else
                bufferSize = 5;
            end
            
            % the 'frame' is the field across which the dots drift. The final
            % output of this this function will consist of 'snapshots' of this
            % frame without buffer. We store the size of the frame and a
            % coordinate system for it here:
            frameSzX = stimSz(1) + 2.*bufferSize;
            frameSzY = stimSz(2) + 2.*bufferSize;
            [xFrame, yFrame] = meshgrid([1:frameSzX], [1:frameSzY]);
            
            % adjust center point for frameSz
            frameScaling = [frameSzX frameSzY]./[this.width this.height];
            ptCenter = ptCenter.*frameScaling;
                        
            % nDots is the number of coherently moving dots in the stimulus.
            % nDotsNonCoherent is the number of noncoherently moving dots.
            nDots = round(dotCoherence.*dotDensity.*numel(xFrame));
            nDotsNonCoherent = round((1-dotCoherence).*dotDensity.*numel(xFrame));
            
            % Set the initial dot positions.
            % dotPositions is a matrix of positions of the coherently moving
            % dots in [x,y] coordinates; each row in dotPositions stores the
            % position of one dot
            if strcmp(densityStyle, 'exact')
                z = zeros(numel(xFrame), 1);
                z(1:nDots) = 1;
                ord = rand(size(z));
                [~, ord] = sort(ord);
                z = z(ord);
                dotPositions = [xFrame(z==1), yFrame(z==1)];
            else
                dotPositions = rand(nDots, 2) * [frameSzX 0; 0 frameSzY];
            end
            
            % s will store the output. After looping over each frame, we will
            % trim away the buffer from s to obtain the final result.
            res = zeros(frameSzX, frameSzY, stimSz(3));
            toInterpolate = -floor((3/2)*dotRadius):floor((3/2)*dotRadius);
            dSz = floor((3/2)*dotRadius);

            % dxdt for rotations/expansions/contractions and deformations
            expand = @(x,y,w0,th) ([w0*(x.*cos(th)-y.*sin(th)), ...
                w0*(x.*sin(th)+y.*cos(th))]);
            deform = @(x,y,w0,th) ([w0*(x.*cos(th)+y.*sin(th)), ...
                w0*(x.*sin(th)-y.*cos(th))]);

            for t = 1:stimSz(3)
                % update the dot positions according to RDK type
                switch lower(type)
                    case 'linear'
                        dotVelocity = [cos(dotDirection), ...
                                       sin(dotDirection)];
                        dotVelocity = dotVelocity*dotSpeed;
                        dotPositions = dotPositions + repmat(dotVelocity, ...
                            size(dotPositions, 1), 1);
                        
                        % wrap around for all dots that have gone past the image
                        % borders
                        w = find(dotPositions(:,1)>frameSzX+.5);
                        dotPositions(w,1) = dotPositions(w,1) - frameSzX;
                        
                        w = find(dotPositions(:,1)<.5);
                        dotPositions(w,1) = dotPositions(w,1) + frameSzX;
                        
                        w = find(dotPositions(:,2)>frameSzY+.5);
                        dotPositions(w,2) = dotPositions(w,2) - frameSzY;
                        
                        w = find(dotPositions(:,2)<.5);
                        dotPositions(w,2) = dotPositions(w,2) + frameSzY;
                    case 'rotexpcontr'
                        w0 = 0.05;
                        dxdt = expand(dotPositions(:,1)-ptCenter(1), ...
                            dotPositions(:,2)-ptCenter(2), w0, dotDirection);
                        dxdt = dxdt*dotSpeed;
                        dotPositions = dotPositions + dxdt;
                        
                        if abs(mod(2*pi-dotDirection,2*pi))<=pi/2
                            % replace dots that have gone past the image borders (a
                            % little more tricky than the linear case)
                            radiusFOE = abs(dotRadius)*5.*frameScaling;

                            % dominated by expansion
                            % if dots go past borders, replace near FOE
                            w = find(dotPositions(:,1)>frameSzX+0.5);
                            dotPositions(w,1) = ptCenter(1) ...
                                + (2*rand(numel(w),1)-1)*radiusFOE(1);
                            w = find(dotPositions(:,1)<0.5);
                            dotPositions(w,1) = ptCenter(1) ...
                                + (2*rand(numel(w),1)-1)*radiusFOE(1);
                            
                            w = find(dotPositions(:,2)>frameSzY+0.5);
                            dotPositions(w,2) = ptCenter(2) ...
                                + (2*rand(numel(w),1)-1)*radiusFOE(2);
                            w = find(dotPositions(:,2)<0.5);
                            dotPositions(w,2) = ptCenter(2) ...
                                + (2*rand(numel(w),1)-1)*radiusFOE(2);
                        elseif abs(mod(2*pi-dotDirection,2*pi)-pi)<=pi/2
                            % dominated by contraction
                            % the trickiest: if points get close to FOE, move
                            % them out to the borders
                            
                            dist2 = (dotPositions(:,1)-ptCenter(1)).^2 ...
                                + (dotPositions(:,2)-ptCenter(2)).^2;
                            radiusFOE = abs(dotRadius)*2*max(frameScaling);
                            
                            % TODO: it is kinda tricky to make the points not
                            % cluster near the FOE...
                            prob = 1; % min(0.5,sum(dist2<radiusFOE^2)/radiusFOE^2)*2;
                            w = find(rand<prob & dist2<radiusFOE^2);
                            
                            % new position: somewhere in the outer quarter of
                            % the frame
                            theta = rand(numel(w),1)*2*pi;
                            noise = rand(numel(w),1); % prevent syncing up
                            dotPositions(w,1) = (cos(theta)+1)/2.*(frameSzX-noise/4);
                            dotPositions(w,2) = (sin(theta)+1)/2.*(frameSzY-noise/4);
                        end
                    case 'deform'
                        w0 = 0.05;
                        dxdt = deform(dotPositions(:,1)-ptCenter(1), ...
                            dotPositions(:,2)-ptCenter(2), w0, dotDirection);
                        dxdt = dxdt*dotSpeed;
                        dotPositions = dotPositions + dxdt;

                        w = find(dotPositions(:,1)>frameSzX+.5);
                        dotPositions(w,1) = dotPositions(w,1) - frameSzX;
                        
                        w = find(dotPositions(:,1)<.5);
                        dotPositions(w,1) = dotPositions(w,1) + frameSzX;
                        
                        w = find(dotPositions(:,2)>frameSzY+.5);
                        dotPositions(w,2) = dotPositions(w,2) - frameSzY;
                        
                        w = find(dotPositions(:,2)<.5);
                        dotPositions(w,2) = dotPositions(w,2) + frameSzY;

                    otherwise
                        error(['Unknown RDK type "' type '"'])
                end
                
                
                % add noncoherent dots and make a vector of dot positions for
                % this frame only
                dotPositionsNonCoherent = rand(nDotsNonCoherent, 2) ...
                    * [frameSzX-1 0; 0 frameSzY-1] + .5;
                
                % create a temporary matrix of positions for dots to be shown in
                % this frame
                tmpDotPositions = [dotPositions; dotPositionsNonCoherent];
                
                % prepare a matrix of zeros for this frame
                thisFrame = zeros(size(xFrame));
                if dotRadius > 0
                    % in each frame, don't show dots near the edges of the
                    % frame. That's why we have a buffer. The reason we don't
                    % show them is that we don't want to deal with edge 
                    % handling
                    w1 = find(tmpDotPositions(:,1) > (frameSzX - bufferSize ...
                        + (3/2)*dotRadius));
                    w2 = find(tmpDotPositions(:,1) < (bufferSize - (3/2)*dotRadius));
                    w3 = find(tmpDotPositions(:,2) > (frameSzY - bufferSize ...
                        + (3/2)*dotRadius));
                    w4 = find(tmpDotPositions(:,2) < (bufferSize - (3/2)*dotRadius));
                    w = [w1; w2; w3; w4];
                    tmpDotPositions(w, :) = [];
                    
                    % add the dots to thisFrame
                    for p = 1:size(tmpDotPositions, 1)
                        % find the center point of the current dot, in thisFrame
                        % coordinates. This is where the dot will be placed.
                        cpX = round(tmpDotPositions(p, 1));
                        cpY = round(tmpDotPositions(p, 2));
                        
                        xToInterpol = toInterpolate ...
                            + (round(tmpDotPositions(p,1)) ...
                            - tmpDotPositions(p,1));
                        yToInterpol = toInterpolate ...
                            + (round(tmpDotPositions(p,2)) ...
                            - tmpDotPositions(p,2));
                        [xToInterpol, yToInterpol] = meshgrid(xToInterpol, ...
                            yToInterpol);
                        thisSmallDot = interp2(xLargeDot, yLargeDot, ...
                            largeDot, xToInterpol, yToInterpol, interpMethod);
                        
                        % now add this small dot to the frame.
                        thisFrame(cpX-dSz:cpX+dSz, cpY-dSz:cpY+dSz) = ...
                            thisFrame(cpX-dSz:cpX+dSz, cpY-dSz:cpY+dSz) + ...
                            thisSmallDot;
                        
                    end
                else
                    tmpDotPositions(tmpDotPositions(:,1) > frameSzX, :) = [];
                    tmpDotPositions(tmpDotPositions(:,1) < 1, :) = [];
                    tmpDotPositions(tmpDotPositions(:,2) > frameSzY, :) = [];
                    tmpDotPositions(tmpDotPositions(:,2) < 1, :) = [];
                    tmpDotPositions = round(tmpDotPositions);
                    
                    w = sub2ind(size(thisFrame), tmpDotPositions(:,1), ...
                        tmpDotPositions(:,2));
                    
                    thisFrame(w) = 1;
                end
                % Add this frame to the final output
                res(:,:,t) = thisFrame;
            end
            % Now trim away the buff
            res = res(bufferSize+1:end-bufferSize, ...
                bufferSize+1:end-bufferSize, :);
            res(res>1) = 1;
        end
        
        function res = privMakePlaid(this, length, dir, freq, angle, contr)
            % Private method to create a drifting plaid stimulus with mean
            % intensity value 128 and a contrast of contr. The plaid stimulus is
            % made of two sinusoidal gratings ("grating components") separated
            % by a specified angle.
            %
            % This method was initially authored by Timothy Saint and Eero
            % P. Simoncelli at NYU. It was adapted for coordinate system
            % [x,y,t]. For more information see beginning of this file.
            firstDirection = mod(dir + angle/2, 2*pi);
            secondDirection = mod(dir - angle/2, 2*pi);
            firstGrating = this.privMakeSin(length, firstDirection, freq, ...
                contr/2, 0);
            secondGrating = this.privMakeSin(length, secondDirection, freq, ...
                contr/2, 0);
            res = firstGrating + secondGrating - 0.5;
        end
        
        function res = privMakeSin(this, length, dir, freq, contr, phase)
            % A private method that creates a drifting sinusoidal grating with
            % mean intensity value 128 and a contrast of CONTR.
            %
            % This method was initially authored by Timothy Saint and Eero
            % P. Simoncelli at NYU. It was adapted for coordinate system
            % [x,y,t]. For more information see beginning of this file.
            stimSz = [this.width, this.height, length];
            if numel(freq)~=2
                error('Frequency must be [spatFreq tempFreq]')
            end
            sf = freq(1);  % spatial frequency
            tf = freq(2);  % temporal frequency
            
            % Make a coordinate system
            x = (1:stimSz(1)) - (floor(stimSz(1)/2)+1);
            y = (1:stimSz(2)) - (floor(stimSz(2)/2)+1);
            t = (0:stimSz(3)-1);
            [x, y, t] = ndgrid(x, y, t);
            
            % adjust drift direction to make up for flipped y-axis in imagesc
            dir = mod(2*pi-dir,2*pi);
            
            res = cos(2*pi*sf*cos(dir)*x + ...
                2*pi*sf*sin(dir)*y - ...
                2*pi*tf*t + ...
                2*pi*phase);
            
            res = contr.*res./2 + .5;
        end
        
        function privPauseOnKeyPressCallback(this,~,eventData)
            % Callback function to pause plotting
            switch eventData.Key
                case 'p'
                    disp('Paused. Press any key to continue.');
                    waitforbuttonpress;
                case 'q'
                    disp('Quitting.')
                    close % close figure
                    this.plotAbortPlotting = true;
            end
        end
        
        function privSetFrame(this,index,frame)
            % Private method to replace/insert a single frame
            if ~isnumeric(index) || size(frame,3)>1
                error('Cannot set more than one frame at the same time')
            end
            if size(frame,1) ~= this.width
                error(['Stimulus width (' num2str(this.width) ') does not ' ...
                    'match frame width (' num2str(size(frame,1)) ')'])
            end
            if size(frame,2) ~= this.height
                error(['Stimulus height (' num2str(this.height) ') does ' ...
                    'not match frame height (' num2str(size(frame,2)) ')'])
            end
            
            % replace existing frame or append
            this.stim(:,:,index) = frame;
            
            % update attributes
            this.needToLoad = false; % stim is up-to-date
            this.length = size(this.stim,3);
        end
        
        function len = privSizeOf(this)
            % returns the total number of elements (pixels) in a stimulus
            len = this.width*this.height*this.length*this.channels;
        end
    end
end