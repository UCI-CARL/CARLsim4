classdef InputStimulus < handle
    % IS = InputStimulus(varargin) creates a new instance of class
    % VisualStimulus.
    %
    % IS = VisualStimulus(width,height,mode) initializes an empty VisualStimulus
    % object of canvas size width*height and a given image mode. width and
    % height will be rounded to the nearest integer (number of pixels). Image
    % mode can be either 'gray' (grayscale) or 'rgb'.
    %
    % IS = VisualStimulus(width,height) initializes an empty VisualStimulus
    % object of canvas size width*height in grayscale mode.
    %
    % IS = VisualStimulus(fileName) loads a VisualStimulus object from file.
    % fileName must be a valid relative/absolute path to a binary file that has
    % been saved using method VisualStimulus.saveToFile.
    %
    % Version 3/31/14
    % Author: Michael Beyeler <mbeyeler@uci.edu>
    %
    % This class uses some scripts from an open-source MATLAB package of
    % Simoncelli & Heeger's Motion Energy model, obtained from
    % http://www.cns.nyu.edu/~lcv/MTmodel/ version 1.0 (10/14/05).
    % Authors: Timothy Saint <saint@cns.nyu.edu> and Eero P. Simoncelli
    % <eero.simoncelli@nyu.edu>
    
    %% PROPERTIES
    properties
    end
    
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
        fileSignature = 304698591;
    end
    
    
    %% PUBLIC METHODS
    methods
        % Matlab doesn't supper function overloading, so use varargin instead
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
            obj.supportedImageModes = {'gray','rgb'};
            
            % the following noise types are supported
            obj.supportedNoiseTypes = {'gaussian','poisson','salt & pepper',...
                'speckle'};
            
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
        
        function addNoiseToExistingFrames(this, frames, type, varargin)
            % IS.addNoiseToExistingFrames(frames, type, varargin) adds noise of
            % a given type to all specified frames. FRAMES is a vector of frame
            % numbers. TYPE is a string that specifies any of the following
            % types of noise: 'gaussian', 'localvar', 'poisson', 'salt &
            % pepper', 'speckle'.
            %
            % FRAMES     - A vector of frame numbers to which noise should be
            %              added.
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
            if ~isvector(frames)
                error('Input argument FRAMES must be a vector')
            end
            
            for i=1:numel(frames)
                % extract specified frame
                img = this.getFrames(frames(i));
                
                % add noise
                noisy = this.privAddNoiseToFrame(img,type,varargin);
                
                % store frame
                this.privSetFrame(frames(i),noisy);
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
            % LENGTH     - The number of noise frames to append
            %
            % GRAYVAL    - The mean grayscale value of the background. Must be
            %              in the range [0,255]. Default: 128.
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
            
            if ~isnumeric(length)
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
                noisy = this.privAddNoiseToFrame(blank,type,varargin);
                
                img = cat(3, img, noisy);
            end
            
            % add frames to existing stimulus
            this.privAppendFrames(img);
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
            % LENGTH        - The number of frames to create. Default is 50.
            %
            % PLAIDDIR      - The drifting direction of the stimulus in radians
            %                 (where 0=rightward). Default is 0.
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
            if nargin<2,length=50;end
            
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
        
        function addRdkDrift(this, length, dotDirection, dotSpeed, ...
                dotDensity, dotCoherence, dotRadius, densityStyle, ...
                sampleFactor, interpMethod)
            % This method was initially authored by Timothy Saint and Eero
            % P. Simoncelli at NYU. It was adapted for coordinate system
            % [x,y,t]. For more information see beginning of this file.
            if nargin<10,interpMethod='linear';end
            if nargin<9, sampleFactor=10;end
            if nargin<8, densityStyle='exact';end
            if nargin<7, dotRadius=-1;end
            if nargin<6, dotCoherence=1;end
            if nargin<5, dotDensity=-1;end
            if nargin<4, dotSpeed=1;end
            if nargin<3, dotDirection=0;end
            if nargin<2, length=20;end
            
            % if this object was created by loading a file, read the header
            % (loadHeaderOnly=true) to get stimulus dimensions
            this.privLoadStimIfNecessary(true);
            
            res = this.privMakeDots(length, dotDirection, dotSpeed, ...
                dotDensity, dotCoherence, dotRadius, densityStyle, ...
                sampleFactor, interpMethod);

            % append to existing frames
            this.privAppendFrames(res);
            
            % update default save file name
            lastName = this.defaultSaveName(max(1,end-7):end);
            if ~strcmp(lastName,'RdkDrift') % only save distinct names
                this.defaultSaveName = cat(2,this.defaultSaveName,'RdkDrift');
            end
        end
        
        function addRdkExpansion(this, length, FOE, density, speed)
            if nargin<5,speed=1;end
            if nargin<4,density=0.2;end
            expand = @(x,y,w0,th) ([w0*(x*cos(th)-y*sin(th)),w0*(x*sin(th)+y*cos(th))]);
            
            % if this object was created by loading a file, read the header
            % (loadHeaderOnly=true) to get stimulus dimensions
            this.privLoadStimIfNecessary(true);
            
            % get number of dots from density
            nDots = round(density*this.width*this.height);
            
            % starting points are random
            posDots = [randi(this.width, [1 nDots]);
                randi(this.height,[1 nDots])];
            
            w0 = 0.05; % good for expansion
            th = 0;
            fudge = 2; % fudge factor for small retina sizes
            
            res=[];
            for i=1:length
                % init retina
                retina = zeros(this.width, this.height);
                
                % adjust the center of the motion pattern to (x0,x0)
                % fudge factor used for small retina sizes
                dxdt = expand((posDots(1,:)'-FOE(1))*fudge, ...
                    (posDots(2,:)'-FOE(2))*fudge, ...
                    w0, th)';
                
                % scale dxdt with speed value
                dxdt = dxdt*speed;
                
                % update points
                posDots = posDots + dxdt;
                
                % if points disappear from retina, remove points and insert new
                % ones at random position
                [~,c] = find((posDots(1,:)>this.width) | (posDots(1,:)<1));
                posDots(:,c) = [randi(this.width, [1 numel(c)]);
                    randi(this.height,[1 numel(c)])];
                [~,c] = find((posDots(2,:)>this.height) | (posDots(2,:)<1));
                posDots(:,c) = [randi(this.width, [1 numel(c)]);
                    randi(this.height,[1 numel(c)])];
                
                % translate 2D subscripts to index, set to 1
                retina(sub2ind(size(retina),round(posDots(1,:)), ...
                    round(posDots(2,:)))) = 1;
                
                % append to frames
                res = cat(3, res, retina);
            end
            
            % append to existing frames
            this.privAppendFrames(res);
            
            % update default save file name
            lastName = this.defaultSaveName(max(1,end-8):end);
            if ~strcmp(lastName,'RdkExpand') % only save distinct names
                this.defaultSaveName = cat(2,this.defaultSaveName,'RdkExpand');
            end
        end
        
        
        function addSinGrating(this, length, sinDir, sinFreq, sinContrast, ...
                sinPhase)
            % IS.addSinGrating(length, sinDir, sinFreq, sinContrast, sinPhase)
            % will add a drifting sinusoidal grating with mean intensity value
            % 128 and a contrast of sinContrast to the existing stimulus.
            %
            % LENGTH       - The number of frames to create. Default is 50.
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
            if nargin<2,length=50;end
            
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
        
        
        function displayFrames(this,frames)
            % img = IS.displayFrames(frames) displays the specified frames in
            % the current figure/axes.
            %
            % FRAMES    - A list of frame numbers. For example, requesting
            %             frames=[1 2 8] will return the first, second, and
            %             eighth frame in a width-by-height-by-3 matrix.
            %             Default: display all frames.
            this.privLoadStimIfNecessary(); % need to load stim first
            if nargin<2,frames = 1:this.length;end
            
            set(gcf,'KeyPressFcn',@this.privPauseOnKeyPressCallback)
            
            % display frame in specified axes
            img = this.getFrames(frames);
            for i=frames
                if this.plotAbortPlotting
                    % user pressed button to quit plotting
                    this.plotAbortPlotting = false;
                    close;
                    return
                end
                
                colormap gray
                imagesc(permute(img(:,:,i),[2 1 3]),[0 1])
                text(2,this.height-1,num2str(find(frames==i)), ...
                    'FontSize',10,'BackgroundColor','white')
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
        
        function saveToFile(this, fileName)
            % IS.saveToFile(fileName) saves a VisualStimulus object to fileName.
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
            if ~isnumeric(frame) || numel(frame)~=1
                error('Cannot act on more than one frame at once')
            end
            if min(frame(:))<0 || max(frame(:))>1
                error('Grayscale values must be in the range [0,1]')
            end
            
            switch lower(type)
                case 'gaussian'
                    valMean = 0;
                    valVar = 0.01;
                    if numel(varargin)>=1,valMean=varargin{1};end
                    if numel(varargin)>=2,valVar =varargin{2};end
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
                    if numel(varargin)>=1,valD=varargin{1};end
                    noisy = imnoise(frame,'salt & pepper',valD);
                case 'speckle'
                    valVar = 0.04;
                    if numel(varargin)>=1,valVar=varargin{1};end
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
            % Private method to load a VisualStimulus object from fileName.
            % This file must have been created using method
            % VisualStimulus.saveToFile.
            % Make sure to have read access to the specified file.
            %
            % FILENAME       - relative or absolute path to a binary file
            %                  containing a VisualStimulus object.
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
        
        % function s = mkDots(stimSz, dotDirection, dotSpeed, dotDensity, dotCoherence,
        %                   dotRadius, densityStyle, sampleFactor, interpMethod)
        %
        % MKDOTS makes a drifting dot stimulus.
        %
        % Required arguments:
        % stimSz            The dimensions of the stimulus, in [Y X T] coordinates;
        % dotDirection      The direction of movement, in radians, with 0 = rightward.
        % (0=rightward, pi/2=upward; angle increases counterclockwise)
        %                   If dotDirection is just one number, the dots will move
        %                   in that direction at all times. If dotDirection is a
        %                   vector of the same length as the number of frames in
        %                   the stimulus, the direction of movement in each frame
        %                   will be specified by the corresponding element of
        %                   dotDirection.
        % dotSpeed          The speed of movement, in frames/second.
        %                   If dotSpeed is just one number, the dots will move
        %                   with that speed at all times. If dotSpeed is a
        %                   vector of the same length as the number of frames in
        %                   the stimulus, the speed of movement in each frame
        %                   will be specified by the corresponding element of
        %                   dotSpeed.
        %
        % Optional arguments:
        % dotDensity        The density of the dots, which can be from 0 to 1. DEFAULT = .1
        % dotCoherence      The coherence of dot motion. DEFAULT = 1.
        % dotRadius         The radius of the dots. If dotRadius < 0, the dots will
        %                   be single pixels. If dotRadius > 0, the dots will be
        %                   Gaussian blobs with sigma = .5 * dotRadius. DEFAULT = -1
        % dotPlacementStyle The number of dots is calculated by multiplying the
        %                   dotDensity by the size of the image window. If the dots
        %                   are placed randomly, some dots will overlap, so the
        %                   actual dot density will be lower than dotDensity. If,
        %                   however, the dots are placed exactly so that they never
        %                   overlap, all the dots will move in register. Both of
        %                   these are problems that only occur when you use pixel
        %                   dots rather than Gaussian dots. Your choices for
        %                   dotPlacementStyle are 'random' (for the first problem)
        %                   or 'exact' (for the second problem). DEFAULT = 'random'
        % sampleFactor      Only important if you are using Gaussian dots. The dots
        %                   are made by calculating one large Gaussian dot, and
        %                   interpolating from that dot to make the smaller dots.
        %                   This parameter specifies how much larger the large dot
        %                   is than the smaller dots. DEFAULT = 10.
        % interpMethod      Only important if you are using Gaussian dots.
        %                   Specifies the interpolation method used in creating the
        %                   smaller dots. Choices: 'linear', 'cubic', 'nearest',
        %                   'v4'. DEFAULT = 'linear'
        
        function s = privMakeDots(this, length, dotDirection, dotSpeed, ...
                dotDensity, dotCoherence, dotRadius, densityStyle, ...
                sampleFactor, interpMethod)
            % This method was initially authored by Timothy Saint and Eero
            % P. Simoncelli at NYU. It was adapted for coordinate system

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

            % resize dotDirection and dotSpeed if necessary
            if numel(dotDirection) == 1
                dotDirection = repmat(dotDirection, stimSz(3), 1);
            elseif numel(dotDirection) ~= length
                error(['If DOTDIRECTION is a vector, it must have the same ' ...
                    'number of entries as there are frames in the stimulus'])
            end
            if numel(dotSpeed) == 1
                dotSpeed = repmat(dotSpeed, stimSz(3), 1);
            elseif numel(dotSpeed) ~= length
                error(['If DOTSPEED is a vector, it must have the same ' ...
                    'number of entries as there are frames in the stimulus'])
            end
            
            
            % make the large dot for sampling
            xLargeDot = linspace(-(3/2)*dotRadius, (3/2)*dotRadius, ceil(3*dotRadius*sampleFactor));
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
%             yFrame = flipud(yFrame);
            
            
            % nDots is the number of coherently moving dots in the stimulus.
            % nDotsNonCoherent is the number of noncoherently moving dots.
            nDots = round(dotCoherence.*dotDensity.*numel(xFrame));
            nDotsNonCoherent = round((1-dotCoherence).*dotDensity.*numel(xFrame));
            
            % Set the initial dot positions.
            % dotPositions is a matrix of positions of the coherently moving dots in
            %   [y, x] coordinates; each row in dotPositions stores the position of one
            %   dot.
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
            
            % s will store the output. After looping over each frame, we will trim away
            % the buffer from s to obtain the final result.
            s = zeros(frameSzX, frameSzY, stimSz(3));
            toInterpolate = [-floor((3/2)*dotRadius):floor((3/2)*dotRadius)];
            dSz = floor((3/2)*dotRadius);
            for t = 1:stimSz(3)
                
                % move the positions of all the dots
                dotVelocity = [cos(dotDirection(t)), sin(dotDirection(t))];
                dotVelocity = dotVelocity*dotSpeed(t);
                dotPositions = dotPositions + repmat(dotVelocity, size(dotPositions, 1), 1);
                
                % wrap around for all dots that have gone past the image borders
                w = find(dotPositions(:,1) > frameSzX + .5);
                dotPositions(w,1) = dotPositions(w,1) - frameSzX;
                
                w = find(dotPositions(:,1) < .5);
                dotPositions(w,1) = dotPositions(w,1) + frameSzX;
                
                w = find(dotPositions(:,2) > frameSzY + .5);
                dotPositions(w,2) = dotPositions(w,2) - frameSzY;
                
                w = find(dotPositions(:,2) < .5);
                dotPositions(w,2) = dotPositions(w,2) + frameSzY;
                
                % add noncoherent dots and make a vector of dot positions for this
                % frame only.
                dotPositionsNonCoherent = rand(nDotsNonCoherent, 2) * [frameSzX-1 0; 0 frameSzY-1] + .5;
                
                % create a temporary matrix of positions for dots to be shown in this
                % frame.
                tmpDotPositions = [dotPositions; dotPositionsNonCoherent];
                
                % prepare a matrix of zeros for this frame
                thisFrame = zeros(size(xFrame));
                if dotRadius > 0
                    % in each frame, don't show dots near the edges of the frame. That's
                    % why we have a buffer. The reason we don't show them is that we don't
                    % want to deal with edge handling.
                    w1 = find(tmpDotPositions(:,1) > frameSzX - bufferSize + (3/2)*dotRadius);
                    w2 = find(tmpDotPositions(:,1) < bufferSize - (3/2)*dotRadius);
                    w3 = find(tmpDotPositions(:,2) > frameSzY - bufferSize + (3/2)*dotRadius);
                    w4 = find(tmpDotPositions(:,2) < bufferSize - (3/2)*dotRadius);
                    w = [w1; w2; w3; w4];
                    tmpDotPositions(w, :) = [];
                    
                    % add the dots to thisFrame
                    for p = 1:size(tmpDotPositions, 1)
                        
                        % find the center point of the current dot, in thisFrame
                        % coordinates. This is where the dot will be placed.
                        cpX = round(tmpDotPositions(p, 1));
                        cpY = round(tmpDotPositions(p, 2));
                        
                        xToInterpolate = toInterpolate + (round(tmpDotPositions(p,1)) - tmpDotPositions(p,1));
                        yToInterpolate = toInterpolate + (round(tmpDotPositions(p,2)) - tmpDotPositions(p,2));
                        [xToInterpolate, yToInterpolate] = meshgrid(xToInterpolate, yToInterpolate);
                        thisSmallDot = interp2(xLargeDot, yLargeDot, largeDot, ...
                            xToInterpolate, yToInterpolate, interpMethod);
                        
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
                    
                    w = sub2ind(size(thisFrame), tmpDotPositions(:,1), tmpDotPositions(:,2));
                    
                    thisFrame(w) = 1;
                end
                % Add this frame to the final output
                s(:,:,t) = thisFrame;
            end
            % Now trim away the buff
            s = s(bufferSize+1:end-bufferSize, bufferSize+1:end-bufferSize, :);
            s(s>1) = 1;
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
            %
            % LENGTH       - The number of frames to create.
            % DIR          - The drifting direction of the stimulus in radians
            %                (where 0=rightward, pi/2=upward; angle increases
            %                counterclockwise).
            % FREQ         - 2-D vector of stimulus frequency. The first
            %                component is the spatial frequency (cycles/pixels),
            %                whereas the second component is the temporal
            %                frequency (cycles/frame)
            % ANGLE        - The angle between the grating components in
            %                radians. Default is (2/3)*pi = 120 degrees.
            % CONTR        - The overall plaid contrast. Default is 1.
            firstDirection = mod(dir + angle/2, 2*pi);
            secondDirection = mod(dir - angle/2, 2*pi);
            firstGrating = this.privMakeSin(length, firstDirection, freq, ...
                contr/2, 0);
            secondGrating = this.privMakeSin(length, secondDirection, freq, ...
                contr/2, 0);
            res = firstGrating + secondGrating;
        end
        
        function res = privMakeSin(this, length, dir, freq, contr, phase)
            % A private method that creates a drifting sinusoidal grating with
            % mean intensity value 128 and a contrast of CONTR.
            %
            % This method was initially authored by Timothy Saint and Eero
            % P. Simoncelli at NYU. It was adapted for coordinate system
            % [x,y,t]. For more information see beginning of this file.
            %
            % LENGTH       - The number of frames to create.
            % DIR          - The drifting direction of the stimulus in radians
            %                (where 0=rightward, pi/2=upward; angle increases
            %                counterclockwise).
            % FREQ         - 2-D vector of stimulus frequency. The first
            %                component is the spatial frequency (cycles/pixels),
            %                whereas the second component is the temporal
            %                frequency (cycles/frame)
            % CONTR        - The grating contrast.
            % PHASE        - The initial phase of the grating in periods.
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
                    this.plotAbortPlotting = true;
            end
        end
        
        function privSetFrame(this,index,frame)
            % Private method to replace/insert a single frame
            if ~isnumeric(index) || size(frame,3)>1
                error('Cannot set more than one frame at the same time')
            end
            if size(frame,1) ~= stim.width
                error(['Stimulus width (' num2str(stim.width) ') does not ' ...
                    'match frame width (' num2str(size(frame,1)) ')'])
            end
            if size(frame,2) ~= stim.height
                error(['Stimulus height (' num2str(stim.height) ') does ' ...
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