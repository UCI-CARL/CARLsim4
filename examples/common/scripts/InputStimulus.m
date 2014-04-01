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
        width;
        height;
        length;
        stim;
        channels;
        mode;
        
        supportedImageModes;
        supportedNoiseTypes;
        supportedStimulusTypes;
    end
    
    properties (SetAccess = private, GetAccess = private)
        needToLoad;
        loadFile;
        defaultSaveName;
        plotAbortPlotting;
    end
    
    
    %% PUBLIC METHODS
    methods
        % Matlab doesn't supper function overloading, so use varargin instead
        function obj = InputStimulus(varargin)
            obj.width = -1;
            obj.height = -1;
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
        
        function createPlaid(this, length, plaidDir, gratFreq, ...
                plaidAngle, plaidContrast, append)
            % IS.createPlaid(length, dir, freq, angle, contr, append) creates a
            % drifting plaid stimulus with mean intensity value 128 and a
            % contrast of value CONTR. The plaid stimulus is made of two
            % sinusoidal gratings ("grating components") separated by a
            % specified ANGLE.
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
            %
            % APPEND        - A flag to indicate whether the hereby created
            %                 frames should be appended to any existing frames
            %                 (true), or whether existing frames should be
            %                 discarded (false). Default is false (thus
            %                 replacing all previously created frames).
            %
            if nargin<7,append=false;end
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
            if ~islogical(append),error('Append flag must be true or false'),end
            
            % if this object was created by loading a file, read the header
            % (loadHeaderOnly=true) to get stimulus dimensions
            this.loadStimIfNecessary(true);
            
            % use S&H script
            res = this.mkPlaid(length, plaidDir, gratFreq, plaidAngle, ...
                plaidContrast);
            
            % append to existing frames or replace
            if append
                this.loadStimIfNecessary(); % need to update stim first
                this.stim = cat(3, this.stim, res);
            else
                this.stim = res;
            end
            
            % update attributes
            this.needToLoad = false; % stim is up-to-date
            this.length = size(this.stim,3);
            
            % update default save file name
            this.defaultSaveName = cat(2,this.defaultSaveName,'Plaid');
        end
        
        function createSinGrating(this, length, sinDir, sinFreq, ...
                sinContrast, sinPhase, append)
            % IS.createSinGrating(length, sinDir, sinFreq, sinContrast,
            % sinPhase, append) creates a drifting sinusoidal grating with mean
            % intensity value 128 and a contrast of sinContrast.
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
            % APPEND       - A flag to indicate whether the hereby created
            %                frames should be appended to any existing frames
            %                (true), or whether existing frames should be
            %                discarded (false). Default is false (thus replacing
            %                all previously created frames).
            %
            % This method is based on a script by Timothy Saint and Eero P.
            % Simoncelli at NYU. It was adapted for coordinate system [x,y,t].
            % For more information see beginning of this file.
            %
            if nargin<7,append=false;end
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
            if ~islogical(append),error('Append flag must be true or false'),end
            
            % if this object was created by loading a file, read the header
            % (loadHeaderOnly=true) to get stimulus dimensions
            this.loadStimIfNecessary(true);
            
            % use S&H script
            res = this.mkSin(length, sinDir, sinFreq, sinContrast, sinPhase);
            
            % append to existing frames or replace
            if append
                this.loadStimIfNecessary(); % need to load stim first
                this.stim = cat(3, this.stim, res);
            else
                this.stim = res;
            end
            
            % update attributes
            this.needToLoad = false; % stim is up-to-date
            this.length = size(this.stim,3);
            
            % update default save file name
            this.defaultSaveName = cat(2,this.defaultSaveName,'Grating');
        end
        
        
        function displayFrames(this,frames)
            % img = IS.dispFrame(frame) displays the specified frames in the
            % current figure/axes.
            %
            % FRAME     - A list of frame numbers. For example, requesting
            %             frames=[1 2 8] will return the first, second, and
            %             eighth frame in a width-by-height-by-3 matrix.
            %             Default: display all frames.
            this.loadStimIfNecessary(); % need to load stim first
            if nargin<2,frames = 1:this.length;end
            
            set(gcf,'KeyPressFcn',@this.pauseOnKeyPressCallback)
            
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
                imagesc(permute(img(:,:,i),[2 1 3]),[0 1]);
                text(2,this.height-1,num2str(find(frames==i)), ...
                    'FontSize',10,'BackgroundColor','white');
                drawnow;
                pause(0.1);
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
            if ~isnumeric(frames) || sum(frames>this.length)>0 || sum(frames<1)>0
                error(['Specified frames must be in the range [1,' ...
                    num2str(this.length) ']'])
            end
            
            this.loadStimIfNecessary(); % need to load stim first
            
            % return frames
            img = this.stim(:,:,frames);
        end
        
        function insertNoiseFrames(this, length, grayVal, type, varargin)
            if nargin<4,type='gaussian';end
            if nargin<3,grayVal=0.5;end
            if nargin<2,length=1;end
            
            % base mean value
            blank = ones(this.width, this.height)*grayVal;
            
            img=[];
            for i=1:length
                % for each frame, add noise
                
                switch lower(type)
                    case 'gaussian'
                        if numel(varargin)<2
                            error(['Need to specify mean and variance of ' ...
                                'Gaussian noise'])
                        end
                        noise = imnoise(blank,'gaussian',varargin{1},varargin{2});
                    case 'poisson'
                        noise = imnoise(blank,'poisson');
                    case 'salt & pepper'
                        if numel(varargin)<1
                            error(['Need to specify noise density of Salt & ' ...
                                'Pepper noise'])
                        end
                        noise = imnoise(blank,'salt & pepper',varargin{1});
                    case 'speckle'
                        if numel(varargin)<1
                            error('Need to specify variance of speckle noise')
                        end
                        noise = imnoise(blank,'speckle',varargin{1});
                    otherwise
                        error(['Unknown noise type. Currently supported are: ' ...
                            this.supportedNoiseTypes])
                end
                
                % confine to [0,1]
                noise = max(0, min(1,noise));
                img = cat(3, img, noise);
            end
            

            % by default, this function appends to stimulus
            % so the stimulus must be loaded first
            this.loadStimIfNecessary();
            
            % append noise
            this.stim = cat(3, this.stim, img);
            
            % update attributes
            this.needToLoad = false; % stim is up-to-date
            this.length = size(this.stim,3);
        end
        
        function loadFromFile(this, fileName, loadHeaderOnly)
            % IS.loadFromFile(fileName) loads a VisualStimulus object from
            % fileName. This file must have been created using method
            % VisualStimulus.saveToFile.
            % Make sure to have read access to the specified file.
            %
            % FILENAME    - relative or absolute path to a binary file
            %               containing a VisualStimulus object.
            % LOADHEADERONLY - A flag to indicate whether only the header should
            %                  be read. This is helpful if one only cares about
            %                  the stimulus dimensions and such. Default: false.
            if nargin<3,loadHeaderOnly=false;end
            fid=fopen(fileName,'r');
            if fid==-1
                error(['Could not open "' fileName '" with read permission.']);
            end
            
            % read version number
            version = fread(fid,1,'float');
            if (version ~= 1.0)
                error(['Unknown file type, must have Version 1.0 (Version ' ...
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
                    num2str(this.sizeOf()) ...
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
        
        function saveToFile(this, fileName)
            % IS.saveToFile(fileName) saves a VisualStimulus object to fileName.
            % Later the stimulus can be loaded by using method
            % VisualStimulus.loadFromFile.
            %
            % FILENAME       - The name of the file to create (optional).
            %                  Default is to use a name consisting of a stimulus
            %                  description and the stimulus dimensions (to be
            %                  created in the current directory).
            %                  Make sure to have write access to the specified
            %                  file, and that the directory exists.
            if nargin<2
                fileName = [this.defaultSaveName upper(this.mode(1)) ...
                    this.mode(2:end) '_' num2str(this.width) ...
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
        function loadStimIfNecessary(this, loadHeaderOnly)
            % IS.loadStimIfNecessary() will load the stimulus file if necessary
            % (which is determined by the flag needToLoad)
            if nargin<2,loadHeaderOnly=false;end
            if this.needToLoad
                this.loadFromFile(this.loadFile,loadHeaderOnly);
                
                % if loaded the stimulus, set the following flag to false
                % but don't if only header was read
                if ~loadHeaderOnly
                    this.needToLoad = false;
                end
            end
        end
        
        function res = mkPlaid(this, length, dir, freq, angle, contr)
            % res = IS.mkPlaid(length, dir, freq, angle, contr) creates a
            % drifting plaid stimulus with mean intensity value 128 and a
            % contrast of contr. The plaid stimulus is made of two sinusoidal
            % gratings ("grating components") separated by a specified angle.
            %
            % This method was initially authored by Timothy Saint and Eero
            % P. Simoncelli at NYU. It was adapted for coordinate system
            % [x,y,t]. For more information see beginning of this file.
            %
            % LENGTH       - The number of frames to create.
            % DIR          - The drifting direction of the stimulus in radians
            %                (where 0=rightward).
            % FREQ         - 2-D vector of stimulus frequency. The first
            %                component is the spatial frequency (cycles/pixels),
            %                whereas the second component is the temporal
            %                frequency (cycles/frame)
            % ANGLE        - The angle between the grating components in
            %                radians. Default is (2/3)*pi = 120 degrees.
            % CONTR        - The overall plaid contrast. Default is 1.
            firstDirection = mod(dir + angle/2, 2*pi);
            secondDirection = mod(dir - angle/2, 2*pi);
            firstGrating = this.mkSin(length, firstDirection, freq, ...
                contr/2, 0);
            secondGrating = this.mkSin(length, secondDirection, freq, ...
                contr/2, 0);
            res = firstGrating + secondGrating;
        end
        
        function res=mkSin(this, length, dir, freq, contr, phase)
            % res = IS.mkSin(length, dir, freq, contrast, phase) creates a
            % drifting sinusoidal grating with mean intensity value 128 and a
            % contrast of contr.
            %
            % This method was initially authored by Timothy Saint and Eero
            % P. Simoncelli at NYU. It was adapted for coordinate system
            % [x,y,t]. For more information see beginning of this file.
            %
            % LENGTH       - The number of frames to create.
            % DIR          - The drifting direction of the stimulus in radians
            %                (where 0=rightward).
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
            
            res = cos(2*pi*sf*cos(dir)*x + ...
                2*pi*sf*sin(dir)*y - ...
                2*pi*tf*t + ...
                2*pi*phase);
            
            res = contr.*res./2 + .5;
        end
        
        function pauseOnKeyPressCallback(this,~,eventData)
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
        
        function len = sizeOf(this)
            % returns the total number of elements (pixels) in a stimulus
            len = this.width*this.height*this.length*this.channels;
        end
    end
end