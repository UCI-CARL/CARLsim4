classdef GroupMonitor < handle
    % A GroupMonitor can be used to monitor properties as well as the
    % activity of a specific neuronal group.
    %
    % A GroupMonitor will assume that a corresponding spike file has been
    % created during the CARLsim simulation. 
    %
    % Example usage:
    % >> GM = GroupMonitor('excit','results/');
    % >> GM.plot; % hit 'p' to pause, 'q' to quit
    % >> GM.setPlotType('heatmap'); % switch to heat map
    % >> GM.setRecordingAttributes('fps',10); % set recording FPS
    % >> GM.recordMovie; % plots heat map and saves as 'movie.avi'
    % >> % etc.
    %
    % Version 10/5/2014
    % Author: Michael Beyeler <mbeyeler@uci.edu>
    
    %% PROPERTIES
    % public
    properties (SetAccess = private)
        name;               % group name
        resultsFolder;      % results folder
        errorMode;          % program mode for error handling
        supportedErrorModes;% supported error modes
        supportedPlotTypes; % cell array of supported plot types
    end
    
    % private
    properties (Hidden, Access = private)
        spkObj;             % SpikeReader object
        spkFilePrefix;      % spike file prefix, e.g. "spk"
        spkFileSuffix;      % spike file suffix, e.g. ".dat"
        spkData;            % buffer for spike data
        
        needToInitSR;       % flag whether we need to init SpikeReader
        needToLoadData;     % flag whether we need to load spike data

        grid3D;             % Grid3D topography of group
        plotType;           % current plot type
        
        plotAbortPlotting;  % flag whether to abort plotting (on-click)
        plotBgColor;        % bg color of plot (for plotting)
        plotDispFrameNr;    % flag whether to display frame number
        plotFPS;            % frames per second for plotting
        plotBinWinMs;       % binning window size (time)
        plotStepFrames;     % flag whether to waitforbuttonpress btw frames
        plotInteractiveMode;% flag whether to allow click/key events
        
        recordBgColor;      % bg color of plot (for recording)
        recordFile;         % filename for recording
        recordFPS;          % frames per second for recording
        recordWinSize;      % window size of plot for recording
        
        errorFlag;          % error flag (true if error occured)
        errorMsg;           % error message
    end
    
    
    %% PUBLIC METHODS
    methods
        function obj = GroupMonitor(name, resultsFolder, errorMode)
            % GM = GroupMonitor(name,resultsFolder,errorMode) creates a new
            % instance of class GroupMonitor, which can be used to monitor
            % group properties and activity.
            % A GroupMonitor will assume that a corresponding spike file
            % has been created during the CARLsim simulation. Otherwise no
            % plotting can be done.
            %
            % NAME           - A string specifying the group to monitor.
            %                  This should be the same as the name that was
            %                  given to the group via CARLsim::createGroup.
            % RESULTSFOLDER  - Path to the results directory (where the
            %                  spike file lives). Default: current dir.
            % ERRORMODE      - Error Mode in which to run SpikeReader. The
            %                  following modes are supported:
            %                   - 'standard' Errors will be fatal (returned
            %                                via Matlab function error())
            %                   - 'warning'  Errors will be warnings
            %                                returned via Matlab function
            %                                warning())
            %                   - 'silent'   No exceptions will be thrown,
            %                                but object will populate the
            %                                properties errorFlag and
            %                                errorMsg.
            %                  Default: 'standard'.
            obj.name = name;
            obj.unsetError()
            obj.loadDefaultParams();
            
            if nargin<3
                obj.errorMode = 'standard';
            else
                if ~obj.isErrorModeSupported(errorMode)
                    obj.throwError(['errorMode "' errorMode '" is ' ...
                        'currently not supported. Choose from the ' ...
                        'following: ' ...
                        strjoin(obj.supportedErrorModes,', ') ...
                        '.'], 'standard')
                    return
                end
                obj.errorMode = errorMode;
            end
            if nargin<2
                obj.resultsFolder = '';
            else
                obj.resultsFolder = resultsFolder;
            end
            if nargin<1
                obj.throwError('No group name given.');
                return
            end
        end
                
        function delete(obj)
            % destructor, implicitly called
            clear obj.spkObj;
        end
        
        function plotType = getDefaultPlotType(obj)
            % plotType = GM.getDefaultPlotType() returns the default
            % plotting type for this group, which is determined by the
            % group's Grid3D topography. For example, a 1D topography will
            % prefer a raster plot, whereas a 2D topography will prefer a
            % heat map.
            % See a list of all currently supported plot types in the help
            % section of GM.plot and in the variable GM.supportedPlotTypes.
            % The plotting type can also be set manually using
            % GM.setPlotType.
            obj.unsetError()
            obj.initSpikeReader() % required to access Grid3D prop
            plotType = 'default';
            
            % find dimensionality of Grid3D
            % i.e., Nx1x1 is 1D, NxMx1 is 2D, NxMxL is 3D
            dims = 3-sum(obj.grid3D==1);
            if dims==1
                plotType = 'raster'; % 1D layout
            elseif dims==2
                plotType = 'heatmap'; % 2D layout
            else
                % \TODO add more logic and more types
                plotType = 'raster';
            end
        end
        
        function [errFlag,errMsg] = getError(obj)
            % [errFlag,errMsg] = getError() returns the current error
            % status.
            % If an error has occurred, errFlag will be true, and the
            % message can be found in errMsg.
            errFlag = obj.errorFlag;
            errMsg = obj.errorMsg;
        end
        
        function spkFile = getSpikeFileName(obj)
            % spkFile = GM.getSpikeFileName() returns the name of the
            % spike file according to specified prefix and suffix.
            % Prefix and suffix can be set using GM.setSpikeFileAttributes.
            spkFile = [ obj.resultsFolder ... % the results folder
                filesep ...                   % platform-specific separator
                obj.spkFilePrefix ...         % something like 'spk_'
                obj.name ...                  % the name of the group
                obj.spkFileSuffix ];          % something like '.dat'
        end
                
        function hasValid = hasValidSpikeFile(obj)
            % hasValid = GM.hasValidSpikeFile() determines whether a valid
            % spike file can be found for the group.
            % If no file can be found, the prefix and suffix of the spike
            % file name need to be updated. This can be done using
            % GM.setSpikeFileAttributes.
            obj.unsetError()
            
            spkFile = obj.getSpikeFileName();
            SR = SpikeReader(spkFile, false, 'silent');
            [errFlag,~] = SR.getError();
            hasValid = ~errFlag;
        end
                
        function plot(obj, plotType, frames, binWindowMs, stepFrames)
            % GM.plot(plotType, frames, binWindowMs, stepFrames) plots the
            % specified frames in the current figure/axes. A list of
            % plotting attributes can be set directly as input arguments.
            %
            % The full list of available attributes can be set using
            % GM.setPlottingAttributes.
            %
            % PLOTTYPE     - The plotting type to use. If not set, the
            %                default plotting type will be used, which is
            %                determined by the Grid3D topography of the
            %                group.
            %                The following types are currently supported:
            %                 - heatmap   a topological map of group
            %                             activity where hotter colors mean
            %                             higher firing rate
            %                 - raster    a raster plot with binning window
            %                             binWindowMs
            %                Default: 'default'.
            % FRAMES       - A list of frame numbers. For example,
            %                requesting frames=[1 2 8] will display the
            %                first, second, and eighth frame.
            %                Default: display all frames.
            % BINWINDOWMS  - The binning window (ms) in which the data will
            %                be displayed. Default: 1000.
            % STEPFRAMES   - A boolean flag that indicates whether to wait
            %                for user input (button press) before
            %                displaying the next frame. Default: false.
            if nargin<5,stepFrames=obj.plotStepFrames;end
            if nargin<4,binWindowMs=obj.plotBinWinMs;end
            if nargin<3 || isempty(frames) || frames==-1
                obj.initSpikeReader()
                frames = 1:ceil(obj.spkObj.getSimDurMs()/binWindowMs);
            end
            if nargin<2,plotType=obj.plotType;end
            obj.unsetError()
            
            % verify input
            if ~Utilities.verify(frames,{{'isvector','isnumeric',[1 inf]}})
                obj.throwError('Frames must be a numeric vector e[1,inf]')
                return
            end
            if ~Utilities.verify(binWindowMs,{{'isscalar',[1 inf]}})
                obj.throwError('Frame duration must be a scalar e[1 inf]')
                return
            end
            if ~Utilities.verify(stepFrames,{'islogical','isnumeric'})
                obj.throwError('stepFrames must be true/false');return
            end
            
            % reset abort flag, set up callback for key press events
            if obj.plotInteractiveMode
                obj.plotAbortPlotting = false;
                set(gcf,'KeyPressFcn',@obj.pauseOnKeyPressCallback)
            end
            
            % load data and reshape for plotting if necessary
            obj.loadDataForPlotting(plotType, binWindowMs);
            
            % display frame in specified axes
            for i=frames
                if obj.plotInteractiveMode && obj.plotAbortPlotting
                    % user pressed button to quit plotting
                    obj.plotAbortPlotting = false;
                    close;
                    return
                end
                
                obj.plotFrame(i,plotType,binWindowMs,obj.plotDispFrameNr);
                drawnow

                % wait for button press or pause
                if obj.plotInteractiveMode
                    if stepFrames || i==frames(end)
                        waitforbuttonpress;
                    else
                        pause(1.0/obj.plotFPS)
                    end
                end
            end
            if obj.plotInteractiveMode,close;end
        end
        
        function recordMovie(obj, fileName, frames, binWindowMs, fps, winSize)
            % NM.recordMovie(fileName, frames, frameDur, fps, winSize)
            % takes an AVI movie of a list of frames using the VIDEOWRITER
            % utility.
            %
            % FILENAME     - A string enclosed in single quotation marks
            %                that specifies the name of the file to create.
            %                Default: 'movie.avi'.
            % FRAMES       - A list of frame numbers. For example,
            %                requesting frames=[1 2 8] will return the
            %                first, second, and eighth frame in a
            %                width-by-height-by-3 matrix.
            %                Default: return all frames.
            % BINWINDOWMS  - The binning window (ms) in which the data will
            %                be displayed. Default: 1000.
            % FPS          - Rate of playback for the video in frames per
            %                second. Default: 10.
            % WINSIZE      - A 2-element vector specifying the window size
            %                of the video as width x height in pixels. Set
            %                to [0 0] in order to automatically make the 
            %                movie window fit to the size of the plot
            %                window. Default: [0 0].
            if nargin<6,winSize=obj.recordWinSize;end
            if nargin<5,fps=obj.recordFPS;end
            if nargin<4,binWindowMs=obj.plotBinWinMs;end
            if nargin<3 || isempty(frames) || frames==-1
                obj.initSpikeReader()
                frames = 1:ceil(obj.spkObj.getSimDurMs()/binWindowMs);
            end
            if nargin<2,fileName=obj.recordFile;end
            obj.unsetError()
            
            % verify input
            if ~Utilities.verify(fileName,'ischar')
                obj.throwError('File name must be a string');return
            end
            if ~Utilities.verify(frames,{{'isvector','isnumeric',[1 inf]}})
                obj.throwError('Frames must be a numeric vector e[1,inf]')
                return
            end
            if ~Utilities.verify(binWindowMs,{{'isscalar',[1 inf]}})
                obj.throwError('Frame duration must be a scalar e[1 inf]')
                return
            end
            if ~Utilities.verify(fps,{{'isscalar',[0.01 100]}})
                obj.throwError('FPS must be in range [0.01,100]');return
            end
            if ~Utilities.verify(winSize,{{'isvector','isnumeric',[0 inf]}})
                obj.throwError(['Window size must be a numeric vector ' ...
                    'with values > 0']);return
            end
            
            % load data and reshape for plotting if necessary
            obj.loadDataForPlotting(obj.plotType, binWindowMs);
            
            % display frames in specified axes
            set(gcf,'color',obj.plotBgColor);
            if sum(winSize>0)==2
                set(gcf,'Position',[100 100 winSize]);
            end
            set(gcf,'PaperPositionMode','auto');
            
            % open video object
            vidObj = VideoWriter(fileName);
            vidObj.Quality = 100;
            vidObj.FrameRate = fps;
            open(vidObj);
            
            % display frame in specified axes
            for i=frames
                obj.plotFrame(i,obj.plotType,binWindowMs,obj.plotDispFrameNr);
                drawnow
                writeVideo(vidObj, getframe(gcf));
            end
            close(gcf)
            close(vidObj);
            disp(['created file "' fileName '"'])
        end
        
        function setGrid3D(obj, grid3D)
            % GM.setGrid3D(grid3D) sets the Grid3D topography of the group.
            % The total number of neurons in the group (width x height x
            % depth) cannot change.
            % GRID3D         - A 3-element vector that specifies the width,
            %                  the height, and the depth of the 3D neuron
            %                  grid. The product of these dimensions should
            %                  equal the total number of neurons in the
            %                  group.
            obj.unsetError()
            
            % used to rearrange group layout
            if prod(grid3D) ~= prod(obj.grid3D)
                obj.throwError(['Population size cannot change when ' ...
                    'assigning new Grid3D property (old: ' ...
                    num2str(prod(obj.grid3D)) ', new: ' ...
                    num2str(prod(grid3D)) '.'])
                return
            end
            
            if logical(sum(obj.grid3D~=grid3D))
                obj.needToLoadData = true;
            end
            obj.grid3D = grid3D;
        end
                
        function setPlotType(obj, plotType)
            % GM.setPlotType(plotType) applies a certain plotting type to
            % the group. The default plot type is determined by the Grid3D
            % topography of the group. For example, a 1D topography will
            % prefer a raster plot, whereas a 2D topography will prefer a
            % heatmap.
            %
            % PLOTTYPE    - The plotting type to apply.
            %               The following types are currently supported:
            %                   - heatmap   a topological map of group
            %                               activity where hotter colors
            %                               mean higher firing rate
            %                   - raster    a raster plot with binning
            %                               window: binWindowMs
            obj.unsetError()
            
            % find default plot type if necessary
            if strcmpi(plotType,'default')
                plotType = obj.getDefaultPlotType();
            end
            
            % make sure plot type is supported
            if ~obj.isPlotTypeSupported(plotType)
                obj.throwError(['plotType "' plotType '" is currently ' ...
                    'not supported.'])
%                     'Choose from the following: ' ...
%                     strjoin(obj.supportedPlotTypes, ', ') '.'])
                return
            end
            
            % set plot type
            if ~strcmpi(obj.plotType,plotType)
                obj.needToLoadData = true;
            end
            obj.plotType = plotType;
        end
        
        function setPlottingAttributes(obj, varargin)
            % GM.setPlottingAttributes(varargin) can be used to set default
            % settings that will apply to all activity plots.
            % This function provides control over additional attributes
            % that are not available as input arguments to GM.plot or
            % GM.plotFrame.
            % GM.setPlottingAttributes('propertyName1',value1,...) sets the
            % value of 'propertyName1' to value1.
            %
            % Calling the function without input arguments will restore the
            % default settings.
            %
            % BGCOLOR         - Set background color for figure. Must be of
            %                   type ColorSpec (char such as 'w','b','k' or
            %                   a 3-element vector for RGB channels).
            %                   Default: 'w'.
            % BINWINDOWMS     - The binning window (ms) in which the data
            %                   will be displayed. Default: 1000.
            % DISPFRAMENR     - A boolean flag that indicates whether to
            %                   display the frame number. Default: true.
            % FPS             - The frames per second for the plotting
            %                   loop. Default: 5.
            % INTERACTIVEMODE - A boolean flag to set InteractiveMode on or
            %                   off. If it is off, key events/FPS/stepping
            %                   will take no effect (helpful if you want to
            %                   take over control yourself). Default: true.
            % STEPFRAMES      - A boolean flag that indicates whether to
            %                   wait for user input (button press) before
            %                   displaying the next frame. Default: false.
            obj.unsetError()
            
            if isempty(varargin)
                % set default values
                obj.plotDispFrameNr = true;
                obj.plotBgColor = 'w';
                obj.plotFPS = 5;
                obj.plotBinWinMs = 1000;
                obj.plotStepFrames = false;
                obj.plotInteractiveMode = true;
                return;
            end
            
            % init error types
            throwErrFileEnding = false;
            throwErrNumeric = false;
            throwErrOutOfRange = false;
            
            nextIndex = 1;
            while nextIndex<length(varargin)
                attr = varargin{nextIndex};   % this one is attribute name
                val  = varargin{nextIndex+1}; % next is attribute value
                
                switch lower(attr)
                    case 'bgcolor'
                        % background color for figure
                        obj.plotBgColor = val;
                    case 'binwindowms'
                        % binning window size in ms
                        throwErrNumeric = ~isnumeric(val);
                        reqRange = [1 inf];
                        throwErrOutOfRange = val<reqRange(1) | val>reqRange(2);
                        obj.plotBinWinMs = val;
                    case 'dispframenr'
                        % whether to display frame number
                        throwErrNumeric = ~isnumeric(val) & ~islogical(val);
                        obj.plotDispFrameNr = logical(val);
                    case 'fps'
                        % frames per second
                        throwErrNumeric = ~isnumeric(val);
                        reqRange = [0.01 100];
                        throwErrOutOfRange = val<reqRange(1) | val>reqRange(2);
                        obj.plotFPS = val;
                    case 'interactivemode'
                        % interactive mode
                        throwErrNumeric = ~isnumeric(val) && ~islogical(val);
                        obj.plotInteractiveMode = logical(val);
                    case 'stepframes'
                        % whether to wait for button press before next frame
                        throwErrNumeric = ~isnumeric(val) & ~islogical(val);
                        obj.plotStepFrames = logical(val);
                    otherwise
                        % attribute does not exist
                        if isnumeric(attr) || islogical(attr)
                            attr = num2str(attr);
                        end
                        obj.throwError(['Unknown attribute "' attr '"'])
                        return
                end
                
                % throw errors
                if throwErrFileEnding
                    obj.throwError(['File ending for attr "' attr ...
                        '" must be "' reqFileEnding '"'])
                    return
                elseif throwErrNumeric
                    obj.throwError(['Value for attr "' attr ...
                        '" must be numeric'])
                    return
                elseif throwErrOutOfRange
                    obj.throwError(['Value for attr "' attr ...
                        '" must be in range [' num2str(reqRange(1)) ...
                        ',' num2str(reqRange(2)) ']'])
                    return
                end
                
                % advance index to next attr
                nextIndex = nextIndex + 2;
            end
        end
        
        function setRecordingAttributes(obj, varargin)
            % GM.setRecordingAttributes(varargin) can be used to set
            % default settings that will apply to all activity recordings.
            % This function provides control over additional attributes
            % that are not available as input arguments to GM.recordMovie.
            % GM.setRecordingAttributes('propertyName1',value1,...) sets
            % the value of 'propertyName1' to value1.
            %
            % Calling the function without input arguments will restore the
            % default settings.
            %
            % BGCOLOR        - Set background color for figure. Must be of
            %                  type ColorSpec (char such as 'w','b','k' or
            %                  a 3-element vector for RGB channels). The
            %                  default is white.
            % FILENAME       - File name where movie will be stored.
            %                  Currently the only supported file ending is
            %                  ".avi".
            % FPS            - The frames per second for the movie. The
            %                  default is 10.
            % WINSIZE        - A 2-element vector specifying the window
            %                  size of the video as width x height in
            %                  pixels.Set to [0 0] in order to
            %                  automatically make the movie window fit to
            %                  the size of the plot window.
            obj.unsetError()
            
            if isempty(varargin)
                % set default values
                obj.recordBgColor = 'w';
                obj.recordFile = 'movie.avi';
                obj.recordFPS = 5;
                obj.recordWinSize = [0 0];
                return;
            end
            
            % init error types
            % \TODO use Utilities.verify and obj.throwError
            throwErrFileEnding = false;
            throwErrNumeric = false;
            throwErrOutOfRange = false;
            
            nextIndex = 1;
            while nextIndex<length(varargin)
                attr = varargin{nextIndex};   % this one is attribute name
                val  = varargin{nextIndex+1}; % next is attribute value
                
                switch lower(attr)
                    case 'bgcolor'
                        % background color for figure
                        obj.recordBgColor = val;
                    case 'filename'
                        % filename for recorded movie (must be .avi)
                        reqFileEnding = '.avi';
                        throwErrFileEnding = ~strcmpi(val(max(1,end-3):end), ...
                            reqFileEnding);
                        obj.recordFile = val;
                    case 'fps'
                        % frames per second
                        throwErrNumeric = ~isnumeric(val);
                        reqRange = [0.01 100];
                        throwErrOutOfRange = val<reqRange(1) | val>reqRange(2);
                        obj.recordFPS = val;
                    case 'winsize'
                        % window size
                        throwErrNumeric = ~isnumeric(val);
                        reqRange = [1 inf];
                        throwErrOutOfRange = val<reqRange(1) | val>reqRange(2);
                        obj.recordWinSize = val;
                    otherwise
                        % attribute does not exist
                        if isnumeric(attr) || islogical(attr)
                            attr = num2str(attr);
                        end
                        obj.throwError(['Unknown attribute "' attr '"'])
                        return
                end
                
                % throw errors
                if throwErrFileEnding
                    obj.throwError(['File ending for attr "' attr ...
                        '" must be "' reqFileEnding '"'])
                    return
                elseif throwErrNumeric
                    obj.throwError(['Value for attr "' attr ...
                        '" must be numeric'])
                    return
                elseif throwErrOutOfRange
                    obj.throwError(['Value for attr "' attr ...
                        '" must be in range [' num2str(reqRange(1)) ...
                        ',' num2str(reqRange(2)) ']'])
                    return
                end
                
                % advance index to next attr
                nextIndex = nextIndex + 2;
            end
        end
        
        function setSpikeFileAttributes(obj,prefix,suffix)
            % obj.setSpikeFileAttributes(prefix,suffix)
            % Defines the naming conventions for spike files. They should
            % all reside within SAVEFOLDER (specified in constructor), and
            % be made of a common prefix, the population name (specified in
            % ADDPOPULATION), and a common suffix.
            % Example: files 'results/spkV1.dat', 'results/spkMT.dat'
            %   -> saveFolder = 'results/'
            %   -> prefix = 'spk'
            %   -> suffix = '.dat'
            %   -> name of population = 'V1' or 'MT'
            if nargin<3,suffix='.dat';end
            if nargin<2,prefix='spk';end
            obj.unsetError()
            
            % need to re-load if file name changes
            if ~strcmpi(obj.spkFilePrefix,prefix) ...
                    || ~strcmpi(obj.spikeFileSuffix,suffix)
                obj.needToInitSR = true;
                obj.needToLoadData = true;
            end
            obj.spkFilePrefix=prefix;
            obj.spkFileSuffix=suffix;
        end
    end
    
    %% PRIVATE METHODS
    methods (Hidden, Access = private)
        function initSpikeReader(obj)
            % private method to initialize SpikeReader
            obj.unsetError()
            
            spkFile = obj.getSpikeFileName();
            obj.spkObj = SpikeReader(spkFile, false, 'silent');
            
            % make sure spike file is valid
            [errFlag,errMsg] = obj.spkObj.getError();
            if errFlag
                obj.throwError(errMsg)
                return
            end
            obj.grid3D = obj.spkObj.getGrid3D();
            obj.needToInitSR = false;
            obj.needToLoadData = true;
        end
        
        function isSupported = isErrorModeSupported(obj, errMode)
            % determines whether an error mode is currently supported
            isSupported = sum(ismember(obj.supportedErrorModes,errMode))>0;
        end
        
        function isSupported = isPlotTypeSupported(obj, plotType)
            % determines whether a plot type is currently supported
            isSupported = sum(ismember(obj.supportedPlotTypes,plotType))>0;
        end
        
        function loadDataForPlotting(obj, plotType, plotBinWinMs)
            % Private function to prepare data for plotting.
            % The flag needToInitSR keeps track of changes to the spike
            % file name, so that a new SpikeReader object will be created
            % if the path to the spike file changes.
            % The flag needToLoadData keeps track of plotting settings, so
            % that the data is reloaded if attributes such as the plotting
            % type, the binning window, or the preferred Grid3D topography
            % changes.
            % Once the data is loaded, it is buffered. Repeated calls to
            % this function with the same parameters will thus not incur
            % additional computational cost.
            if nargin<3,plotBinWinMs=obj.plotBinWinMs;end
            if nargin<2,plotType=obj.plotType;end
            obj.unsetError()
            if obj.needToInitSR,obj.initSpikeReader();end
            
            % parse plot type and make it permanent
            if strcmpi(plotType,'default')
                % find default type based on Grid3D
                plotType = obj.getDefaultPlotType();
            elseif isempty(plotType)
                % use current plot type
                plotType = obj.plotType;
            end
            obj.setPlotType(plotType);
            
            % if plotting has not changed since last time, we do not need
            % to do any more work, just reload data from private property
            if sum(obj.plotBinWinMs~=plotBinWinMs) || isempty(obj.spkData)
                obj.needToLoadData = true;
            end
            if ~obj.needToLoadData
                return
            end
            
            % if plot type or bin has changed, we need to re-read and
            % re-format the spike data
            if strcmpi(obj.plotType,'heatmap')
                % heat map uses user-set frameDur for both binning and
                % plotting
                spkBuffer = obj.spkObj.readSpikes(plotBinWinMs);
                
                % reshape according to group dimensions
                numFrames = size(spkBuffer,1);
                spkBuffer = reshape(spkBuffer, numFrames, ...
                    obj.grid3D(1), obj.grid3D(2), ...
                    obj.grid3D(3));
                
                % reshape for plotting
                % \TODO consider the case 1xNxM and Nx1xM
                spkBuffer = permute(spkBuffer,[3 2 4 1]); % Matlab: Y, X
                spkBuffer = reshape(spkBuffer,obj.grid3D(2),[],numFrames);
            elseif strcmpi(obj.plotType,'raster')
                % raster uses user-set frameDur just for plotting
                % binning is not required, use AER instead
                spkBuffer = obj.spkObj.readSpikes(-1);
            else
                obj.throwError(['Unrecognized plot type "' obj.plotType '".'])
                return
            end
            
            % store spike data
            obj.spkData = spkBuffer;
            obj.needToLoadData = false;
        end
        
        function loadDefaultParams(obj)
            % private function to load default parameter values
            obj.spkObj = [];
            obj.spkData = [];
            
            obj.plotType = 'default';
            obj.setSpikeFileAttributes()
            obj.setPlottingAttributes()
            obj.setRecordingAttributes()

            obj.needToInitSR = true;
            obj.needToLoadData = true;
            
            obj.grid3D = -1;
            
            obj.supportedPlotTypes  = {'heatmap', 'raster'};
            obj.supportedErrorModes = {'standard', 'warning', 'silent'};
        end
        
        function pauseOnKeyPressCallback(obj,~,eventData)
            % Callback function to pause plotting
            switch eventData.Key
                case 'p'
                    disp('Paused. Press any key to continue.');
                    waitforbuttonpress;
                case 'q'
                    obj.plotAbortPlotting = true;
            end
        end
        
        function plotFrame(obj, frameNr, plotType, frameDur, dispFrameNr)
            % Private method to display a single frame depending on
            % plotType. This is where the raster plots and heat maps are
            % implemented.
            if nargin<5,dispFrameNr=true;end
            if nargin<4,frameDur=obj.plotBinWinMs;end
            if nargin<3,plotType=obj.plotType;end
            obj.unsetError()
            
            % load data and reshape for plotting if necessary
            obj.loadDataForPlotting(plotType, frameDur);
            
            if strcmpi(obj.plotType,'heatmap')
                maxD = max(obj.spkData(:));
                frame = obj.spkData(:,:,frameNr);
                imagesc(frame, [0 maxD])
                axis image square
                title([obj.name ', rate = [0 , ' ...
                    num2str(maxD*1000/frameDur) ' Hz]'])
                xlabel('nrX')
                ylabel('nrY')
%                 set(gca, 'XTick', 0:obj.grid3D(2):obj.grid3D(2)*obj.grid3D(3))
                
                % if enabled, display the frame number in lower left corner
                if dispFrameNr
                    text(2,size(obj.spkData,1)-1,num2str(frameNr), ...
                        'FontSize',10,'BackgroundColor','white')
                end
            elseif strcmpi(obj.plotType,'raster')
                % find beginning and end of time bin to be displayed
                startTime = (frameNr-1)*frameDur;
                stopTime  = frameNr*frameDur;
                times = obj.spkData(1,:)>=startTime & obj.spkData(1,:)<stopTime;
                plot(obj.spkData(1,times),obj.spkData(2,times),'.k')
                axis([startTime stopTime 1 prod(obj.grid3D)])
                axis square
                title(['Group ' obj.name])
                xlabel('Time (ms)')
                ylabel('Neuron ID')
                set(gca, 'YTick', 0:round(prod(obj.grid3D)/10):prod(obj.grid3D))

                % if enabled, display the frame number in lower left corner
                if dispFrameNr
                    dX = startTime+(stopTime-startTime)*0.05;
                    dY = prod(obj.grid3D)*0.05;
                    text(dX, dY, num2str(frameNr), ...
                        'FontSize',10, 'BackgroundColor','white')
                end
            else
                obj.throwError(['Unrecognized plot type "' obj.plotType '".'])
                return
            end
        end
        
        function throwError(obj, errorMsg, errorMode)
            % GM.throwError(errorMsg, errorMode) throws an error with a
            % specific severity (errorMode). In all cases, obj.errorFlag is
            % set to true and the error message is stored in obj.errorMsg.
            % Depending on errorMode, an error is either thrown as fatal,
            % thrown as a warning, or not thrown at all.
            % If errorMode is not given, obj.errorMode is used.
            if nargin<3,errorMode=obj.errorMode;end
            obj.errorFlag = true;
            obj.errorMsg = errorMsg;
            if strcmpi(errorMode,'standard')
                error(errorMsg)
            elseif strcmpi(errorMode,'warning')
                warning(errorMsg)
            end
        end
        
        function unsetError(obj)
            % unsets error message and flag
            obj.errorFlag = false;
            obj.errorMsg = '';
        end
    end
end