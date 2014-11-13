classdef ConnectionMonitor < handle
    % A ConnectionMonitor can be used to monitor properties of a specific
    % connection (preGrp -> postGrp).
    
    
    %% PROPERTIES
    
    % public
    properties (SetAccess = private)
        connId;             % connection ID
        grpPreName;         % name of pre-synaptic group
        grpPostName;        % name of post-synaptic group
        resultsFolder;      % results folder
        errorMode;          % program mode for error handling
        supportedErrorModes;% supported error modes
        supportedPlotTypes; % cell array of supported plot types
%     end
    
    % private
%     properties (Hidden, Access = private)
        connObj;            % ConnectionReader object
        connFilePrefix;     % conn file prefix, e.g. "conn"
        connFileSuffix;     % conn file suffix, e.g. ".dat"
        connFileSeparator;  % conn file separator, e.g. '_'
        weights;
        
        needToInitCR;       % flag whether we need to init ConnectionReader
        needToLoadData;     % flag whether we need to load connect data
        
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
        function obj = ConnectionMonitor(grpPreName, grpPostName, ...
                resultsFolder, errorMode)
            obj.grpPreName = grpPreName;
            obj.grpPostName = grpPostName;
            obj.unsetError();
            obj.loadDefaultParams();
            
            if nargin<4
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
            if nargin<3
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
            clear obj.connObj;
        end
        
        function plotType = getDefaultPlotType(obj)
            % plotType = CM.getDefaultPlotType() returns the default
            % plotting type for this connection.
            % Default type for small weight matrices is 'heatmap', and for
            % large ones it is 'histogram'.
            % See a list of all currently supported plot types in the help
            % section of CM.plot and in the variable CM.supportedPlotTypes.
            % The plotting type can also be set manually using the method
            % CM.setPlotType.
            obj.unsetError()
            obj.initConnectionReader() % required to access dimensions
            
            % find weight matrix size
            nElem = obj.connObj.getNumNeuronsPre() ...
                * obj.connObj.getNumNeuronsPost();
            
            % if size is too big, show histogram, else weight matrix
            if nElem >= 1e7
                plotType = 'histogram';
            else
                plotType = 'heatmap';
            end
        end
        
        function [errFlag,errMsg] = getError(obj)
            % [errFlag,errMsg] = CM.getError() returns the current error
            % status.
            % If an error has occurred, errFlag will be true, and the
            % message can be found in errMsg.
            errFlag = obj.errorFlag;
            errMsg = obj.errorMsg;
        end
        
        function connFile = getConnectFileName(obj)
            % connFile = CM.getConnectFileName() returns the name of the
            % connect file according to specified prefix, suffix, and
            % separator.
            % The full file name should be [prefix sep grpPreName sep
            % grpPostName suffix]
            % Prefix, suffix, and separator can be set using the function
            % CM.setConnectFileAttributes.
            connFile = [ obj.resultsFolder ... % the results folder
                filesep ...                   % platform-specific separator
                obj.connFilePrefix ...        % something like 'conn'
                obj.connFileSeparator ...     % something like '_'
                obj.grpPreName ...            % name of the pre-group
                obj.connFileSeparator ...
                obj.grpPostName ...           % name of the post-group
                obj.connFileSuffix ];         % something like '.dat'
        end
        
        function hasValid = hasValidConnectFile(obj)
            % hasValid = CM.hasValidConnectFile() determines whether a
            % valid connect file can be found for the connection.
            % If no file can be found, the prefix, suffix, and separator of
            % the connect file name need to be updated. This can be done
            % using CM.setConnectFileAttributes.
            obj.unsetError()
            
            connFile = obj.getConnectFileName();
            CR = ConnectionReader(connFile, 'silent');
            [errFlag,~] = SR.getError();
            hasValid = ~errFlag;
        end
        
        function plot(obj, plotType, frames, stepFrames)
            % CM.plot(plotType, frames, stepFrames) plots the specified
            % frames (or snapshots) in the current figure/axes. A list of
            % plotting attributes can be set directly as input arguments.
            %
            % The full list of available attributes can be set using the
            % method CM.setPlottingAttributes.
            %
            % PLOTTYPE     - The plotting type to use. If not set, the
            %                default plotting type will be used.
            %                The following types are currently supported:
            %                 - heatmap   a topological map of the weight
            %                             matrix where hotter colors mean
            %                             higher firing rate (first dim=pre
            %                             and second dim=post).
            %                 - histogram a histogram of all weight values
            %                Default: 'default'.
            % FRAMES       - A list of frame (or snapshot) numbers. For
            %                example, requesting frames=[1 2 8] will
            %                display the first, second, and eighth frame.
            %                Default: display all frames.
            % STEPFRAMES   - A boolean flag that indicates whether to wait
            %                for user input (button press) before
            %                displaying the next frame. Default: false.
            if nargin<4,stepFrames=obj.plotStepFrames;end
            if nargin<3 || isempty(frames) || frames==-1
                obj.initConnectionReader()
                frames = 1:ceil(obj.spkObj.getNumSnapshots());
            end
            if nargin<2,plotType=obj.plotType;end
            obj.unsetError()
            
            % verify input
            if ~Utilities.verify(frames,{{'isvector','isnumeric',[1 inf]}})
                obj.throwError('Frames must be a numeric vector e[1,inf]')
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
            
            % load data and reshape for plotting
            obj.loadDataForPlotting(plotType);
            
            % display frames in specified axes
            for i=frames
                if obj.plotInteractiveMode && obj.plotAbortPlotting
                    % user pressed button to quit plotting
                    obj.plotAbortPlotting = false;
                    close;
                    return
                end
                
                obj.plotFrame(i,plotType,obj.plotDispFrameNr);
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
        
        function recordMovie(obj, fileName, frames, fps, winSize)
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
        
        function setConnectFileAttributes(obj,prefix,suffix,separator)
            % CM.setConnectFileAttributes(prefix,suffix,separator)
            % Defines the naming conventions for connect files. They should
            % all reside within resultsFolder (specified in constructor),
            % and be made of a common prefix, the name of pre and post
            % groups, and a common suffix.
            % Example: file 'results/conn_V1_MT.dat'
            %   -> resultsFolder = 'results/'
            %   -> prefix = 'conn'
            %   -> separator = '_'
            %   -> suffix = '.dat'
            %   -> name of groups = 'V1' (pre) and 'MT' (post)
            if nargin<4,separator='_';end
            if nargin<3,suffix='.dat';end
            if nargin<2,prefix='conn';end
            obj.unsetError()
            
            % need to re-load if file name changes
            if ~strcmpi(obj.connFilePrefix,prefix) ...
                    || ~strcmpi(obj.connFileSuffix,suffix) ...
                    || ~strcmpi(obj.connFileSeparator,separator)
                obj.needToInitCR = true;
                obj.needToLoadData = true;
            end
            obj.connFilePrefix=prefix;
            obj.connFileSuffix=suffix;
            obj.connFileSeparator=separator;
        end
        
    end
    
    %% PRIVATE METHODS
    methods (Hidden, Access = private)
        function initConnectionReader(obj)
        end
        
        function isSupported = isErrorModeSupported(obj, errMode)
            % determines whether an error mode is currently supported
            isSupported = sum(ismember(obj.supportedErrorModes,errMode))>0;
        end
        
        function isSupported = isPlotTypeSupported(obj, plotType)
            % determines whether a plot type is currently supported
            isSupported = sum(ismember(obj.supportedPlotTypes,plotType))>0;
        end
        
        function loadDefaultParams(obj)
            % private function to load default parameter values
            obj.connObj = [];
            obj.weights = [];
            
            obj.plotType = 'default';
            obj.setConnectFileAttributes()
            obj.setPlottingAttributes()
            obj.setRecordingAttributes()
            
            obj.needToInitCR = true;
            obj.needToLoadData = true;
            
            obj.supportedPlotTypes = {'heatmap','histogram'};
            obj.supportedErrorModes = {'standard','warning','silent'};
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
        
        function plotFrame(obj, frameNr, plotType, dispFrameNr)
            % Private method to display a single frame depending on
            % plotType. This is where the raster plots and heat maps are
            % implemented.
            if nargin<4,dispFrameNr=true;end
            if nargin<3,plotType=obj.plotType;end
            obj.unsetError()
            
            % load data and reshape for plotting if necessary
            obj.loadDataForPlotting(plotType);
            
            if strcmpi(obj.plotType,'heatmap')
                maxD = max(obj.weights(:));
                frame = reshape(obj.weights(frameNr,:),obj.nNeurPre,obj.nNeurPost);
                imagesc(frame, [0 maxD])
                axis image square
                title([obj.grpNamePre '->' obj.grpNamePost ', t=' ...
                    num2str(obj.timeStamps(frameNr)) 'ms, wt = [0 , ' ...
                    num2str(maxD) ']')
                xlabel('nrPre')
                ylabel('nrPost')
                
                % if enabled, display the frame number in lower left corner
                if dispFrameNr
                    text(2,size(obj.spkData,1)-1,num2str(frameNr), ...
                        'FontSize',10,'BackgroundColor','white')
                end
            elseif strcmpi(obj.plotType,'histogram')
                error('Not yet implemented')
                return
            else
                obj.throwError(['Unrecognized plot type "' obj.plotType '".'])
                return
            end
        end
        
        function throwError(obj, errorMsg, errorMode)
            % CM.throwError(errorMsg, errorMode) throws an error with a
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