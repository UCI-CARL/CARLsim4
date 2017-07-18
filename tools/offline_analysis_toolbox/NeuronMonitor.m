classdef NeuronMonitor < handle
    % A NeuronMonitor can be used to monitor properties as well as the
    % activity of a specific neuron.
    %
    % A NeuronMonitor will assume that a corresponding neuron state file
    % was created during the CARLsim simulation. 
    %
    % Example usage:
    % >> nM = GroupMonitor('excit','results/');
    % >> nM.plot; % default plotting mode. hit 'p' to pause, 'q' to quit
    % >> nM.setRecordingAttributes('fps',10); % set recording FPS
    % >> nM.recordMovie; % plots heat map and saves as 'movie.avi'
    % >> % etc.
    %
    % Version 6/22/2017
    % Author:Ting-Shuo Chou <tingshuc@uci.edu> 
    %        Michael Beyeler <mbeyeler@uci.edu>
    
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
        nObj;             % NeuronReader object
        nFilePrefix;      % spike file prefix, e.g. "n_"
        nFileSuffix;      % spike file suffix, e.g. ".dat"
        nData;            % buffer for spike data
        
        needToInitNR;       % flag whether we need to init NeuronReader
        needToLoadData;     % flag whether we need to load neuron data

        grid3D;             % Grid3D topography of group
        plotType;           % current plot type
        
		plotTitleName;      % group name for plot title (parsed)
        plotHistNumBins;    % number of histogram bins
        plotHistShowRate;   % flag whether to plot mean rates (Hz) in hist

        plotAbortPlotting;  % flag whether to abort plotting (on-click)
        plotBgColor;        % bg color of plot (for plotting)
        plotDispFrameNr;    % flag whether to display frame number
        plotFPS;            % frames per second for plotting
        plotBinWinMs;       % binning window size (time)
        
        plotInteractiveMode;% flag whether to allow click/key events
        plotStepFrames;     % flag whether to waitforbuttonpress btw frames
        plotStepFramesFW;    % flag whether to make a step forward
        plotStepFramesBW;    % flag whether to make a step backward
        
        recordBgColor;      % bg color of plot (for recording)
        recordFile;         % filename for recording
        recordFPS;          % frames per second for recording
        recordWinSize;      % window size of plot for recording
        
        errorFlag;          % error flag (true if error occured)
        errorMsg;           % error message
    end
    
    
    %% PUBLIC METHODS
    methods
        function obj = NeuronMonitor(name, resultsFolder, errorMode)
            % nM = NeuronMonitor(name,resultsFolder,errorMode) creates a new
            % instance of class NeuronMonitor, which can be used to monitor
            % neuron properties and activity.
            % A NeuronMonitor will assume that a corresponding neuron file
            % has been created during the CARLsim simulation. Otherwise no
            % plotting can be done.
            %
            % NAME           - A string specifying the group which includes
            %                  the neuron. This should be the same as the
            %                  name that was given to the group via
            %                  CARLsim::createGroup.
            % RESULTSFOLDER  - Path to the results directory (where the
            %                  neuron file lives). Default: current dir.
            % ERRORMODE      - Error Mode in which to run NeuronReader. The
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
				obj.setErrorMode('standard');
			else
				obj.setErrorMode(errorMode);
            end
            if nargin<2 || nargin>=2 && strcmpi(resultsFolder,'')
                obj.resultsFolder = '.';
            else
                if strcmpi(resultsFolder(end),filesep)
                    resultsFolder = resultsFolder(1:end-1);
				end
				obj.resultsFolder = resultsFolder;
			end
            if nargin<1
                obj.throwError('No group name given.');
                return
            end
            
            % make sure neuron file is valid
			if ~obj.hasValidNeuronFile()
				obj.throwWarning(['Could not find valid neuron file "' ...
					obj.getNeuronFileName() '". Use ' ...
					'setNeuronFileAttributes to set a proper neuron ' ...
					'file prefix/suffix'])
				return
			end
        end
                
        function delete(obj)
            % destructor, implicitly called
            clear obj.nObj;
        end
        
        function plotType = getDefaultPlotType(obj)
            % plotType = nM.getDefaultPlotType() returns the default
            % plotting type for this group, which is determined by the
            % group's Grid3D topography. For example, a 1D topography will
            % prefer a raster plot, whereas a 2D topography will prefer a
            % heat map.
            % See a list of all currently supported plot types in the help
            % section of GM.plot and in the variable GM.supportedPlotTypes.
            % The plotting type can also be set manually using
            % GM.setDefaultPlotType.
            obj.unsetError()
            obj.initNeuronReader() % required to access Grid3D prop
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
        
        function grid = getGrid3D(obj)
            % grid = GM.getGrid3D() returns the current 3D grid dimensions
            % of the group. Grid3D is a 3-element vector, where the first
            % dimension corresponds to the number of neurons in x
            % direction, the second dimension to y, and the third dimension
            % to z.
            obj.unsetError()
            obj.initNeuronReader();
            
            grid = obj.grid3D;
        end
        
        function nNeur = getNumNeurons(obj)
            % nNeur = GM.getNumNeurons() returns the number of neurons in
            % the group.
            obj.unsetError()
            obj.initNeuronReader();
            
            nNeur = prod(obj.grid3D);
        end
        
        function simDurMs = getSimDurMs(obj)
            % simDurMs = GM.getSimDurMs() returns the estimated simulation
            % duration in milliseconds. This is equivalent to the time
            % stamp of the last spike that occurred.
            obj.unsetError()
			obj.initNeuronReader()
            simDurMs = obj.nObj.getSimDurMs();
        end
        
        function nFile = getNeuronFileName(obj)
            % nFile = GM.getNeuronFileName() returns the name of the
            % spike file according to specified prefix and suffix.
            % Prefix and suffix can be set using GM.setNeuronFileAttributes.
            nFile = [ obj.resultsFolder ... % the results folder
                filesep ...                   % platform-specific separator
                obj.nFilePrefix ...         % something like 'spk_'
                obj.name ...                  % the name of the group
                obj.nFileSuffix ];          % something like '.dat'
        end
                
        function hasValid = hasValidNeuronFile(obj)
            % hasValid = GM.hasValidNeuronFile() determines whether a valid
            % spike file can be found for the group.
            % If no file can be found, the prefix and suffix of the spike
            % file name need to be updated. This can be done using
            % GM.setNeuronFileAttributes.
            obj.unsetError()
            
            nFile = obj.getNeuronFileName();
            nR = NeuronReader(nFile, false, 'silent');
            [obj.errorFlag,obj.errorMsg] = nR.getError();
            hasValid = ~obj.errorFlag;
        end
                
        function plot(obj, plotType, frames, binWindowMs)
            % GM.plot(plotType, frames, binWindowMs) plots the specified
            % frames in the current figure/axes. A list of plotting
            % attributes can be set directly as input arguments.
            %
            % If InteractiveMode is on, press 's' at any time to enter
            % stepping mode. In this mode, pressing the right arrow key
            % will step forward to display the next frame in the list,
            % whereas pressing the left arrow key will step backward to
            % display the last frame in the list. Exit stepping mode by
            % pressing 's' again.
            %
            % The full list of available attributes can be set using
            % GM.setPlottingAttributes.
			%
			% For a list of supported plot types see member variable
			% GM.supportedPlotTypes.
            %
            % PLOTTYPE     - The plotting type to use. If not set, the
            %                default plotting type will be used, which is
            %                determined by the Grid3D topography of the
            %                group.
            %                The following types are currently supported:
			%                 - flowfield A 2D vector field where the
			%                             length of the vector is given as
			%                             the firing rate of the neuron
			%                             times the corresponding vector
			%                             orientation. The latter is given
			%                             by the third grid dimension, z.
			%                             For example Grid3D(10,10,4) plots
			%                             a 10x10 vector flow field,
			%                             assuming that neurons with z=0
			%                             code for direction=0deg, z=1 is
			%                             90deg, z=2 is 180deg, z=3 is
			%                             270deg. Vectors with length
			%                             smaller than 10 % of the max in
			%                             each frame are not shown.
            %                 - heatmap   A topological map of group
            %                             activity where hotter colors mean
            %                             higher firing rate. 
			%                 - histogram A histogram of firing rates.
            %                             Histogram options ('histNumBins'
            %                             and 'histShowRate') can be set
            %                             via GM.setPlottingAttributes.
            %                 - raster    A raster plot with binning window
            %                             binWindowMs
            %                Default: 'default'.
            % FRAMES       - A list of frame numbers. For example,
            %                requesting frames=[1 2 8] will display the
            %                first, second, and eighth frame.
            %                Default: display all frames.
            % BINWINDOWMS  - The binning window (ms) in which the data will
            %                be displayed. Default: 1000.
            if nargin<4,binWindowMs=obj.plotBinWinMs;end
            if nargin<3 || isempty(frames) || sum(frames==-1)>0
                obj.initNeuronReader()
                frames = 1:ceil(obj.nObj.getSimDurMs()/binWindowMs);
            end
            if nargin<2,plotType=obj.plotType;end
            obj.unsetError()
			obj.initNeuronReader()
            
            % verify input
            if ~Utilities.verify(frames,{{'isvector','isnumeric',[1 inf]}})
                obj.throwError('Frames must be a numeric vector e[1,inf]')
                return
            end
            if ~Utilities.verify(binWindowMs,{{'isscalar',[1 inf]}})
                obj.throwError('Frame duration must be a scalar e[1 inf]')
                return
			end
			
			% start plotting in regular mode
			obj.plotAbortPlotting = false;
			obj.plotStepFrames = false;
            
            % reset abort flag, set up callback for key press events
            if obj.plotInteractiveMode
                obj.plotAbortPlotting = false;
                set(gcf,'KeyPressFcn',@obj.pauseOnKeyPressCallback)
            end
            
            % load data and reshape for plotting if necessary
            obj.loadDataForPlotting(plotType, binWindowMs);
            
            % display frame in specified axes
            % use a while loop instead of a for loop so that we can
            % implement stepping backward
            idx = 1;
			while idx <= numel(frames)
				if obj.plotInteractiveMode && obj.plotAbortPlotting
					% user pressed button to quit plotting
					close;
					return;
				end
				
				% plot the frame
				obj.plotFrame(frames(idx), plotType, binWindowMs, ...
					obj.plotDispFrameNr);
				drawnow
				
				% in interactive mode, key press events are active
				if obj.plotInteractiveMode
					if idx>=numel(frames)
						waitforbuttonpress;
						close;
						return;
					else
						if obj.plotStepFrames
							% stepping mode: wait for user input
							while ~obj.plotAbortPlotting ...
									&& ~obj.plotStepFramesFW ...
									&& ~obj.plotStepFramesBW
								pause(0.1)
							end
							if obj.plotStepFramesBW
								% step one frame backward
								idx = max(1, idx-1);
							else
								% step one frame forward
								idx = idx + 1;
							end
							obj.plotStepFramesBW = false;
							obj.plotStepFramesFW = false;
						else
							% wait according to frames per second, then
							% step forward
							pause(1.0/obj.plotFPS)
							idx = idx + 1;
						end
					end
				else
					% wait according to frames per second, then
					% step forward
					pause(1.0/obj.plotFPS)
					idx = idx + 1;
				end
			end
		end
        
        function recordMovie(obj, fileName, plotType, frames, binWindowMs, fps, winSize)
            % GM.recordMovie(fileName, frames, frameDur, fps, winSize)
            % takes an AVI movie of a list of frames using the VIDEOWRITER
            % utility.
            %
            % FILENAME     - A string enclosed in single quotation marks
            %                that specifies the name of the file to create.
            %                Default: 'movie.avi'.
            % PLOTTYPE     - The plotting type to use. If not set, the
            %                default plotting type will be used, which is
            %                determined by the Grid3D topography of the
            %                group. For a list of supported plot types see
            %                member variable GM.supportedPlotTypes.
            % FRAMES       - A list of frame numbers. For example,
            %                requesting frames=[1 2 8] will return the
            %                first, second, and eighth frame in a
            %                width-by-height-by-3 matrix.
            %                Default: -1 (return all frames).
            % BINWINDOWMS  - The binning window (ms) in which the data will
            %                be displayed. Default: 1000.
            % FPS          - Rate of playback for the video in frames per
            %                second. Default: 10.
            % WINSIZE      - A 2-element vector specifying the window size
            %                of the video as width x height in pixels. Set
            %                to [0 0] in order to automatically make the 
            %                movie window fit to the size of the plot
            %                window. Default: [0 0].
            if nargin<7,winSize=obj.recordWinSize;end
            if nargin<6,fps=obj.recordFPS;end
            if nargin<5,binWindowMs=obj.plotBinWinMs;end
            if nargin<4 || isempty(frames) || sum(frames==-1)>0
                obj.initNeuronReader()
                frames = 1:ceil(obj.nObj.getSimDurMs()/binWindowMs);
            end
            if nargin<3,plotType=obj.plotType;end
            if nargin<2,fileName=obj.recordFile;end
            obj.unsetError()
			obj.initNeuronReader()
            
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
            obj.loadDataForPlotting(plotType, binWindowMs);
            
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
                obj.plotFrame(i,plotType,binWindowMs,obj.plotDispFrameNr);
                drawnow
                writeVideo(vidObj, getframe(gcf));
            end
            close(gcf)
            close(vidObj);
            disp(['created file "' fileName '"'])
		end
		
		function setErrorMode(obj, errMode)
			% GM.setErrorMode(errMode) sets the default error mode of the
			% GroupMonitor object to errMode.
			%
			% For a list of supported error mode, see the property
			% GM.supportedErrorModes.
			if nargin<2,errMode='standard';end
			
			if ~obj.isErrorModeSupported(errMode)
				obj.throwError(['errorMode "' errMode '" is ' ...
					'currently not supported. Choose from the ' ...
					'following: ' ...
					strjoin(obj.supportedErrorModes,', ') ...
					'.'], 'standard')
				return
			end
            obj.errorMode = errMode;
		end
        
        function setGrid3D(obj, dim0, dim1, dim2, updDefPlotType)
            % GM.setGrid3D(dim0, dim1, dim2) sets the Grid3D topography of
            % the group. The total number of neurons in the group (width x
            % height x depth) cannot change.
            % If one of the three arguments are set to -1, its value will
            % be automatically adjusted so that the total number of neurons
            % in the group stays the same.
            % DIM0           - Number of neurons in first (x, width)
            %                  dimension.
            % DIM1           - Number of neurons in second (y, height)
            %                  dimension.
            % DIM2           - Number of neurons in thrid (z, depth)
            %                  dimension.
            % UPDDEFPLOTTYPE - A flag whether to update the default plot
            %                  type, given this new Grid3D topography
            %                  arrangement. Default: false
            obj.unsetError()
            obj.initNeuronReader() % so we have accurate grid3D info
            
            if nargin<5,updDefPlotType=false;end
            if nargin<4,dim2=1;end
            if nargin<3,dim1=1;end
            if nargin<2,dim0=obj.getNumNeurons();end
            
            if sum(mod([dim0 dim1 dim2],1)~=0) > 0
                obj.throwError('Grid dimensions must be all integers.');
                return
            end
            if sum([dim0 dim1 dim2]==-1) > 1
                obj.throwError(['There can be at most one dimension ' ...
                    'with value -1.'])
                return
            end
            
            if dim0==-1
                dim0 = round(obj.getNumNeurons()/dim1/dim2);
            elseif dim1==-1
                dim1 = round(obj.getNumNeurons()/dim0/dim2);
            elseif dim2==-1
                dim2 = round(obj.getNumNeurons()/dim0/dim1);
            end
            
            grid = [dim0 dim1 dim2];
            
            % used to rearrange group layout
            if prod(grid) ~= prod(obj.grid3D)
                obj.throwError(['Population size cannot change when ' ...
                    'assigning new Grid3D property (old: ' ...
                    num2str(prod(obj.grid3D)) ', new: ' ...
                    num2str(prod(grid)) ').'])
                return
            end
            
            % if we rearranged the grid layout, we need to re-load the data
            % for plotting (but don't init NeuronReader, otherwise the new
            % setting will be overwritten)
            if logical(sum(obj.grid3D~=grid))
                obj.needToLoadData = true;
            end
            obj.grid3D = grid;
            
            if updDefPlotType
                % set default plot type for this arrangement
                obj.setDefaultPlotType('default');
            end
        end
                
        function setDefaultPlotType(obj, plotType)
            % GM.setDefaultPlotType(plotType) applies a certain plotting type to
            % the group. The default plot type is determined by the Grid3D
            % topography of the group. For example, a 1D topography will
            % prefer a raster plot, whereas a 2D topography will prefer a
            % heatmap.
            %
            % PLOTTYPE    - The plotting type to apply.
            %               The following types are currently supported:
            %                 - flowfield A 2D vector field where the
            %                             length of the vector is given as
            %                             the firing rate of the neuron
            %                             times the corresponding vector
            %                             orientation. The latter is given
            %                             by the third grid dimension, z.
            %                             For example Grid3D(10,10,4) plots
            %                             a 10x10 vector flow field,
            %                             assuming that neurons with z=0
            %                             code for direction=0deg, z=1 is
            %                             90deg, z=2 is 180deg, z=3 is
            %                             270deg. Vectors with length
            %                             smaller than 10 % of the max in
            %                             each frame are not shown.
            %                 - heatmap   A topological map of group
            %                             activity where hotter colors mean
            %                             higher firing rate. 
            %                 - histogram A histogram of firing rates.
            %                             Histogram options ('histNumBins'
            %                             and 'histShowRate') can be set
            %                             via GM.setPlottingAttributes.
            %                 - raster    A raster plot with binning window
            %                             binWindowMs
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
            % that are not available as input arguments to GM.plot.
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
            % HISTNUMBINS     - Number of bins for histogram. Default: 10.
            % HISTSHOWRATE    - A boolean flag to plot mean firing rates
            %                   (Hz) for plotType='histogram' instead of
            %                   mere number of spikes. Default: true.
            % INTERACTIVEMODE - A boolean flag to set InteractiveMode on or
            %                   off. If it is off, key events/FPS/stepping
            %                   will take no effect (helpful if you want to
            %                   take over control yourself). Default: true.
            obj.unsetError()
            
            if isempty(varargin)
                % set default values
                obj.plotDispFrameNr = true;
                obj.plotBgColor = 'w';
                obj.plotFPS = 5;
                obj.plotBinWinMs = 1000;
                obj.plotHistNumBins = 10;
                obj.plotHistShowRate = true;
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
                    case 'histnumbins'
                        % number of bins
                        throwErrNumeric = ~isnumeric(val);
                        reqRange = [0 inf];
                        throwErrOutOfRange = val<reqRange(1) | val>reqRange(2);
                        obj.plotHistNumBins = val;
                    case 'histshowrate'
                        % whether to display frame number
                        throwErrNumeric = ~isnumeric(val) & ~islogical(val);
                        obj.plotHistShowRate = logical(val);
                    case 'interactivemode'
                        % interactive mode
                        throwErrNumeric = ~isnumeric(val) && ~islogical(val);
                        obj.plotInteractiveMode = logical(val);
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
        
        function setNeuronFileAttributes(obj,prefix,suffix)
            % nM.setNeuronFileAttributes(prefix,suffix) defines the naming
            % conventions for the neuron file. The file should reside within
            % resultsFolder, be made of a prefix, the population name, and
            % a suffix.
			% The default will be something like 'results/n_{group}.dat"
			%
			% PREFIX         - A prefix string for the file name. Default:
			%                  'n_'.
			% SUFFIX         - A suffix string for the file name,
			%                  containing the file extension. Default:
			%                  '.dat'
			%
            % Example: files 'results/spk_V1.dat', 'results/spkMT.dat'
            %   -> saveFolder = 'results/'
            %   -> prefix = 'spk_'
            %   -> suffix = '.dat'
            %   -> name of population = 'V1' or 'MT'
            if nargin<3,suffix='.dat';end
            if nargin<2,prefix='n_';end
            obj.unsetError()
            
            % need to re-load if file name changes
            if ~strcmpi(obj.nFilePrefix,prefix) ...
                    || ~strcmpi(obj.nFileSuffix,suffix)
                obj.needToInitNR = true;
                obj.needToLoadData = true;
            end
            obj.nFilePrefix=prefix;
            obj.nFileSuffix=suffix;
        end
    end
    
    %% PRIVATE METHODS
    methods (Hidden, Access = private)
        function initNeuronReader(obj)
            % private method to initialize NeuronReader
            if ~obj.needToInitNR
                return
            end
            obj.unsetError()
            
            nFile = obj.getNeuronFileName();
            obj.nObj = NeuronReader(nFile, false, 'silent');
            
            % make sure spike file is valid
            [obj.errorFlag,obj.errorMsg] = obj.nObj.getError();
            if obj.errorFlag
                obj.throwError(obj.errorMsg)
                return
            end
            obj.grid3D = obj.nObj.getGrid3D();
            obj.needToInitNR = false;
            obj.needToLoadData = true;
        end
        
        function isSupported = isErrorModeSupported(obj, errMode)
            % determines whether an error mode is currently supported
			if nargin<2,errMode='standard';end
            isSupported = sum(ismember(obj.supportedErrorModes,errMode))>0;
        end
        
        function isSupported = isPlotTypeSupported(obj, plotType)
            % determines whether a plot type is currently supported
            isSupported = sum(ismember(obj.supportedPlotTypes,plotType))>0;
        end
        
        function loadDataForPlotting(obj, plotType, plotBinWinMs)
            % Private function to prepare data for plotting.
            % The flag needToInitNR keeps track of changes to the spike
            % file name, so that a new NeuronReader object will be created
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
            if obj.needToInitNR,obj.initNeuronReader();end
            
            % parse plot type and make it permanent
            if strcmpi(plotType,'default')
                % find default type based on Grid3D
                plotType = obj.getDefaultPlotType();
            elseif isempty(plotType)
                % use current plot type
                plotType = obj.plotType;
            end
            obj.setDefaultPlotType(plotType);
            
            % if plotting has not changed since last time, we do not need
            % to do any more work, just reload data from private property
            if sum(obj.plotBinWinMs~=plotBinWinMs) || isempty(obj.nData)
                obj.needToLoadData = true;
            end
            if ~obj.needToLoadData
                return
            end
            
            % if plot type or bin has changed, we need to re-read and
            % re-format the spike data
            if strcmpi(obj.plotType,'flowfield')
                % flowfield uses 3D spike buffer to calc flow field
                spkBuffer = obj.nObj.readSpikes(plotBinWinMs);
                
                % reshape according to group dimensions
                numFrames = size(spkBuffer,1);
                spkBuffer = reshape(spkBuffer, numFrames, ...
                    obj.grid3D(1), obj.grid3D(2), ...
                    obj.grid3D(3));
                
                % find direction of vector component by looking at third
                % dimension of grid, equally spaced in 0:2*pi
                dir = (0:obj.grid3D(3)-1)*2*pi/obj.grid3D(3);
                
                % calc flow field
                flowX = zeros(obj.grid3D(1),obj.grid3D(2),numFrames);
                flowY = zeros(obj.grid3D(1),obj.grid3D(2),numFrames);
                for f=1:numFrames
					tmpX = flowX(:,:,f);
					tmpY = flowY(:,:,f);
                    for d=1:obj.grid3D(3)
						tmpX = tmpX + cos(dir(d)) ...
							.* squeeze(spkBuffer(f,:,:,d));
						tmpY = tmpY + sin(dir(d)) ...
							.* squeeze(spkBuffer(f,:,:,d));
					end
					
					% don't show vector if len < 10 % of max
					maxLen = max(max( sqrt(tmpX.^2+tmpY.^2) ));
					indTooShort = sqrt(tmpX.^2+tmpY.^2) < 0.1*maxLen;
					tmpX(indTooShort) = 0;
					tmpY(indTooShort) = 0;
					flowX(:,:,f) = tmpX;
					flowY(:,:,f) = tmpY;
                end
                clear spkBuffer;
                spkBuffer(:,:,1,:) = flowX;
                spkBuffer(:,:,2,:) = flowY;
            elseif strcmpi(obj.plotType,'heatmap')
                % heat map uses user-set frameDur for both binning and
                % plotting
                spkBuffer = obj.nObj.readSpikes(plotBinWinMs);
                
                % reshape according to group dimensions
                numFrames = size(spkBuffer,1);
                spkBuffer = reshape(spkBuffer, numFrames, ...
                    obj.grid3D(1), obj.grid3D(2), ...
                    obj.grid3D(3));
                
                % reshape for plotting
                % \TODO consider the case 1xNxM and Nx1xM
                spkBuffer = permute(spkBuffer,[3 2 4 1]); % Matlab: Y, X
                spkBuffer = reshape(spkBuffer,obj.grid3D(2),[],numFrames);
            elseif strcmpi(obj.plotType,'histogram')
                % hist uses smaller bin size than frameDur
                histBinMs = plotBinWinMs/obj.plotHistNumBins;
                spkBuffer = sum(obj.nObj.readSpikes(histBinMs),2)';
                
                if obj.plotHistShowRate
                    % compute firing rate (Hz) from number of spikes
                    spkBuffer = 1000.0*spkBuffer/obj.getNumNeurons()/histBinMs;
                end
            elseif strcmpi(obj.plotType,'raster')
                % raster uses user-set frameDur just for plotting
                % binning is not required, use AER instead
                spkBuffer = obj.nObj.readSpikes(-1);
            else
                obj.throwError(['Unrecognized plot type "' obj.plotType '".'])
                return
            end
            
            % store spike data
            obj.nData = spkBuffer;
            obj.needToLoadData = false;
        end
        
        function loadDefaultParams(obj)
            % private function to load default parameter values
            obj.nObj = [];
            obj.nData = [];
            
            obj.plotType = 'default';
            obj.setNeuronFileAttributes()
            obj.setPlottingAttributes()
            obj.setRecordingAttributes()
            
            obj.needToInitNR = true;
            obj.needToLoadData = true;
            
            obj.plotStepFrames = false;
            obj.plotStepFramesFW = false;
            obj.plotStepFramesBW = false;
			
			% for the group name in plot titles, mask underscores so that
			% they're not interpreted as LaTeX; except for '_{', which
			% should be interpreted as LaTeX for lowerscript
			obj.plotTitleName = regexprep(strrep(obj.name, '_', '\_'),'\\_{','_{');
            
            obj.grid3D = -1;
            
            obj.supportedPlotTypes  = {'flowfield', 'heatmap', ...
                'histogram', 'raster'};
            obj.supportedErrorModes = {'standard', 'warning', 'silent'};
			
			% disable backtracing for warnings and errors
			warning off backtrace
        end
        
        function pauseOnKeyPressCallback(obj,~,eventData)
            % Callback function to pause plotting
            switch eventData.Key
                case 'p'
                    disp('Paused. Press any key to continue.');
                    waitforbuttonpress;
                case 'q'
                    obj.plotStepFrames = false;
                    obj.plotAbortPlotting = true;
                case 's'
                    obj.plotStepFrames = ~obj.plotStepFrames;
                    if obj.plotStepFrames
                        disp(['Entering Stepping mode. Step forward ' ...
                            'with right arrow key, step backward with ' ...
                            'left arrow key.']);
                    end
                case 'leftarrow'
                    if obj.plotStepFrames
                        obj.plotStepFramesBW = true;
                    end
                case 'rightarrow'
                    if obj.plotStepFrames
                        obj.plotStepFramesFW = true;
                    end
                otherwise
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
			if numel(size(obj.nData))==4 && frameNr > size(obj.nData,4)
				warning(['frameNr=' num2str(frameNr) ' exceeds ' ...
					'number of frames (' num2str(size(obj.nData,4)) ...
					')'])
				return;
			end
            
            if strcmpi(obj.plotType,'flowfield')
                frame = obj.nData(:,:,:,frameNr);
                [x,y] = meshgrid(1:obj.grid3D(1),1:obj.grid3D(2));
                quiver(x',y',frame(:,:,1)',frame(:,:,2)');
                axis equal
                axis([1 max(2,size(frame,1)) 1 max(2,size(frame,2))])
                title(['Group ' obj.plotTitleName])
                xlabel('nrX')
                ylabel('nrY')
                set(gca, 'XTick', [1 obj.grid3D(1)/2 obj.grid3D(1)])
                set(gca, 'YTick', [1 obj.grid3D(2)/2 obj.grid3D(2)])

                % if enabled, display the frame number in lower left corner
                if dispFrameNr
                    text(2, 2, num2str(frameNr), 'FontSize', 10, ...
                        'BackgroundColor','white')
                end
            elseif strcmpi(obj.plotType,'heatmap')
                maxD = max(obj.nData(:));
                frame = obj.nData(:,:,frameNr);
                imagesc(frame, [0 maxD])
                axis image equal
                axis([1 obj.grid3D(1)*obj.grid3D(3) 1 obj.grid3D(2)])
                title(['Group ' obj.plotTitleName ', rate = [0 , ' ...
                    num2str(maxD*1000/frameDur) ' Hz]'])
                xlabel('nrX')
                ylabel('nrY')
%                 set(gca, 'XTick', 0:obj.grid3D(2):obj.grid3D(2)*obj.grid3D(3))
                
                % if enabled, display the frame number in lower left corner
                if dispFrameNr
                    text(2,size(obj.nData,1)-1,num2str(frameNr), ...
                        'FontSize',10,'BackgroundColor','white')
                end
            elseif strcmpi(obj.plotType,'histogram')
                histBinMs = frameDur/obj.plotHistNumBins;
                edges = (frameNr-1)*frameDur + histBinMs/2 : histBinMs ...
                    : frameNr*frameDur;
                dataStart = (frameNr-1)*obj.plotHistNumBins+1;
                dataEnd = frameNr*obj.plotHistNumBins;
                data = obj.nData(1, dataStart:dataEnd);
                bar(edges, data, 1.0)
                xlabel('time slice')
                if obj.plotHistShowRate
                    ylabel('Mean firing rate (Hz)')
                else
                    ylabel('number of spikes')
                end
                title(['Group ' obj.plotTitleName])
%                 axis square
                a=axis; maxD = max(data(:))*1.2;
                axis([a(1) a(2) 0 max(maxD,1)])
            elseif strcmpi(obj.plotType,'raster')
                % find beginning and end of time bin to be displayed
                startTime = (frameNr-1)*frameDur;
                stopTime  = frameNr*frameDur;
                times = obj.nData(1,:)>=startTime & obj.nData(1,:)<stopTime;
                plot(obj.nData(1,times),obj.nData(2,times),'.k')
                axis image square
                axis([startTime stopTime -1 prod(obj.grid3D)])
                title(['Group ' obj.plotTitleName])
                xlabel('Time (ms)')
                ylabel('Neuron ID')
                set(gca, 'YTick', 0:round(prod(obj.grid3D)/10):prod(obj.grid3D)+1)

                % if enabled, display the frame number in lower left corner
                if dispFrameNr
                    dX = startTime+(stopTime-startTime)*0.05;
                    dY = prod(obj.grid3D)*0.05-1;
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
        
        function throwWarning(obj, errorMsg, errorMode)
            % THROWWARNING(errorMsg, errorMode) throws a warning with a
            % specific severity (errorMode).
            % If errorMode is not given, obj.errorMode is used.
            if nargin<3,errorMode=obj.errorMode;end
            
            if strcmpi(errorMode,'standard')
                warning(errorMsg)
            elseif strcmpi(errorMode,'warning')
                disp(errorMsg)
            end
		end
		
		function unsetError(obj)
            % unsets error message and flag
            obj.errorFlag = false;
            obj.errorMsg = '';
        end
    end
end