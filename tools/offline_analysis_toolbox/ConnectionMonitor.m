classdef ConnectionMonitor < handle
	% A ConnectionMonitor can be used to monitor properties of a specific
	% connection (preGrp -> postGrp).
	%
	% A ConnectionMonitor will assume that a corresponding connect file has
	% been created during the CARLsim simulation.
	%
	% Example usage:
	% >> CM = ConnectionMonitor('excit','inhib','results/');
	% >> CM.plot; % hit 'p' to pause, 'q' to quit
	% >> CM.setDefaultPlotType('histogram'); % switch to hist mode
	% >> CM.setRecordingAttributes('fps',2); % set recording FPS
	% >> CM.recordMovie; % plots hist and saves as 'movie.avi'
	% >> % etc.
	%
	% Version 3/11/2015
	% Author: Michael Beyeler <mbeyeler@uci.edu>
	
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
	end
	
	% private
	properties (Hidden, Access = private)
		CR;                 % ConnectionReader object
		connFilePrefix;     % conn file prefix, e.g. "conn"
		connFileSuffix;     % conn file suffix, e.g. ".dat"
		connFileSeparator;  % conn file separator, e.g. '_'
		weights;            % weight matrix for all snapshots
		timeStamps;         % timestamps for all snapshots
		
		needToInitCR;       % flag whether we need to init ConnectionReader
		needToLoadData;     % flag whether we need to load connect data
		
		plotType;           % current plot type
		
		plotHistData;       % weight matrices binned for hist
		plotHistBins;       % edges for hist
		plotHistNumBins;    % number of histogram bins
		
		plotTitlePreName;	% name of pre-group for plot titles (parsed)
		plotTitlePostName;  % name of post-group for plot titles (parsed)
		
		plotAbortPlotting;  % flag whether to abort plotting (on-click)
		plotBgColor;        % bg color of plot (for plotting)
		plotDispFrameNr;    % flag whether to display frame number
		plotFPS;            % frames per second for plotting
		plotSubplotsPerFig; % max number of subplots per figure
		
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
				if strcmpi(resultsFolder(end),filesep)
					resultsFolder = resultsFolder(1:end-1);
				end
				obj.resultsFolder = resultsFolder;
			end
			if nargin<1
				obj.throwError('No group name given.');
				return
			end

            % make sure connect file is valid
			if ~obj.hasValidConnectFile()
				obj.throwWarning(['Could not find valid connect file "' ...
					obj.getConnectFileName() '". Use ' ...
					'setConnectFileAttributes to set a proper connect ' ...
					'file prefix/suffix/separator'])
				return
			end
		end
		
		function delete(obj)
			% destructor, implicitly called
			clear obj.CR;
		end
		
		function plotType = getDefaultPlotType(obj)
			% plotType = CM.getDefaultPlotType() returns the default
			% plotting type for this connection.
			% Default type for small weight matrices is 'heatmap', and for
			% large ones it is 'histogram'.
			% See a list of all currently supported plot types in the help
			% section of CM.plot and in the variable CM.supportedPlotTypes.
			% The plotting type can also be set manually using the method
			% CM.setDefaultPlotType.
			obj.unsetError()
			obj.initConnectionReader() % required to access dimensions
			
			% find weight matrix size
			nElem = obj.CR.getNumNeuronsPre() ...
				* obj.CR.getNumNeuronsPost();
			
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
		
		function grid = getGrid3DPre(obj)
            % grid = CM.getGrid3DPre() returns the current 3D grid dimensions
            % of the pre-synaptic group.
            % Grid3D is a 3-element vector, where the first
            % dimension corresponds to the number of neurons in x
            % direction, the second dimension to y, and the third dimension
            % to z.
            obj.unsetError()
            obj.initConnectionReader()
            
            grid = obj.CR.getGrid3DPre();
        end

		function grid = getGrid3DPost(obj)
            % grid = CM.getGrid3DPost() returns the current 3D grid dimensions
            % of the pre-synaptic group.
            % Grid3D is a 3-element vector, where the first
            % dimension corresponds to the number of neurons in x
            % direction, the second dimension to y, and the third dimension
            % to z.
            obj.unsetError()
            obj.initConnectionReader()
            
            grid = obj.CR.getGrid3DPost();
        end

		function xyz = getNeuronLocation3DPre(obj,neurId)
			% xyz = CM.getNeronLocation3DPre(neurId) returns the 3D coordinates
			% of the neurId-th neuron in the pre-synaptic group (1-indexed).
			% The 3D coordinates of the neuron are determined by the Grid3D
			% dimensions of the group.
			obj.unsetError()
			obj.initConnectionReader()

			if ~Utilities.verify(neurId,{{'isnumeric',[1 obj.getNumNeuronsPre()]}})
				obj.throwError(['Neuron ID must be in the range e[1,' num2str(obj.getNumNeuronsPre()) '']')
				return
			end

			neurId = neurId - 1;
			grid3D = obj.CR.getGrid3DPre();
			xyz(1) = mod(neurId, grid3D(1));
			xyz(2) = mod( floor(neurId/grid3D(1)), grid3D(2) );
			xyz(3) = floor(neurId / (grid3D(1)*grid3D(2)));
		end

		function xyz = getNeuronLocation3DPost(obj,neurId)
			% xyz = CM.getNeronLocation3DPre(neurId) returns the 3D coordinates
			% of the neurId-th neuron in the pre-synaptic group (1-indexed).
			% The 3D coordinates of the neuron are determined by the Grid3D
			% dimensions of the group.
			obj.unsetError()
			obj.initConnectionReader()

			if ~Utilities.verify(neurId,{{'isnumeric',[1 obj.getNumNeuronsPost()]}})
				obj.throwError(['Neuron ID must be in the range e[1,' num2str(obj.getNumNeuronsPost()) '']')
				return
			end

			neurId = neurId - 1;
			grid3D = obj.CR.getGrid3DPost();
			xyz(1) = mod(neurId, grid3D(1));
			xyz(2) = mod( floor(neurId/grid3D(1)), grid3D(2) );
			xyz(3) = floor(neurId / (grid3D(1)*grid3D(2)));
		end

		function nSnap = getNumSnapshots(obj)
			% nSnap = CM.getNumSnapshots() returns the number of snapshots
			% that have been recorded.
			obj.unsetError()
			obj.initConnectionReader()
			nSnap = obj.CR.getNumSnapshots();
		end
		
		function snapshots = getSnapshots(obj, frames)
			% snapshots = CM.getSnapshots(frames) returns all the snapshots
			% specified by the list FRAMES in a 3D weight matrix: where the
			% first dimension is the number of presynaptic neurons, the
			% second dimension is the number of postsynaptic neurons, and
			% the third dimension is the snapshot number. For example,
			% snapshots(3,4,1) will return the weight from preNeurId==3 to
			% postNeurId==4 for the first recorded snapshot.
			%
			% FRAMES       - A list of frame (or snapshot) numbers. For
			%                example, requesting frames=[1 2 8] will
			%                display the first, second, and eighth frame.
			%                Default: display all frames.
			obj.unsetError()
			obj.initConnectionReader()
			if nargin<2 || isempty(frames) || sum(frames==-1)>0
				frames = 1:ceil(obj.CR.getNumSnapshots());
			end
			
			% verify input
			if ~Utilities.verify(frames,{{'isvector','isnumeric',[1 inf]}})
				obj.throwError('Frames must be a numeric vector e[1,inf]')
				return
			end
			if sum(frames>=1)~=numel(frames) ...
					|| sum(frames<=obj.CR.getNumSnapshots())~=numel(frames)
				obj.throwError(['Frame number must be e[1,' ...
					num2str(obj.CR.getNumSnapshots()) ']'])
				return
			end
			
			% read all the timestamps and weights
			[~,snapshots] = obj.CR.readWeights(frames);
			snapshots = reshape(snapshots, ...
				numel(frames), ...
				obj.CR.getNumNeuronsPre(), ...
				obj.CR.getNumNeuronsPost());
			
			% reshape
			snapshots = permute(snapshots,[2 3 1]); % X Y T
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
			[errFlag,~] = CR.getError();
			hasValid = ~errFlag;
		end
		
		function plot(obj, plotType, frames, neurons)
			% CM.plot(plotType, frames, neurons) plots the specified
			% frames (or snapshots) in the current figure/axes. A list of
			% plotting attributes can be set directly as input arguments.
			%
			% If InteractiveMode is on, press 's' at any time to enter
			% stepping mode. In this mode, pressing the right arrow key
			% will step forward to display the next frame in the list,
			% whereas pressing the left arrow key will step backward to
			% display the last frame in the list. Exit stepping mode by
			% pressing 's' again.
			%
			% The full list of available attributes can be set using the
			% method CM.setPlottingAttributes.
			%
			% PLOTTYPE     - The plotting type to use. If not set, the
			%                default plotting type will be used.
			%                The following types are currently supported:
			%                 - heatmap         A topological map of the
			%                                   weight matrix where hotter
			%                                   colors mean higher firing
			%                                   rate (first dim=pre and
			%                                   second dim=post).
			%                 - histogram       A histogram of all weight
			%                                   values.
			%                 - receptivefield  A spatial map of a neuron's
			%                                   receptive field (post
			%                                   group). Use input argument
			%                                   NEURONS to specify a list
			%                                   of post-neurons.
			%                 - responsefield   A spatial map of a neuron's
			%                                   response field (pre group).
			%                                   Use input argument NEURONS
			%                                   to specify a list of
			%                                   pre-neurons.
			%                Default: 'default'.
			% FRAMES       - A list of frame (or snapshot) numbers. For
			%                example, requesting frames=[1 2 8] will
			%                display the first, second, and eighth frame.
			%                Default: display all frames.
			% NEURONS      - A list of neuron IDs for which to generate
			%                receptive fields or response fields. For
			%                example, requestion neurons=[1 2 8] will
			%                display the receptive (response) field of the
			%                first, second, and eight neuron in post (pre).
			%                Default: display all neurons.
			obj.unsetError()
			obj.initConnectionReader()
			if nargin<2,plotType=obj.plotType;end
			if nargin<3 || isempty(frames) || sum(frames==-1)>0
				frames = 1:ceil(obj.CR.getNumSnapshots());
			end
			if nargin<4 || isempty(neurons) ...
					|| numel(neurons)==1 && neurons==-1
				if strcmpi(plotType,'receptivefield')
					neurons = 1:obj.CR.getNumNeuronsPost();
				elseif strcmpi(plotType,'responsefield')
					neurons = 1:obj.CR.getNumNeuronsPre();
				else
					neurons = -1;
				end
			end
			
			% verify input
			if strcmpi(plotType,'receptivefield')
				if ~Utilities.verify(neurons,{{'isvector','isnumeric', ...
						[1 obj.CR.getNumNeuronsPost()]}})
					obj.throwError(['Neurons must be a numeric vector ' ...
						'e[1,' num2str(obj.CR.getNumNeuronsPost()) ']'])
					return
				end
			elseif strcmpi(plotType,'responsefield')
				if ~Utilities.verify(neurons,{{'isvector','isnumeric', ...
						[1 obj.CR.getNumNeuronsPre()]}})
					obj.throwError(['Neurons must be a numeric vector ' ...
						'e[1,' num2str(obj.CR.getNumNeuronsPre()) ']'])
					return
				end
			end
			if ~Utilities.verify(frames,{{'isvector','isnumeric',[1 inf]}})
				obj.throwError('Frames must be a numeric vector e[1,inf]')
				return
			end
			
			% reset abort flag, set up callback for key press events
			if obj.plotInteractiveMode
				obj.plotAbortPlotting = false;
				set(gcf,'KeyPressFcn',@obj.pauseOnKeyPressCallback)
			end
			
			% load data and reshape for plotting
			obj.loadDataForPlotting(plotType);
			
			% display frame in specified axes
			% use a while loop instead of a for loop so that we can
			% implement stepping backward
			idx = 1;
			while idx <= numel(frames)
				if obj.plotInteractiveMode && obj.plotAbortPlotting
					% user pressed button to quit plotting
					obj.plotAbortPlotting = false;
					close;
					return
				end
				
				% plot the frame
				obj.plotFrame(frames(idx), plotType, neurons, obj.plotDispFrameNr);
				drawnow 
				
				% in interactive mode, key press events are active
				if obj.plotInteractiveMode
					if idx==numel(frames)
						try
							waitforbuttonpress;
						catch
						end
						idx = idx + 1; % needed to exit
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
			if obj.plotInteractiveMode,close all;end
		end
				
		function recordMovie(obj, fileName, plotType, frames, fps, winSize)
			% CM.recordMovie(fileName, frames, fps, winSize) takes an AVI
			% movie of a list of frames using the VIDEOWRITER utility.
			%
			% FILENAME     - A string enclosed in single quotation marks
			%                that specifies the name of the file to create.
			%                Default: 'movie.avi'.
            % PLOTTYPE     - The plotting type to use. If not set, the
            %                default plotting type will be used, which is
            %                determined by the Grid3D topography of the
            %                group. For a list of supported plot types see
            %                member variable CM.supportedPlotTypes.
			% FRAMES       - A list of frame numbers. For example,
			%                requesting frames=[1 2 8] will return the
			%                first, second, and eighth frame in a
			%                width-by-height-by-3 matrix.
			%                Default: -1 (return all frames).
			% FPS          - Rate of playback for the video in frames per
			%                second. Default: 10.
			% WINSIZE      - A 2-element vector specifying the window size
			%                of the video as width x height in pixels. Set
			%                to [0 0] in order to automatically make the
			%                movie window fit to the size of the plot
			%                window. Default: [0 0].
			if nargin<6,winSize=obj.recordWinSize;end
			if nargin<5,fps=obj.recordFPS;end
			if nargin<4 || isempty(frames) || sum(frames==-1)>0
				obj.initConnectionReader()
				frames = 1:ceil(obj.CR.getNumSnapshots);
			end
			if nargin<3,plotType=obj.plotType;end
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
			if ~Utilities.verify(fps,{{'isscalar',[0.01 100]}})
				obj.throwError('FPS must be in range [0.01,100]');return
			end
			if ~Utilities.verify(winSize,{{'isvector','isnumeric',[0 inf]}})
				obj.throwError(['Window size must be a numeric vector ' ...
					'with values > 0']);return
			end
			
			% load data and reshape for plotting if necessary
			obj.loadDataForPlotting(plotType);
			
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
				obj.plotFrame(i,plotType,obj.plotDispFrameNr);
				drawnow
				writeVideo(vidObj, getframe(gcf));
			end
			close(gcf)
			close(vidObj);
			disp(['created file "' fileName '"'])
		end
		
		function setDefaultPlotType(obj, plotType)
			% CM.setDefaultPlotType(plotType) applies a certain plotting type to
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
			if nargin<2,plotType='default';end
			
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
			% CM.setPlottingAttributes(varargin) can be used to set default
			% settings that will apply to all activity plots.
			% This function provides control over additional attributes
			% that are not available as input arguments to CM.plot or
			% CM.plotFrame.
			% CM.setPlottingAttributes('propertyName1',value1,...) sets the
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
			% HISTNUMBINS     - Number of bins for histogram. Default: 20.
			% INTERACTIVEMODE - A boolean flag to set InteractiveMode on or
			%                   off. If it is off, key events/FPS/stepping
			%                   will take no effect (helpful if you want to
			%                   take over control yourself). Default: true.
			% SUBPLOTSPERFIG  - Maximum number of subplots per figure.
			%					Default: 80.
			obj.unsetError()
			
			if isempty(varargin)
				% set default values
				obj.plotDispFrameNr = true;
				obj.plotBgColor = 'w';
				obj.plotFPS = 5;
				obj.plotHistNumBins = 20;
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
					case 'interactivemode'
						% interactive mode
						throwErrNumeric = ~isnumeric(val) && ~islogical(val);
						obj.plotInteractiveMode = logical(val);
					case 'subplotsperfig'
						reqRange = [1 100];
						throwErrNumeric = ~isnumeric(val) ...
							&& val<reqRange(1) || val>reqRange(2);
						obj.plotSubplotsPerFig = val;
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
			% CM.setRecordingAttributes(varargin) can be used to set
			% default settings that will apply to all activity recordings.
			% This function provides control over additional attributes
			% that are not available as input arguments to CM.recordMovie.
			% CM.setRecordingAttributes('propertyName1',value1,...) sets
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
				obj.recordFPS = 2;
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
        function [nrP, nrR, nrC] = findPlotLayout(obj, numSubPlots)
            % given a total number of subplots, what should be optimal
            % number of rows and cols in the figure?
			
			% plot should have at most 100 subplots
			nrP = ceil(numSubPlots/obj.plotSubplotsPerFig);
			if nrP>10
				obj.throwWarning(['Can plot at most 10 figures, ' ...
					num2str(nrP) ' requested.'])
				nrP = 10;
			end
			numSubPlotsPerPage = min(numSubPlots,obj.plotSubplotsPerFig);
            nrR = floor(sqrt(numSubPlotsPerPage));
            nrC = ceil(numSubPlotsPerPage*1.0/nrR);
		end
		
		function initConnectionReader(obj)
			% private method to initialize ConnectionReader
			obj.unsetError()
			
			connFile = obj.getConnectFileName();
			obj.CR = ConnectionReader(connFile,'silent');
			
			% make sure connect file is valid
			[errFlag,errMsg] = obj.CR.getError();
			if errFlag
				obj.throwError(errMsg)
				return
			end
			obj.needToInitCR = false;
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
		
		function loadDataForPlotting(obj, plotType)
			% private method to prepare data for plotting
			% The flag needToInitCR keeps track of changes to the connect
			% file name, so that a new ConnectionReader object will be
			% created if the path to the connect file changes.
			% The flag needToLoadData keeps track of plotting settings, so
			% that the data is reloaded if attributes such as the plotting
			% type changes.
			% Once the data is loaded, it is buffered. Repeated calls to
			% this function with the same parameters will thus not incur
			% additional computational cost.
			if nargin<2,plotType=obj.plotType;end
			obj.unsetError();
			if obj.needToInitCR,obj.initConnectionReader();end
			
			% if we have never run this function (empty weights) or if the
			% requested plottype is not what it was before, we need to
			% reload data
			if ~strcmpi(plotType,'default') && ~strcmpi(obj.plotType,plotType) ...
					|| isempty(obj.weights) || isempty(obj.timeStamps)
				obj.needToLoadData = true;
			end
			
			% if we don't need to load, exit
			if ~obj.needToLoadData
				return
			end
			
			% parse plot type and make it permanent
			if strcmpi(plotType,'default')
				plotType = obj.getDefaultPlotType();
			elseif isempty(plotType)
				if strcmpi(obj.plotType,'default')
					plotType = obj.getDefaultPlotType();
				else
					% use current plot type
					plotType = obj.plotType;
				end
			end
			obj.setDefaultPlotType(plotType);
			
			% read all the timestamps and weights
			[obj.timeStamps,obj.weights] = obj.CR.readWeights();

			% re-format the data
			if strcmpi(obj.plotType,'heatmap') ...
					|| strcmpi(obj.plotType,'receptivefield') ...
					|| strcmpi(obj.plotType,'responsefield')
				% reshape to 3-D matrix
				obj.weights = reshape(obj.weights, ...
					obj.CR.getNumSnapshots(), ...
					obj.CR.getNumNeuronsPost(), ...
					obj.CR.getNumNeuronsPre());
				
				% reshape for plotting
 				obj.weights = permute(obj.weights,[2 3 1]); % Y X T
			elseif strcmpi(obj.plotType,'histogram')
				obj.plotHistBins = linspace(obj.CR.getMinWeight(), ...
					obj.CR.getMaxWeight(), ...
					obj.plotHistNumBins);
				for i=1:numel(obj.timeStamps)
					obj.plotHistData(i,:) = histc(abs(obj.weights(i,:)), ...
						obj.plotHistBins);
				end
			else
				obj.throwError(['Unrecognized plot type "' obj.plotType '".'])
				return
			end
			
			obj.needToLoadData = false;
		end
		
		function loadDefaultParams(obj)
			% private function to load default parameter values
			obj.CR = [];
			obj.weights = [];
			obj.timeStamps = [];
			
			obj.plotType = 'default';
			obj.setConnectFileAttributes()
			obj.setPlottingAttributes()
			obj.setRecordingAttributes()
			
			obj.plotHistData = [];
			obj.plotHistBins = [];
			
			obj.plotStepFrames = false;
			obj.plotStepFramesFW = false;
			obj.plotStepFramesBW = false;
			
			% for the group name in plot titles, mask underscores so that
			% they're not interpreted as LaTeX; except for '_{', which
			% should be interpreted as LaTeX for lowerscript
			obj.plotTitlePreName = regexprep(strrep(obj.grpPreName, '_', '\_'),'\\_{','_{');
			obj.plotTitlePostName = regexprep(strrep(obj.grpPostName, '_', '\_'),'\\_{','_{');

			obj.plotSubplotsPerFig = 80;
			
			obj.needToInitCR = true;
			obj.needToLoadData = true;
			
			obj.supportedPlotTypes = {'heatmap', 'histogram', ...
				'receptivefield', 'responsefield'};
			obj.supportedErrorModes = {'standard','warning','silent'};

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
		
		function plotFrame(obj, frameNr, plotType, neurons, dispFrameNr)
			% Private method to display a single frame depending on
			% plotType. This is where the raster plots and heat maps are
			% implemented.
			if nargin<5,dispFrameNr=obj.plotDispFrameNr;end
			if nargin<4,neurons=-1;end
			if nargin<3,plotType=obj.plotType;end
			obj.unsetError()
			
			% load data and reshape for plotting if necessary
			obj.loadDataForPlotting(plotType);
			
			subTitle = '';
			if strcmpi(obj.plotType,'heatmap')
				imagesc(obj.weights(:,:,frameNr), [0 max(obj.CR.getMaxWeight(),1e-10)])
				axis image square
				xlabel('nrPre')
				ylabel('nrPost')
				subTitle = ['wt = [0 , ' num2str(obj.CR.getMaxWeight()) ']'];
				
				% if enabled, display the frame number in lower left corner
				if dispFrameNr
					text(2,size(obj.weights,1)-1,num2str(frameNr), ...
						'FontSize',10,'BackgroundColor','white')
				end
			elseif strcmpi(obj.plotType,'histogram')
				bar(obj.plotHistBins, obj.plotHistData(frameNr,:))
				xlabel('weight value (magnitude)')
				ylabel('number of synapses')
				subTitle = ['wt = [0 , ' num2str(obj.CR.getMaxWeight()) ']'];
				% if enabled, display the frame number in lower left corner
				if dispFrameNr
					text(0,0.1*max(obj.plotHistData(frameNr,:)),num2str(frameNr), ...
						'FontSize',10,'BackgroundColor','white')
				end
			elseif strcmpi(obj.plotType,'responsefield')
				% plot the connections from one pre-neuron to all
				% corresponding post-neurons
				grid3DPre = obj.CR.getGrid3DPre();
				grid3DPost = obj.CR.getGrid3DPost();
				
				nPlots = numel(neurons);
				[nPlots, nRows, nCols] = obj.findPlotLayout(nPlots);
				for p=1:nPlots
					for r=1:nRows
						for c=1:nCols
							idxNeur = (r-1)*nCols+c+(p-1)*nRows*nCols;
							idxSubplot = (r-1)*nCols+c;
							if idxNeur > numel(neurons)
								break;
							end
							
							% get all incoming weights to that neuron
							neurIdPre = neurons(idxNeur);
							wts = obj.weights(:,neurIdPre,frameNr); % Y X T
							wts = reshape(wts,grid3DPost);
							
							% find RF in same z-plane
							% for this: find z-coordinate of post, compare to
							% all z-coordinates of pre, find the match
							zPool = floor( (neurIdPre-1)/grid3DPre(1)/grid3DPre(2) );
							zPre = zPool - (grid3DPre(3)-1.0)/2.0;
							zPost = (0:grid3DPost(3)-1) - (grid3DPost(3)-1.0)/2.0;
							[~,j] = min(abs(zPre-zPost));
							zPostIdx = j; % find post-coord in all pre
							
							if sum(zPostIdx)==0
								% this pre-neuron does not connect to any
								% post-neurons in the same plane
								continue;
							end
							
							% plot RF
							subplot(nRows,nCols,idxSubplot)
							imagesc(wts(:,:,zPostIdx)', [0 max(obj.CR.getMaxWeight(),1e-10)])
							axis equal
							axis([1 grid3DPost(1) 1 grid3DPost(2)])
							if grid3DPost(1)>2
								set(gca,'XTick',[1 grid3DPost(1)/2.0 grid3DPost(1)])
								set(gca,'XTickLabel',[-grid3DPost(1)/2.0 0 grid3DPost(1)/2.0])
							else
								set(gca,'XTick',grid3DPost(1))
								set(gca,'XTickLabel',0)
							end
							if grid3DPost(2)>2
								set(gca,'YTick',[1 grid3DPost(2)/2.0 grid3DPost(2)])
								set(gca,'YTickLabel',[-grid3DPost(2)/2.0 0 grid3DPost(2)/2.0])
							else
								set(gca,'YTick',grid3DPost(2))
								set(gca,'YTickLabel',0)
							end
							xlabel('x')
							ylabel('y')
							subTitle = ['wt = [0 , ' num2str(obj.CR.getMaxWeight()) ...
								'], z=' num2str(zPre)];
							
							% if enabled, display the frame number in lower left corner
							if dispFrameNr
								text(2,size(wts,2)-1,num2str(frameNr), ...
									'FontSize',10,'BackgroundColor','white')
							end
						end
					end
					if p<nPlots,figure;end
				end
			elseif strcmpi(obj.plotType,'receptivefield')
				% plot the connections from all corresponding pre-neurons
				% to one post-neuron
				grid3DPre = obj.CR.getGrid3DPre();
				grid3DPost = obj.CR.getGrid3DPost();

				nPlots = numel(neurons);
				[nPlots, nRows, nCols] = obj.findPlotLayout(nPlots);
				for p=1:nPlots
					for r=1:nRows
						for c=1:nCols
							idxNeur = (r-1)*nCols+c+(p-1)*nRows*nCols;
							idxSubplot = (r-1)*nCols+c;
							if idxNeur > numel(neurons)
								break;
							end
							
							% get all incoming weights to that neuron
							neurIdPost = neurons(idxNeur);
							wts = obj.weights(neurIdPost,:,frameNr); % Y X T
							wts = reshape(wts,grid3DPre);
							
							% find RF in same z-plane
							% for this: find z-coordinate of post, compare to
							% all z-coordinates of pre, find the match
							zPool = floor( (neurIdPost-1)/grid3DPost(1)/grid3DPost(2) );
							zPost = zPool - (grid3DPost(3)-1.0)/2.0;
							zPre = (0:grid3DPre(3)-1) - (grid3DPre(3)-1.0)/2.0;
							[~,j] = min(abs(zPre-zPost));
							zPreIdx = j; % find post-coord in all pre
							
							if sum(zPreIdx)==0
								% this pre-neuron does not connect to any
								% post-neurons in the same plane
								continue;
							end
							
							% plot RF
							subplot(nRows,nCols,idxSubplot)
							imagesc(wts(:,:,zPreIdx)', [0 max(obj.CR.getMaxWeight(),1e-10)])
							axis equal
							if grid3DPre(1)>1 && grid3DPre(2)>1
								axis([1 grid3DPre(1) 1 grid3DPre(2)])
							end
							if grid3DPre(1)>1
								set(gca,'XTick',[1 grid3DPre(1)/2.0 grid3DPre(1)])
								set(gca,'XTickLabel',[-grid3DPre(1)/2.0 0 grid3DPre(1)/2.0])
							else
								set(gca,'XTick',grid3DPre(1))
								set(gca,'XTickLabel',0)
							end
							if grid3DPre(2)>1
								set(gca,'YTick',[1 grid3DPre(2)/2.0 grid3DPre(2)])
								set(gca,'YTickLabel',[-grid3DPre(2)/2.0 0 grid3DPre(2)/2.0])
							else
								set(gca,'YTick',grid3DPre(2))
								set(gca,'YTickLabel',0)
							end
							xlabel('x')
							ylabel('y')
							subTitle = ['wt = [0 , ' num2str(obj.CR.getMaxWeight()) ...
								'], z=' num2str(zPost)];
						end
					end
					if p<nPlots,figure;end
				end
			else
				obj.throwError(['Unrecognized plot type "' obj.plotType '".'])
				return
			end
			title({[obj.plotTitlePreName '->' obj.plotTitlePostName ', t=' ...
							num2str(obj.timeStamps(frameNr)) 'ms'],subTitle})
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