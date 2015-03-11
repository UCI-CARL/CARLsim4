classdef NetworkMonitor < handle
    % A NetworkMonitor can be used to monitor properties as well as the
    % activity of a number of neuronal groups in a network.
    %
    % NM = NetworkMonitor(simFile,loadGroupsFromFile,errorMode) creates a
    % new instance of class NetworkMonitor.
    % A NetworkMonitor will assume that the corresponding spike file for
    % each group has been created during the CARLsim simulation.
	%
    % Example usage:
    % A) Automatically add all groups in the network
    %    >> NM = NetworkMonitor('results/sim_random.dat'); % read sim file
    %    >> NM.plot; % hit 'p' to pause, 'q' to quit
    %    >> NM.recordMovie; % plots all groups and saves as 'movie.avi'
    %    >> NM.setGroupPlotType('inhib','raster'); % switch that group to raster
    %    >> NM.removeGroup('input'); % exclude this group from plotting    
    %    >> % etc.
    % B) Add groups one-by-one
    %    >> NM = NetworkMonitor('results/sim_random.dat',false);
    %    >> NM.addGroup('input','raster');
    %    >> NM.plot(1:10,100); % frame=100ms, plot first 10 frames
    %    >> % etc.
	% C) Add groups whose spike files don't follow the default file name,
	%    say 'spikeFile_{groupName}.ext' instead of 'spk_{groupName}.dat'
	%    >> NM = NetworkMonitor('results/sim_random.dat',false);
	%    >> NM.setSpikeFileAttributes('spikeFile_','.ext')
	%    >> NM.addAllGroupsFromFile()
    %
    % Version 2/27/2015
    % Author: Michael Beyeler <mbeyeler@uci.edu>
    
    %% PROPERTIES
    % public
    properties (SetAccess = private)
        simObj;             % instance of simObjulationReader class
        resultsFolder;      % results directory where all spike files live
        
        groupNames;         % cell array of population names
        groupSubPlots;      % cell array of assigned subplot slots
        groupMonObj;        % cell array of GroupMonitor objects

        errorMode;          % program mode for error handling
        supportedErrorModes;% supported error modes
    end
    
    % private
    properties (Hidden, Access = private)
        errorFlag;          % error flag (true if error occured)
        errorMsg;           % error message
        
        ntwFile;            % network file "ntw_{ntwName}.dat"
        numSubPlots;        % number of subplots
        
        plotAbortPlotting;  % flag whether to abort plotting (on-click)
        plotBgColor;        % bg color of plot (for plotting)
        plotDispFrameNr;    % flag whether to display frame number
        plotFPS;            % frames per second for plotting
        plotBinWinMs;       % binning window size (time)

        plotStepFrames;     % flag whether to waitforbuttonpress btw frames
        plotStepFramesFW;    % flag whether to make a step forward
        plotStepFramesBW;    % flag whether to make a step backward
        
        recordBgColor;      % bg color of plot (for recording)
        recordFile;         % filename for recording
        recordFPS;          % frames per second for recording
        recordWinSize;      % window size of plot for recording

        spkFilePrefix;      % spike file prefix, e.g. "spk_"
        spkFileSuffix;      % spike file suffix, e.g. ".dat"
	end
    
    
    %% PUBLIC METHODS
    methods
        function obj = NetworkMonitor(simFile, loadGroupsFromFile, errorMode)
            % NM = NetworkMonitor(simFile,loadGroupsFromFile,errorMode)
            % creates a new instance of class NetworkMonitor, which can be
            % used to monitor network properties and activity.
            % A NetworkMonitor will assume that a corresponding spike file
            % has been created for each group in the network during the
            % CARLsim simulation. If a group does not have a spike file, it
            % cannot be added to the plot.
            %
            % SIMFILE             - Path to a simulation file (of the sort
            %                       "sim_{simName}.dat")
            % LOADGROUPSFROMFILE  - A flag whether to automatically add all
            %                       groups in the network. Default: true.
            % ERRORMODE           - Error Mode in which to run SpikeReader.
            %                       The following modes are supported:
            %                         - 'standard' Errors will be fatal
            %                                      (returned via Matlab
            %                                      function error())
            %                         - 'warning'  Errors will be warnings
            %                                      returned via Matlab
            %                                      function warning())
            %                         - 'silent'   No exceptions will be
            %                                      thrown, but object will
            %                                      populate the properties
            %                                      errorFlag and errorMsg.
            %                       Default: 'standard'.
            obj.unsetError()
            obj.loadDefaultParams()
            
            % parse input arguments
            if nargin<3
				obj.setErrorMode('standard');
			else
				obj.setErrorMode(errorMode);
            end
            if nargin<2,loadGroupsFromFile=true;end
            
            % make sure fileName points to a valid network file
            [filePath,fileName,fileExt] = fileparts(simFile);
            if strcmpi(fileExt,'')
                obj.throwError(['Parameter ntwFile must be a file name, ' ...
                    'directory found.'])
            end
            obj.resultsFolder = filePath;
            obj.ntwFile = [fileName fileExt];
            
            % try to read network file
            obj.readsimulationFile()
            
            % if flag is set, add all groups for plotting from file
            if loadGroupsFromFile
                obj.addAllGroupsFromFile('silent')
            end
        end
        function delete(obj)
            % destructor
        end
        
        function addAllGroupsFromFile(obj, errMode)
			% NM.addAllGroupsFromFile() adds all groups found in the
			% simulation file (see NM constructor). This function is
			% implicitly run if the flag loadGroupsFromFile in the NM
			% constructor is set to true.
			%
			% This function can be helpful if there is a mismatch in the
			% spike file attributes. For example, assume you want to add a
			% group "output" with corresponding spike file "output.dat",
			% but NM is looking for the default name, "spk_output.dat". So
			% you can run:
			% >> NM.setSpikeFileAttributes('','.dat')
			% >> NM.addAllGroupsFromFile()
			% and the missing groups will now be found.
			obj.unsetError()
			if nargin<2,errMode=obj.errorMode;end

            for i=1:numel(obj.simObj.groups)
                obj.addGroup(obj.simObj.groups(i).name, 'default', ...
                    [], [], errMode)
            end
		end
		
		function addGroup(obj, name, plotType, grid3D, subPlots, errorMode)
            % NM.addGroup(name,plotType,grid3D,subPlots,errorMode) adds a
            % specific neuronal group to the NetworkMonitor.
            % This function is implicitly called on all groups if a
            % NetworkMonitor object is created with flag loadGroupsFromFile
            % set to true.
            % If addGroup is called on a group that has already been added,
            % all previously configured properties for that group will be
            % overwritten.
            %
            % NAME           - Group name string, for which a spike file
            %                  should exist. The spike file name is given
            %                  by spkFile=[saveFolder prefix name suffix].
            %                  If it cannot be found, adjust prefix and
            %                  suffix using NM.setSpikeFileAttributes.
            % PLOTTYPE       - The plotting type to use. If not set, the
            %                  default plotting type will be used, which
            %                  is determined by the Grid3D topography of
            %                  the group.
            %                  The following types are currently supported:
            %                   - heatmap   a topological map of group
            %                               activity where hotter colors
            %                               mean higher firing rate
            %                   - raster    a raster plot with binning
            %                               window: binWindowMs
            %                  Default: 'default'.
            % GRID3D         - A 3-element vector that specifies the width,
            %                  the height, and the depth of the 3D neuron
            %                  grid. The product of these dimensions should
            %                  equal the total number of neurons in the
            %                  group.
            %                  By default, this parameter will be read from
            %                  file, but can be overwritten by the user.
            % SUBPLOTS       - A vector of subplot numbers in which to
            %                  display the group's plot.
            %                  By default, each group gets assigned just
            %                  one subplot, but can be manually configured
            %                  by the user.
            % ERRORMODE      - Error Mode in which to perform the function.
            if nargin<6,errorMode=obj.errorMode;end
            if nargin<5,subPlots=[];end
            if nargin<4,grid3D=-1;end
            if nargin<3,plotType='default';end
            obj.unsetError()
            
            % find index of group in simObj struct
            indStruct = obj.getGroupStructId(name);
            if indStruct<=0
                obj.throwError(['Group "' name '" could not be found. '])
                %                     'Choose from the following: ' ...
                %                     strjoin({obj.simObj.groups(:).name},', ') '.' ...
                return
            end
            
            % create GroupMonitor object for this group
			% create in silent to suppress "spike file not found" warning
			% but set proper error mode right after
            GM = GroupMonitor(name, obj.resultsFolder, 'silent');
			GM.setErrorMode(errorMode);
			
			% set spike file attributes
			GM.setSpikeFileAttributes(obj.spkFilePrefix,obj.spkFileSuffix)
            
            % check whether valid spike file found, exit if not found
            if ~GM.hasValidSpikeFile()
	            [obj.errorFlag,obj.errorMsg] = GM.getError();
                obj.throwError(obj.errorMsg, errorMode);
                return % make sure we exit after spike file not found
            end
            
            % set plot type to specific type or find default type
            GM.setDefaultPlotType(plotType);
            
            % disable interactive mode to avoid press key events etc.
            GM.setPlottingAttributes('interactiveMode',false);
            
            % set Grid3D if necessary
            if ~isempty(grid3D) && prod(grid3D)>=1
                GM.setGrid3D(grid3D);
            end
            
            % assign default subplots if necessary
            if isempty(subPlots) || prod(subPlots)<=0
                subPlots = obj.numSubPlots+1;
            end
            
            if obj.existsGroupName(name)
                % replace entries if pop has already been added
                obj.throwWarning(['A population with name "' name '" ' ...
                    'has already been added. Replacing values...']);
                id = obj.getGroupId(name);
                obj.groupMonObj{id}      = GM;
            else
                % else add new entry
                obj.groupNames{end+1}    = name;
                obj.groupMonObj{end+1}   = GM;
                obj.numSubPlots          = obj.numSubPlots+numel(subPlots);
                obj.groupSubPlots{end+1} = subPlots;
            end
        end
        
        function removeGroup(obj, name)
            % NM.removeGroup(name) will remove the group NAME from the
            % NetworkMonitor (and thus the plots).
            % NAME            - Group name. Must be a group that has
            %                   previously been added with NM.addGroup
            %                   (manually or automatically).
            obj.unsetError()
            if ~Utilities.verify(name,'ischar')
                obj.throwError('Group name must be a string');return
            end
            if ~obj.existsGroupName(name)
                obj.throwError(['Group "' name '" could not be found. '])
                return
            end
            
            % remove the group
            ind2keep = not(obj.getGroupId(name));
            obj.groupNames = obj.groupNames(ind2keep);
            obj.groupMonObj = obj.groupMonObj(ind2keep);
            obj.groupSubPlots = obj.groupSubPlots(ind2keep);
            
            % update number of sub plots
            obj.numSubPlots = max(cell2mat(obj.groupSubPlots));
        end
        
        function [errFlag,errMsg] = getError(obj)
            % [errFlag,errMsg] = getError() returns the current error
            % status.
            % If an error has occurred, errFlag will be true, and the
            % message can be found in errMsg.
            errFlag = obj.errorFlag;
            errMsg = obj.errorMsg;
        end
        
        function grid = getGroupGrid3D(obj, groupName)
            % nNeur = NM.getGroupGrid3D(groupName) returns the Grid3D
            % topography of a specific group.
            % GROUPNAME      - Name of the group.
            obj.unsetError()
            gId = obj.getGroupId(groupName);
            grid = obj.groupMonObj{gId}.getGrid3D();
        end
        
        function nNeur = getGroupNumNeurons(obj, groupName)
            % nNeur = NM.getGroupNumNeurons(groupName) returns the number
            % of neurons in a specific group.
            % GROUPNAME      - Name of the group.
            obj.unsetError()
            gId = obj.getGroupId(groupName);
            nNeur = obj.groupMonObj{gId}.getNumNeurons();
        end
        
        function plot(obj, frames, binWindowMs)
            % NM.plot(frames,binWindowMs,stepFrames) plots the specified
            % frames in the current figure/axes.
            % NM.plot will display all the groups the NetworkMonitor knows
            % of (i.e., that have been added via NM.addGroup). To exclude a
            % group from plots, call NM.removeGroup on it.
            % The plotting type of each group can be changed by calling
            % NM.setGroupPlotType.
            %
            % Press 's' at any time to enter stepping mode.
            % In this mode, pressing the right arrow key will step forward
            % to display the next frame in the list, whereas pressing the
            % left arrow key will step backward to the last frame in the
            % list. Exit stepping mode by pressing 's' again.
            %
            % A list of plotting attributes can be set directly as input
            % arguments. The full list of available attributes can be set
            % using NM.setPlottingAttributes.
            %
            % FRAMES       - A list of frame numbers. For example,
            %                requesting frames=[1 2 8] will display the
            %                first, second, and eighth frame.
            %                Default: display all frames.
            % BINWINDOWMS  - The binning window (ms) in which the data will
            %                be displayed. Default: 1000.
            if nargin<3,binWindowMs=obj.plotBinWinMs;end
            if nargin<2 || isempty(frames) || sum(frames==-1)>0
                frames = 1:ceil(obj.simObj.sim.simTimeSec*1000.0/binWindowMs);
            end
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
			
			% start plotting in regular mode
			obj.plotAbortPlotting = false;
			obj.plotStepFrames = false;
            
            % make sure we have something to plot
            if obj.numSubPlots==0
                obj.throwWarning(['Nothing to plot. Try adding some ' ...
					'groups first.'])
                return
            end
            
            % make sure plot layout is valid; i.e., no subplot is assigned
            % to more than one group
            [isValid,invalidSubPlot] = obj.isPlotLayoutValid();
            if ~isValid
                obj.throwError(['Current plot layout is not valid. ' ...
                    'Subplot ' num2str(invalidSubPlot) ' is assigned ' ...
                    'more than once. Fix via setGroupSubPlots().'])
                return
            end
            
            % reset abort flag, set up callback for key press events
            obj.plotAbortPlotting = false;
            set(gcf,'KeyPressFcn',@obj.ntwMonOnKeyPressCallback)
            
            % display frames in specified axes
            grpNames = obj.groupNames;
            set(gcf,'color',obj.plotBgColor);
            [nrR, nrC] = obj.findPlotLayout(obj.numSubPlots);
            % use a while loop instead of a for loop so that we can
            % implement stepping backward
            idx = 1;
            while idx <= numel(frames)
                if obj.plotAbortPlotting
                    % user pressed button to quit plotting
                    close;
                    return
                end
                
                for g=1:numel(grpNames)
                    gId = obj.getGroupId(grpNames{g});
                    subplot(nrR, nrC, obj.groupSubPlots{gId})
                    obj.groupMonObj{gId}.plot([],frames(idx),binWindowMs);
                end
                drawnow
                
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
            end
            close;
        end
        
        function recordMovie(obj, fileName, frames, binWindowMs, fps, winSize)
            % NM.recordMovie(fileName, frames, frameDur, fps, winSize)
            % takes an AVI movie of a list of frames using the VIDEOWRITER
            % utility.
            % The activity of each group will be recorded according to the
            % plot types specified by default or via NM.setGroupPlotType.
            %
            % FILENAME     - A string enclosed in single quotation marks
            %                that specifies the name of the file to create.
            %                Default: 'movie.avi'.
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
            if nargin<6,winSize=obj.recordWinSize;end
            if nargin<5,fps=obj.recordFPS;end
            if nargin<4,binWindowMs=obj.plotBinWinMs;end
            if nargin<3 || isempty(frames) || sum(frames==-1)>0
                frames = 1:ceil(obj.simObj.sim.simTimeSec*1000.0/binWindowMs);
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
            
            % make sure we have something to plot
            if obj.numSubPlots==0
                disp('Nothing to record. Try adding some groups first.')
                return
            end
            
            % make sure plot layout is valid; i.e., no subplot is assigned
            % to more than one group
            [isValid,invalidSubPlot] = obj.isPlotLayoutValid();
            if ~isValid
                obj.throwError(['Current plot layout is not valid. ' ...
                    'Subplot ' num2str(invalidSubPlot) ' is assigned ' ...
                    'more than once. Fix via setGroupSubPlots().'])
                return
            end
            
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
            
            % display and record all frames
            grpNames = obj.groupNames;
            [nrR, nrC] = obj.findPlotLayout(obj.numSubPlots);
            for f=frames
                for g=1:numel(grpNames)
                    gId = obj.getGroupId(grpNames{g});
                    subplot(nrR, nrC, obj.groupSubPlots{gId})
                    obj.groupMonObj{gId}.plot([], f,  binWindowMs);
                end
                drawnow
                writeVideo(vidObj, getframe(gcf));
            end
            close(gcf)
            close(vidObj);
            disp(['created file "' fileName '"'])
        end
        
		function setErrorMode(obj, errMode)
			% NM.setErrorMode(errMode) sets the default error mode of the
			% NetworkMonitor object to errMode.
			%
			% For a list of supported error mode, see the property
			% NM.supportedErrorModes.
			if nargin<2 || isempty(errMode) || strcmpi(errMode,'default')
				errMode='standard';
			end
			
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
		
		function setGroupGrid3D(obj, groupName, dim0, dim1, dim2, ...
                updDefPlotType)
            % NM.setGroupGrid3D(groupName,dim0,dim1,dim2) sets the Grid3D
            % topography of the group. The total number of neurons in the
            % group (width x height x depth) cannot change.
            % If one of the three arguments are set to -1, its value will
            % be automatically adjusted so that the total number of neurons
            % in the group stays the same.
            % GROUPNAME      - Name of the group.
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
            if nargin<6,updDefPlotType=false;end
            gId = obj.getGroupId(groupName);
            obj.groupMonObj{gId}.setGrid3D(dim0,dim1,dim2,updDefPlotType);
        end
        
        function setGroupPlotType(obj, groupName, plotType)
            % NM.setGroupPlotType(groupName,plotType) sets the plotting
            % type of a specific group.
            % GROUPNAME      - Group name string, for which a spike file
            %                  should exist. The spike file name is given
            %                  by spkFile=[saveFolder prefix name suffix].
            %                  If it cannot be found, adjust prefix and
            %                  suffix using NM.setSpikeFileAttributes.
            % PLOTTYPE       - The plotting type to use. If not set, the
            %                  default plotting type will be used, which
            %                  is determined by the Grid3D topography of
            %                  the group.
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
            %                             via NM.setPlottingAttributes.
            %                 - raster    A raster plot with binning window
            %                             binWindowMs
            gId = obj.getGroupId(groupName);
            obj.groupMonObj{gId}.setDefaultPlotType(plotType);
        end
        
        function setGroupSubPlot(obj, groupName, subPlots)
            % NM.setGroupSubPlot(groupName, subPlots) specifies in which
            % subplots the activity of a certain group is displayed.
            % GROUPNAME      - Group name string, for which a spike file
            %                  should exist. The spike file name is given
            %                  by spkFile=[saveFolder prefix name suffix].
            %                  If it cannot be found, adjust prefix and
            %                  suffix using NM.setSpikeFileAttributes.
            % SUBPLOTS       - A vector of subplot numbers in which to
            %                  display the group's plot.
            %                  By default, each group gets assigned just
            %                  one subplot, but can be manually configured
            %                  by the user.
            obj.unsetError()
            if ~Utilities.verify(groupName,'ischar')
                obj.throwError('Group name must be a string');return
            end
            if ~Utilities.verify(subPlots,{{'isvector','isnumeric'}})
                obj.throwError('subPlots must be a numeric vector');return
            end
            
            gId = obj.getGroupId(groupName);
            if sum(gId)==0
                obj.throwError(['Group "' groupName '" could not be found.'])
                return
            end
            obj.groupSubPlots{gId} = subPlots;
            obj.numSubPlots = max(cell2mat(obj.groupSubPlots));
        end
        
        function setPlottingAttributes(obj, varargin)
            % NM.setPlottingAttributes(varargin) can be used to set default
            % settings that will apply to all activity plots.
            % This function provides control over additional attributes
            % that are not available as input arguments to NM.plot or
            % NM.plotFrame.
            % NM.setPlottingAttributes('propertyName1',value1,...) sets the
            % value of 'propertyName1' to value1.
            %
            % Calling the function without input arguments will restore the
            % default settings.
            %
            % BGCOLOR        - Set background color for figure. Must be of
            %                  type ColorSpec (char such as 'w','b','k' or
            %                  a 3-element vector for RGB channels). The
            %                  default is white.
            % BINWINDOWMS    - The binning window (ms) in which the data
            %                  will be displayed. Default: 1000.
            % DISPFRAMENR    - A boolean flag that indicates whether to
            %                  display the frame number. Default: true.
            % FPS            - The frames per second for the plotting loop.
            %                  The default is 5.
            % BINWINDOWMS    - The binning window (ms) in which the data
            %                  will be displayed. Default: 1000.
            obj.unsetError()
            
            if isempty(varargin)
                % set default values
                obj.plotDispFrameNr = true;
                obj.plotBgColor = 'w';
                obj.plotFPS = 5;
                obj.plotBinWinMs = 1000;
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
                    case 'binwindowms'
                        % binning window in ms
                        throwErrNumeric = ~isnumeric(val);
                        reqRange = [1 inf];
                        throwErrOutOfRange = val<reqRange(1) | val>reqRange(2);
                        obj.plotBinWinMs = val;
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
            % NM.setRecordingAttributes(varargin) can be used to set
            % default settings that will apply to all activity recordings.
            % This function provides control over additional attributes
            % that are not available as input arguments to NM.recordMovie.
            % NM.setRecordingAttributes('propertyName1',value1,...) sets
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
                obj.recordFPS = 10;
                obj.recordWinSize = [0 0];
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
            % NM.setSpikeFileAttributes(prefix,suffix)
            % Defines the naming conventions for spike files. They should
            % all reside within the same folder as the simulation file, and
            % be made of a common prefix, the population name (specified in
            % NM.addGroup), and a common suffix.
			%
			% This function can be helpful if there is a mismatch in the
			% spike file attributes. For example, assume you want to add a
			% group "output" with corresponding spike file "output.dat",
			% but NM is looking for the default name, "spk_output.dat". So
			% you can run:
			% >> NM.setSpikeFileAttributes('','.dat')
			% >> NM.addAllGroupsFromFile()
			% and the missing groups will now be found.
			%
            % Example: files 'results/spk_V1.dat', 'results/spkMT.dat'
            %   -> saveFolder = 'results/'
            %   -> prefix = 'spk_'
            %   -> suffix = '.dat'
            %   -> name of population = 'V1' or 'MT'
            if nargin<3,suffix='.dat';end
            if nargin<2,prefix='spk_';end
            obj.unsetError()

            obj.spkFilePrefix=prefix;
            obj.spkFileSuffix=suffix;
		end
    end
    
    
    %% PRIVATE METHODS
    methods (Hidden, Access = private)
        function bool = existsGroupName(obj,name)
            % bool = NM.existsGroupName(name) checks whether a group has
            % been registered under name NAME
            bool = any(strcmpi(obj.groupNames,name));
        end
        
        function [nrR, nrC] = findPlotLayout(obj, numSubPlots)
            % given a total number of subplots, what should be optimal
            % number of rows and cols in the figure?
            % \TODO could be static or in Utilities class
            nrR = floor(sqrt(numSubPlots));
            nrC = ceil(numSubPlots*1.0/nrR);
        end
        
        function id = getGroupId(obj,name)
            % id = NM.getGroupId(name) returns the index of population with
            % name NAME
            %
            % NAME  - A string representing the name of a group that has
            %         been registered by calling NM.addPopulation.
            id = strcmpi(obj.groupNames,name);
        end
        
        function index = getGroupStructId(obj, name)
            % finds the index in the simObj struct for a group name
            
            % convert struct to flattened cell array
            cellsimObj = reshape(struct2cell(obj.simObj.groups),1,[]);
            
            % find index of group name
            [~,j] = find(strcmpi(cellsimObj,name));
            if isempty(j)
                % group not found
                index = -1;
            else
                % convert back to struct index
                index = (j-1)/numel(fieldnames(obj.simObj.groups))+1;
            end
        end
        
        function isSupported = isErrorModeSupported(obj, errMode)
            % determines whether an error mode is currently supported
            isSupported = sum(ismember(obj.supportedErrorModes,errMode))>0;
        end
        
        function [isValid,invalidSubPlot] = isPlotLayoutValid(obj)
            % determines whether a plot layout is legal
            % an illegal plot layout has at least one subplot assigned to
            % more than one groups
            
            % flatten subplot cell array
            sp = cell2mat(obj.groupSubPlots);
            
            % make sure every subplot is assigned at most once
            [cnt,sp] = hist(sp, unique(sp));
            [maxVal,maxPos] = max(cnt);
            
            if maxVal==1
                isValid = true;
                invalidSubPlot = [];
            else
                isValid = false;
                invalidSubPlot = sp(maxPos);
            end
        end
        
        
        function loadDefaultParams(obj)
            obj.supportedErrorModes = {'standard', 'warning', 'silent'};
            
            obj.groupNames = {};
            obj.groupSubPlots = {};
            obj.groupMonObj = {};
            
            obj.setPlottingAttributes()
            obj.setRecordingAttributes()
			obj.setSpikeFileAttributes()
			obj.setErrorMode()

            obj.plotStepFrames = false;
            obj.plotStepFramesFW = false;
            obj.plotStepFramesBW = false;

            obj.numSubPlots = 0;

			% disable backtracing for warnings and errors
			warning off backtrace
        end
        
        function ntwMonOnKeyPressCallback(obj,~,eventData)
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
        
        function readsimulationFile(obj)
            % try and read simulation file
            obj.simObj = SimulationReader([obj.resultsFolder filesep obj.ntwFile]);
        end
        
        function throwError(obj, errorMsg, errorMode)
            % NM.throwError(errorMsg, errorMode) throws an error with a
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