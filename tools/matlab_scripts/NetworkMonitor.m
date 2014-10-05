classdef NetworkMonitor < handle
    % Version 10/3/2014
    % Author: Michael Beyeler <mbeyeler@uci.edu>
    
    %% PROPERTIES
    % public
    properties (SetAccess = private)
        resultsFolder;      % results directory where all spike files live
        ntwFile;            % network file "ntw_{ntwName}.dat"
        
        groupNames;         % cell array of population names
        groupGrid3D;        % cell array of population dimensions
        groupPlotTypes;     % cell array of population plot types

        groupSubPlots;      % cell array of assigned subplot slots
        simObj;                % instance of simObjulationReader class

        numSubPlots;
    end
    
    % private
    properties (Hidden, Access = private)
        
        errorMode;          % program mode for error handling
        errorFlag;          % error flag (true if error occured)
        errorMsg;           % error message

        groupMonObj;        % cell array of GroupMonitor objects
        
        plotAbortPlotting;

        supportedErrorModes;% supported error modes
    end
    
    
    %% PUBLIC METHODS
    methods
        function obj = NetworkMonitor(ntwFile, loadGroupsFromFile, errorMode)
            obj.unsetError()
            obj.loadDefaultParams()
            
            if nargin<3
                obj.errorMode = 'standard';
            else
                if ~obj.isErrorModeSupported(errorMode)
                    obj.throwError(['errorMode "' errorMode '" is currently' ...
                        ' not supported. Choose from the following: ' ...
                        strjoin(obj.supportedErrorModes, ', ') '.'], ...
                        'standard')
                else
                    obj.errorMode = errorMode;
                end
            end
            if nargin<2,loadGroupsFromFile=true;end
            
            [filePath,fileName,fileExt] = fileparts(ntwFile);
            if strcmpi(fileExt,'')
                obj.throwError(['Parameter ntwFile must be a file name, ' ...
                    'directory found.'])
            end
            
            obj.resultsFolder = filePath;
            obj.ntwFile = [fileName fileExt];
                        
            % try to read simulation file
            obj.readsimulationFile()
            
            % if flag is set, add all groups for plotting from file
            if loadGroupsFromFile
                obj.addAllGroupsFromFile()
            end
        end
        
        function delete(obj)
            % destructor, implicitly called to fclose file
        end
        
        function addGroup(obj, name, plotType, grid3D, subPlots, errorMode)
            % obj.addPopulation(name,plotType, grid3D) adds a population to the
            % ActivityMonitor.
            %
            % NAME            - A string identifier that should also be
            %                   present in the population's spike file name
            %                   (spkFile=[saveFolder prefix name suffix]).
            %                   See setSpikeFileAttributes for more
            %                   information.
            % PLOTTYPE        - Specifies how the population will be plotted.
            %                   Currently supported are the following types:
            %                    - heatmap     a topological map where hotter
            %                                  colors mean higher firing rate
            % GRID3D          - A 3-element vector that specifies the width, the
            %                   height, and the depth of the 3D neuron grid.
            %                   The product of these dimensions should equal the
            %                   total number of neurons in the population.
            %                   By default, this parameter will be read from
            %                   file, but can be overwritten by the user.
            if nargin<6,errorMode=obj.errorMode;end
            if nargin<5,subPlots=[];end
            if nargin<4,grid3D=-1;end
            if nargin<3,plotType='default';end

            % find index of group in simObj struct
            indStruct = obj.getGroupStructId(name);
            if indStruct<=0
                obj.throwError(['Group "' name '" could not be found. '])
%                     'Choose from the following: ' ...
%                     strjoin({obj.simObj.groups(:).name},', ') '.' ...
                return
            end
            
            % create GroupMonitor object for this group
            GM = GroupMonitor(name, obj.resultsFolder, errorMode);
            
            % check whether valid spike file found, exit if not found
            [errFlag,errMsg] = GM.getError();
            if errFlag
                obj.throwError(errMsg, errorMode);
                return % make sure we exit after spike file not found
            end
            
            % set plot type to specific type or find default type
            GM.setPlotType(plotType);
            
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
                obj.throwWarning(['A population with name "' name '" has ' ...
                    'already been added. Replacing values...']);
                id = obj.getGroupId(name);
                obj.groupMonObj{id}       = GM;
%                 obj.numSubPlots           = obj.numSubPlots ...
%                                             - numel(obj.groupSubPlots{id}) ...
%                                             + numel(subPlots);
%                 obj.groupSubPlots{id}     = subPlots;
            else
                % else add new entry
                obj.groupNames{end+1}     = name;
                obj.groupMonObj{end+1}    = GM;
                obj.numSubPlots           = obj.numSubPlots + numel(subPlots);
                obj.groupSubPlots{end+1}  = subPlots;
            end
            
        end
        
        function [errFlag,errMsg] = getError(obj)
            % [errFlag,errMsg] = getError() returns the current error status.
            % If an error has occurred, errFlag will be true, and the message
            % can be found in errMsg.
            errFlag = obj.errorFlag;
            errMsg = obj.errorMsg;
        end
        
        function plot(obj, groupNames, frames, frameDur, stepFrames, fps, dispFrameNr)
            if nargin<7,dispFrameNr=true;end
            if nargin<6,fps=5;end
            if nargin<5,stepFrames=false;end
            if nargin<4,frameDur=1000;end
            if nargin<3,frames=-1;end
            if nargin<2,groupNames={};end
            obj.unsetError()

            % reset abort flag, set up callback for key press events
            obj.plotAbortPlotting = false;
            set(gcf,'KeyPressFcn',@obj.pauseOnKeyPressCallback)

            if ~iscell(groupNames)
                obj.throwError('groupNames must be a cell array')
                return
            end
            % set list of groups
            if isempty(groupNames)
                groupNames = obj.groupNames;
            end
            
            % set range of frames
            if isempty(frames) || frames==-1
                frames = 1:ceil(obj.simObj.sim.simTimeSec*1000.0/frameDur);
            end

            % load data and reshape for plotting if necessary
            for i=1:numel(groupNames)
                gId = obj.getGroupId(groupNames{i});
                obj.groupMonObj{gId}.loadDataForPlotting([],frameDur);
            end
            
            % display frames in specified axes
            [nrR, nrC] = obj.findPlotLayout(obj.numSubPlots);
            for f=frames
                if obj.plotAbortPlotting
                    % user pressed button to quit plotting
                    obj.plotAbortPlotting = false;
                    close;
                    return
                end
                
                for g=1:numel(groupNames)
                    gId = obj.getGroupId(groupNames{g});
                    subplot(nrR, nrC, obj.groupSubPlots{gId})
                    obj.groupMonObj{gId}.plotFrame(f,[],frameDur,dispFrameNr);
                end
                drawnow
                
                % wait for button press or pause
                if stepFrames
                    waitforbuttonpress;
                else
                    pause(1.0/fps)
                end
            end
        end
        
        function setPlottingAttributes(this,varargin)
            % AM.setPlottingAttributes(varargin) sets different plotting
            % attributes, such as how many frames to plot per second, or
            % whether to record video. Call function without input arguments to
            % set default values.
            % To take effect, this function must be called before
            % AM.plotPopulations().
            % AM.setPlottingAttributes('PropertyName',VALUE,...) sets the
            % specified property values.
            %
            % BGCOLOR            - Set background color for figure. Must be of 
            %                      type ColorSpec (char such as 'w','b','k' or a
            %                      3-element vector for RGB channels). The
            %                      default is white.
            %
            % FPS                - The frames per second for the plotting loop.
            %                      The default is 5.
            %
            % RECORDMOVIE        - A boolean flag that indicates whether to
            %                      record the plotting loop as an AVI movie. If
            %                      flag is set to false, the following
            %                      parameters with prefix "recordMovie" will not
            %                      take effect.
            %
            % RECORDMOVIEFILE    - File name where movie will be stored.
            %                      Currently the only supported file ending is
            %                      ".avi".
            %
            % RECORDMOVIEFPS     - The frames per second for the movie. The
            %                      default is 10.
            %
            % WAITFORBUTTONPRESS - A boolean flag that indicates whether to wait
            %                      for user to pres key/button before plotting
            %                      the next frame.
            if isempty(varargin)
                % set default values
                this.plotBgColor = 'w';
                this.plotFPS = 5;
                this.plotWaitButton = false;
                this.recordMovie = false;
                this.recordMovieFile = 'movie.avi';
                this.recordMovieFPS = 10;
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
                        this.plotBgColor = val;
                    case 'fps'
                        % frames per second
                        throwErrNumeric = ~isnumeric(val);
                        reqRange = [0.01 100];
                        throwErrOutOfRange = val<reqRange(1) | val>reqRange(2);
                        this.plotFPS = val;
                    case 'recordmovie'
                        % whether to record movie
                        throwErrNumeric = ~isnumeric(val) & ~islogical(val);
                        this.recordMovie = val;
                    case 'recordmoviefile'
                        % filename for recorded movie (must be .avi)
                        reqFileEnding = '.avi';
                        throwErrFileEnding = ~strcmpi(val(max(1,end-3):end), ...
                            reqFileEnding);
                        this.recordMovieFile = val;
                    case 'recordmoviefps'
                        % frames per second for recorded movie
                        throwErrNumeric = ~isnumeric(val);
                        reqRange = [0.01 10];
                        throwErrOutOfRange = val<reqRange(1) | val>reqRange(2);
                        this.recordMovieFPS = val;
                    case 'waitforbuttonpress'
                        % whether to wait for button press before next frame
                        throwErrNumeric = ~isnumeric(val) & ~islogical(val);
                        this.plotWaitButton = logical(val);
                    otherwise
                        % attribute does not exist
                        if isnumeric(attr) || islogical(attr)
                            attr = num2str(attr);
                        end
                        error(['Unknown attribute "' attr '"'])
                end
                
                % throw errors
                if throwErrFileEnding
                    error(['File ending for attr "' attr '" must be ' ...
                        '"' reqFileEnding '"'])
                elseif throwErrNumeric
                    error(['Value for attr "' attr '" must be ' ...
                        'numeric'])
                elseif throwErrOutOfRange
                    error(['Value for attr "' attr '" must be in ' ...
                        'range [' num2str(reqRange(1)) ',' ...
                        num2str(reqRange(2)) ']'])
                end
                
                % advance index to next attr
                nextIndex = nextIndex + 2;
            end
        end
        
    end
    
    %% PRIVATE METHODS
    methods (Hidden, Access = private)
        function addAllGroupsFromFile(obj)
            % add groups in silent mode, so that no errors are thrown if not
            % all spike files are found
            errMode = 'silent';
            
            for i=1:numel(obj.simObj.groups)
                obj.addGroup(obj.simObj.groups(i).name, 'default', ...
                    [], [], errMode)
            end
        end
        
        function bool = existsGroupName(obj,name)
            % bool = AM.existsGroupName(name) checks whether a group has
            % been registered under name NAME
            bool = any(strcmpi(obj.groupNames,name));
        end
        
        function [nrR, nrC] = findPlotLayout(obj, numSubPlots)
            % given a total number of subplots, what should be optimal number
            % of rows and cols in the figure?
            nrR = floor(sqrt(numSubPlots));
            nrC = ceil(numSubPlots*1.0/nrR);
        end

        function id = getGroupId(obj,name)
            % id = AM.getGroupId(name) returns the index of population with
            % name NAME
            %
            % NAME  - A string representing the name of a group that has been
            %         registered by calling AM.addPopulation.
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


        function loadDefaultParams(obj)
            
            obj.supportedErrorModes = {'standard', 'warning', 'silent'};
            
            obj.groupNames = {};
            obj.groupGrid3D = {};
            obj.groupPlotTypes = {};
            obj.groupSubPlots = {};
            obj.groupMonObj = {};
            
            obj.plotAbortPlotting = false;

            obj.numSubPlots = 0;
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
        
        function readsimulationFile(obj)
            % try and read simulation file
            obj.simObj = SimulationReader([obj.resultsFolder filesep obj.ntwFile]);
        end
        
        function throwError(obj, errorMsg, errorMode)
            % THROWERROR(errorMsg, errorMode) throws an error with a specific
            % severity (errorMode). In all cases, obj.errorFlag is set to true
            % and the error message is stored in obj.errorMsg. Depending on
            % errorMode, an error is either thrown as fatal, thrown as warning,
            % or not thrown at all.
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
            % THROWWARNING(errorMsg, errorMode) throws a warning with a specific
            % severity (errorMode).
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