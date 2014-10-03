classdef ActivityMonitor < handle
    %
    % Version 3/29/14
    % Author: Michael Beyeler <mbeyeler@uci.edu>
    
    %% PROPERTIES
    properties (SetAccess = private)
        resultsFolder;      % results directory where all spike files live
        popNames;           % cell array of population names
        popDims;            % cell array of population dimensions
        popPlotTypes;       % cell array of population plot types
        supportedPlotTypes; % cell array of supported plot types

        spkFilePrefix;      % spike file prefix, e.g. "spk"
        spkFileSuffix;      % spike file suffix, e.g. ".dat"
        
        plotAbortPlotting;  % flag to abort plotting (set by callback)
        plotFPS;            % frames per second when plotting
        plotWaitButton;     % whether to wait for buttonpress when plotting
        plotBgColor;        % background color for figure

        
        recordMovie;        % flag whether to record movie
        recordMovieFile;    % where to store recorded movie (must be .avi)
        recordMovieFPS;     % frames per second for recording movie
        
        stimFrameDur;       % frame duration for input stimulus (ms)
        stimLength;         % number of frames in input stimulus
    end
    
    %% PUBLIC METHODS
    methods
        function obj = ActivityMonitor(resultsFolder)
            % constructor
            if nargin<1,resultsFolder='./results/';end
            
            % add trailing '/' if necessary
            if resultsFolder(end)~='/'
                resultsFolder(end+1)='/';
            end
            
            obj.resultsFolder = resultsFolder;
            obj.popNames = {};
            obj.popDims = {};
            obj.supportedPlotTypes = {'heatmap'};
            obj.plotAbortPlotting = false;
            
            obj.setSpikeFileAttributes(); % set default file attr
            obj.setStimulusAttributes();  % set default stim attr
            obj.setPlottingAttributes();  % set default plot attr
        end
        
        function addPopulation(this,name,dims,plotType)
            % obj.addPopulation(name,dims,plotType) adds a population to the
            % ActivityMonitor.
            %
            % NAME            - A string identifier that should also be
            %                   present in the population's spike file name
            %                   (spkFile=[resultsFolder prefix name suffix]).
            %                   See setSpikeFileAttributes for more
            %                   information.
            % DIMS            - A 3-element vector that specifies the width, the
            %                   height, and the number of subpopulations. The
            %                   product of these dimensions should equal the
            %                   total number of neurons in the population.
            % PLOTTYPE        - Specifies how the population will be plotted.
            %                   Currently supported are the following types:
            %                    - heatmap     a topological map where hotter
            %                                  colors mean higher firing rate
            if nargin<4,plotType='heatmap';end
            
            % try to open spike file
            spkFile = this.getSpikeFileName(name);
            fid=fopen(spkFile,'r');
            if fid==-1
                error(['could not open file "' spkFile '"']);
            end
            
            % make sure dims has right dimensions (width*height*subpops)
            if numel(dims)==1
                dims(2) = dims(1);
                dims(3) = 1;
            elseif numel(dims)==2
                dims(3) = 1;
            elseif numel(dims)>3
                error(['invalid dimension size, must be <width x ' ...
                    'height x subpops>']);
            end
            
            % make sure plotType is supported
            if sum(ismember(this.supportedPlotTypes,plotType))==0
                error(['plotType "' plotType '" is currently not ' ...
                    'supported. See obj.supportedPlotTypes for a ' ...
                    'complete list.']);
            end
            
            if this.existsPopName(name)
                % replace entries if pop has already been added
                warning(['A population with name "' name '" has ' ...
                    'already been added. Replacing values']);
                id = this.getPopId(name);
                this.popDims{id}      = dims;
                this.popPlotTypes{id}    = plotType;
            else
                % else add new entry
                this.popNames{end+1}  = name;
                this.popDims{end+1}   = dims;
                this.popPlotTypes{end+1} = plotType;
            end
        end
        
        function printFiringStats(this,names,binSize)
            if nargin<3
                % choose default bin size
                if this.stimFrameDur>-1
                    disp(['No binSize given, using stimFrameDur=' ...
                        num2str(this.stimFrameDur)]);
                    
                    binSize=this.stimFrameDur;
                else
                    warning(['No binSize nor stimFrameDur given, using ' ...
                        'default (1000ms)'])
                    binSize=1000;
                end
            end
            if ischar(names)
                % only one population name given
                name = names;
                clear names;
                names{1} = name;
            end
            
            for i=1:length(names)
                name = names{i};
                
                % read spikes
                spkFile = this.getSpikeFileName(name);
                spk = readSpikes(spkFile,binSize);
                
                % find population dimensions
                dims = this.getPopDims(name);
                if isempty(dims)
                    warning(['Dimensions for group not given, consider ' ...
                        'calling ActivityMonitor.addPopulations on group "' ...
                        name '" first. Using spike file to determine ' ...
                        'population size (#neurons=' num2str(size(spk,2)) ')'])
                end
                
                % grow to right size
                len = max(size(spk,1),this.stimLength);
                if size(spk,1) ~= len
                    spk(len,end) = 0;
                end
                if size(spk,2) ~= prod(dims)
                    spk(end,prod(dims)) = 0;
                end
                
                % convert number of spikes to Hz
                toHz = 1000/binSize;
                
                disp(['Group ' name ':']);
                fprintf(['\tTotal number of spikes: %d\n' ...
                    '\tNumber of spikes per frame: %1.2f +- %1.2f\n'], ...
                    sum(spk(:)), mean(sum(spk,2)), std(sum(spk,2)) );
                fprintf('\tAvg neuron firing per frame: %1.2f +- %1.2f Hz\n', ...
                    mean(spk(:)*toHz), std(spk(:)*toHz));
                fprintf('\tMax neuron firing per frame: %1.2f Hz\n', ...
                    max(spk(:))*toHz);
                fprintf('\tMin neuron firing per frame: %1.2f Hz\n', ...
                    min(spk(:))*toHz);
                
                
            end
        end
        
        function id = getPopId(this,name)
            % id = AM.getPopId(name) returns the index of population with
            % name NAME
            %
            % NAME  - A string representing the name of a group that has been
            %         registered by calling AM.addPopulation.
            id = strcmpi(this.popNames,name);
        end
        
        function dims = getPopDims(this,name)
            % dims = AM.getPopDims(name) finds population name in popNames.
            % Returns [] if name could not be found.
            %
            % NAME  - A string representing the name of a group that has been
            %         registered by calling AM.addPopulation.
            index = strcmpi(this.popNames,name);
            
            if isempty(index) | (numel(this.popDims)<index)
                % index could not be found
                dims = [];
            else
                % use same index for popDims
                dims = this.popDims{index};
            end
        end
        
        function spkFile = getSpikeFileName(this,name)
            % spkFile = AM.getSpikeFileName(name) returns the name of the
            % spike file according to specified prefix and suffix.
            % Prefix and suffix can be set using AM.setSpikeFileAttributes.
            %
            % NAME  - A string representing the name of a group that has been
            %         registered by calling AM.addPopulation.
            spkFile = [ this.resultsFolder ...     % the results folder
                this.spkFilePrefix ...  % something like 'spk_'
                name ...                % the name of the pop
                this.spkFileSuffix ];   % something like '.dat'
        end
        
        
        function plotPopulations(this,names,frames)
            % AM.plotPopulations(names,frameDur,frames)
            % Press p to pause plotting.
            % Press q to quit.
            if nargin<3,frames=-1;end
            if nargin<2,names=this.popNames;end
            if ~iscell(names),error('pops must be a cell array'),end
            
            if ~this.plotWaitButton
                set(gcf,'KeyPressFcn',@this.pauseOnKeyPressCallback)
            end
            
            if frames==-1
                % code for plot all frames
                if this.stimLength==-1
                    error(['Must call setStimulusAttributes with ' ...
                        'attribute "stimLength" before plotting, ' ...
                        'or specify which frames to plot.']);
                end
                frames=1:this.stimLength;
            end
            
            nRows = length(names);
            
            set(gcf,'color',this.plotBgColor);
            if this.recordMovie
                set(gcf,'Position',[100 100 1400 600]);
                set(gcf,'PaperPositionMode','auto');
                
                outputVideo=this.recordMovieFile;
                % start the movie
                % Linux supports only uncompressed movies, where the parameter
                % quality does not take effect. Therefore set to 100.
                % TODO: Add Windows support with user-defined quality param.
                Mov = avifile(  outputVideo, ...
                    'fps', this.recordMovieFPS, ...
                    'quality', 100, ...
                    'compression', 'none');
            end
            
            % read all spike files and reshape all spike arrays
            spk = cell(1,length(names));
            for i=1:length(names)
                dims = this.getPopDims(names{i});
                
                % read spikes
                spkFile = this.getSpikeFileName(names{i});
                spk{i} = readSpikes(spkFile,this.stimFrameDur);
                
                % grow to right size
                len = max(frames(end),this.stimLength);
                if isempty(spk{i})
                    % group was silent
                    spk{i}(len,prod(dims))=0;
                else
                    if size(spk{i},1) ~= len
                        spk{i}(len,end) = 0;
                    end
                    if size(spk{i},2) ~= prod(dims)
                        spk{i}(end,prod(dims)) = 0;
                    end
                end
                
                % reshape according to population dimensions
                spk{i} = reshape(spk{i},len,dims(1),dims(2),dims(3));
                spk{i} = permute(spk{i},[3 2 4 1]); % Matlab: Y, X
                spk{i} = reshape(spk{i},dims(2),[],len);
            end
            
            % plot all frames
            for f=frames
                if this.plotAbortPlotting
                    % user pressed button to quit plotting
                    this.plotAbortPlotting = false;
                    close;
                    break
                end
                
                for i=1:nRows
                    subplot(nRows,1,i)
                    imagesc(spk{i}(:,:,f))
                    axis image
                    title(names{i})
                    xlabel('nrX')
                    ylabel('nrY')
                    set(gca,'XTick',0:dims(2):dims(2)*dims(3))
                end
                
                % add frame to movie or just draw
                if this.recordMovie
                    F=getframe(gcf);
                    Mov = addframe(Mov,F);
                else
                    drawnow
                end
                
                % wait for button press or pause
                if this.plotWaitButton
                    waitforbuttonpress;
                else
                    pause(1/this.plotFPS)
                end
            end
            
            if this.recordMovie
                Mov = close(Mov);
            end
        end
        
        function setSpikeFileAttributes(this,prefix,suffix)
            % obj.setSpikeFileAttributes(prefix,suffix)
            % Defines the naming conventions for spike files. They should
            % all reside within resultsFolder (specified in constructor), and
            % be made of a common prefix, the population name (specified in
            % ADDPOPULATION), and a common suffix.
            % Example: files 'results/spkV1.dat', 'results/spkMT.dat'
            %   -> resultsFolder = 'results/'
            %   -> prefix = 'spk'
            %   -> suffix = '.dat'
            %   -> name of population = 'V1' or 'MT'
            if nargin<3,suffix='.dat';end
            if nargin<2,prefix='spk';end
            
            this.spkFilePrefix=prefix;
            this.spkFileSuffix=suffix;
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
        
        function setStimulusAttributes(this,varargin)
            % obj.setStimulusAttributes(varargin)
            % Sets different stimulus attributes, such as the video length.
            % Example: obj.setStimulusAttributes('videoLength',10);
            % Call without attributes to set default values.
            % 
            % frameDur       - The number of milliseconds each frame represents.
            %                  This number will define the bin size for reading
            %                  the spike files. For example, if frameDur=100,
            %                  then all spikes will be grouped into bins of
            %                  width 100 ms.
            %
            % videoLength    - The number of frames in the stimulus video.
            
            if isempty(varargin)
                % set default values
                this.stimLength = -1;
                return;
            end
            
            % init error types
            throwErrNumeric = false;
            throwErrOutOfRange = false;
            
            nextIndex = 1;
            while nextIndex<length(varargin)
                attr = varargin{nextIndex};   % this one is attribute name
                val  = varargin{nextIndex+1}; % next is attribute value
                
                switch lower(attr)
                    case 'framedur'
                        % frame duration
                        throwErrNumeric = ~isnumeric(val);
                        reqRange = [1 inf];
                        throwErrOutOfRange = val<reqRange(1) | val>reqRange(2);
                        this.stimFrameDur = val;
                    case 'videolength'
                        % number of frames in video
                        throwErrNumeric = ~isnumeric(val);
                        this.stimLength = val;
                    otherwise
                        % attribute does not exist
                        if isnumeric(attr) || islogical(attr)
                            attr = num2str(attr);
                        end
                        error(['Unknown attribute "' attr '"'])
                end
                
                % throw errors
                if throwErrNumeric
                    error(['value for attr "' attr '" must be ' ...
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
        function bool = existsPopName(this,name)
            % bool = AM.existsPopName(name) checks whether a population has
            % been registered under name NAME
            bool = any(strcmpi(this.popNames,name));
        end

        function pauseOnKeyPressCallback(this,~,eventData)
            % Callback function to pause plotting
            switch eventData.Key
                case 'p'
                    disp('Paused. Press any key to continue.');
                    waitforbuttonpress;
                case 'q'
                    this.plotAbortPlotting = true;
            end
        end
    end
end