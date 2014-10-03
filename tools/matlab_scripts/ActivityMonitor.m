classdef ActivityMonitor < handle
    % Version 10/3/2014
    % Author: Michael Beyeler <mbeyeler@uci.edu>
    
    %% PROPERTIES
    % public
    properties (SetAccess = private)
        resultsFolder;      % results directory where all spike files live
        simFile;            % simulation file "sim_{simName}.dat"
        
        groupNames;         % cell array of population names
        groupGrid3D;        % cell array of population dimensions
        groupPlotTypes;     % cell array of population plot types

        numSubPlots;
    end
    
    % private
    properties (Hidden, Access = private)
        Sim;                % instance of SimulationReader class
        spkFilePrefix;      % spike file prefix, e.g. "spk"
        spkFileSuffix;      % spike file suffix, e.g. ".dat"
        
        errorMode;          % program mode for error handling
        errorFlag;          % error flag (true if error occured)
        errorMsg;           % error message

        groupSpkObj;        % cell array of SpikeReader objects
        groupSubPlots;      % cell array of assigned subplot slots

        supportedPlotTypes; % cell array of supported plot types
        supportedErrorModes;% supported error modes
    end
    
    
    %% PUBLIC METHODS
    methods
        function obj = ActivityMonitor(simFile, loadGroupsFromFile, errorMode)
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
            
            [filePath,fileName,fileExt] = fileparts(simFile);
            if strcmpi(fileExt,'')
                obj.throwError(['Parameter simFile must be a file name, ' ...
                    'directory found.'])
            end
            
            obj.resultsFolder = filePath;
            obj.simFile = [fileName fileExt];
                        
            % try to read simulation file
            obj.readSimulationFile()
            
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
            %                   e Thproduct of these dimensions should equal the
            %                   total number of neurons in the population.
            %                   By default, this parameter will be read from
            %                   file, but can be overwritten by the user.
            if nargin<6,errorMode=obj.errorMode;end
            if nargin<5,subPlots=[];end
            if nargin<4,grid3D=-1;end
            if nargin<3,plotType='default';end

            % find index of group in Sim struct
            indStruct = obj.getGroupStructId(name);
            if indStruct<=0
                obj.throwError(['Group "' name '" could not be found. ' ...
                    'Choose from the following: ' ...
                    strjoin({Random.groups(:).name}, ', ') '.'])
                return
            end
            
            % check whether valid spike file found (run SpikeReader in silent)
            spkFile = obj.getSpikeFileName(name);
            Spk = SpikeReader(spkFile, errorMode);
            [errFlag,errMsg] = Spk.getError();
            if errFlag
                obj.throwError(errMsg, errorMode);
                return % make sure we exit after spike file could not be found
            end
            
            % read Grid3D from Sim struct if necessary
            if prod(grid3D)<0
                grid3D = obj.Sim.groups(ind).grid3D;
            end
            
            % assign default plotType if necessary
            if strcmpi(plotType,'default')
                if grid3D(1)>1 && grid3D(2)>1 && grid3D(3)==1
                    plotType = 'heatmap';
                else
                    plotType = 'raster';
                end
            end

            % make sure plotType is supported
            if ~obj.isPlotTypeSupported(plotType)
                obj.throwError(['plotType "' plotType '" is currently not ' ...
                    'supported. Choose from the following: ' ...
                    strjoin(obj.supportedPlotTypes, ', ') '.'])
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
                obj.groupGrid3D{id}       = grid3D;
                obj.groupPlotTypes{id}    = plotType;
                obj.groupSpkObj{id}       = Spk;
                obj.numSubPlots           = obj.numSubPlots ...
                                            - numel(obj.groupSubplots{id}) ...
                                            + numel(subPlots);
                obj.groupSubPlots{id}     = subPlots;
            else
                % else add new entry
                obj.groupNames{end+1}     = name;
                obj.groupGrid3D{end+1}    = grid3D;
                obj.groupSpkObj{end+1}    = Spk;
                obj.groupPlotTypes{end+1} = plotType;
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
        
        function spkFile = getSpikeFileName(obj,name)
            % spkFile = AM.getSpikeFileName(name) returns the name of the
            % spike file according to specified prefix and suffix.
            % Prefix and suffix can be set using AM.setSpikeFileAttributes.
            %
            % NAME  - A string representing the name of a group that has been
            %         registered by calling AM.addPopulation.
            spkFile = [ obj.resultsFolder ...  % the results folder
                filesep ...                    % platform-specific separator
                obj.spkFilePrefix ...          % something like 'spk_'
                name ...                       % the name of the group
                obj.spkFileSuffix ];           % something like '.dat'
        end
        
        function plot(obj, groupNames)
            if nargin<2,groupNames={};end

            if isempty(groupNames)
                groupNames = obj.groupNames;
            end

            [nrR, nrC] = obj.findPlotLayout(obj.numSubPlots);

            % prepare for plotting
            % for plotting we need to keep all extract spike files
            frameDur = 100;
            numFrames = ceil(obj.Sim.sim.simTimeSec*1000.0/frameDur);
            spkBuffer = cell(1,numel(groupNames));
            for i=1:numel(groupNames)
                gId = obj.getGroupId(groupNames{i});

                % read spikes from file
                spkBuffer{i} = obj.groupSpkObj{gId}.readSpikes(frameDur);

                % reshape according to population dimensions
                spkBuffer{i} = reshape(spkBuffer{i}, numFrames, ...
                    obj.groupGrid3D{gId}(1), obj.groupGrid3D{gId}(2), ...
                    obj.groupGrid3D{gId}(3));
                spkBuffer{i} = permute(spkBuffer{i},[3 2 4 1]); % Matlab: Y, X
                spkBuffer{i} = reshape(spkBuffer{i}, obj.groupGrid3D{gId}(2), [], numFrames);
            end

            % plot all frames
            for f=1:numFrames
                for g=1:numel(groupNames)
                    gId = obj.getGroupId(groupNames{g});
                    subplot(nrR, nrC, obj.groupSubPlots{gId})

                    if strcmpi(obj.groupPlotTypes{gId},'heatmap')
                        obj.plotHeatMap(groupNames{g}, spkBuffer{g}(:,:,f))
                    elseif strcmpi(obj.groupPlotTypes{gId},'raster')
                        % obj.plotRaster(spkBuffer{g}(:,:,f))
                        obj.plotHeatMap(groupNames{g}, spkBuffer{g}(:,:,f))
                    else
                        % we really shouldn't be here
                        obj.throwError(['Unrecognized plotType ' obj.groupPlotTypes{gId}])
                    end
                end
                drawnow
                
            end
        end
    end
    
    %% PRIVATE METHODS
    methods (Hidden, Access = private)
        function addAllGroupsFromFile(obj)
            % add groups in silent mode, so that no errors are thrown if not
            % all spike files are found
            errMode = 'silent';
            
            for i=1:numel(obj.Sim.groups)
                obj.addGroup(obj.Sim.groups(i).name, 'default', ...
                    obj.Sim.groups(i).grid3D, [], errMode)
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
            % finds the index in the Sim struct for a group name
            
            % convert struct to flattened cell array
            cellSim = reshape(struct2cell(obj.Sim.groups),1,[]);
            
            % find index of group name
            [~,j] = find(strcmpi(cellSim,name));
            if isempty(j)
                % group not found
                index = -1;
            else
                % convert back to struct index
                index = (j-1)/numel(fieldnames(obj.Sim.groups))+1;
            end
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
            obj.spkFilePrefix = 'spk';
            obj.spkFileSuffix = '.dat';
            
            obj.supportedErrorModes = {'standard', 'warning', 'silent'};
            obj.supportedPlotTypes = {'heatmap','raster'};
            
            obj.groupNames = {};
            obj.groupGrid3D = {};
            obj.groupPlotTypes = {};
            obj.groupSubPlots = {};
            obj.groupSpkObj = {};

            obj.numSubPlots = 0;
        end

        function plotHeatMap(obj, group, data2D)
            dims = obj.groupGrid3D{obj.getGroupId(group)};

            imagesc(data2D)
            axis image
            title(group)
            xlabel('nrX')
            ylabel('nrY')
            set(gca, 'XTick', 0:dims(2):dims(2)*dims(3))
        end
        
        function readSimulationFile(obj)
            % try and read simulation file
            obj.Sim = SimulationReader([obj.resultsFolder filesep obj.simFile]);
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