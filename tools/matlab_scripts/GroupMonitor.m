classdef GroupMonitor < handle
    %
    % Version 10/2/2014
    % Author: Michael Beyeler <mbeyeler@uci.edu>
    
    %% PROPERTIES
    % public
    properties (SetAccess = private)
        name;               % group name
        resultsFolder;      % results folder
    end

    % private
    properties (Hidden, Access = private)
        Spk;                % SpikeReader object
        spkFilePrefix;      % spike file prefix, e.g. "spk"
        spkFileSuffix;      % spike file suffix, e.g. ".dat"
        spkData;

        simLength;
        grid3D;

        errorMode;          % program mode for error handling
        errorFlag;          % error flag (true if error occured)
        errorMsg;           % error message

        supportedErrorModes;% supported error modes
    end
    
    
    %% PUBLIC METHODS
    methods
        function obj = GroupMonitor(name, resultsFolder, errorMode)
            obj.name = name;
            obj.unsetError()
            obj.loadDefaultParams();

            if nargin<3
                obj.errorMode = 'standard';
            else
                if ~obj.isErrorModeSupported(errorMode)
                    obj.throwError(['errorMode "' errorMode '" is currently' ...
                        ' not supported. Choose from the following: ' ...
                        strjoin(obj.supportedErrorModes, ', ') '.'], ...
                        'standard')
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
                obj.throwError('Path to spike file needed.');
                return
            end
        end
        
        function delete(obj)
            % destructor, implicitly called
        end

        function [errFlag,errMsg] = getError(obj)
            % [errFlag,errMsg] = getError() returns the current error status.
            % If an error has occurred, errFlag will be true, and the message
            % can be found in errMsg.
            errFlag = obj.errorFlag;
            errMsg = obj.errorMsg;
        end

        function spkFile = getSpikeFileName(obj)
            % spkFile = AM.getSpikeFileName() returns the name of the
            % spike file according to specified prefix and suffix.
            % Prefix and suffix can be set using AM.setSpikeFileAttributes.
            spkFile = [ obj.resultsFolder ...  % the results folder
                filesep ...                    % platform-specific separator
                obj.spkFilePrefix ...          % something like 'spk_'
                obj.name ...                   % the name of the group
                obj.spkFileSuffix ];           % something like '.dat'
        end

        function plot(obj, frames, dispFrameNr)
            obj.loadData()

            if nargin<3,dispFrameNr = true;end
            if nargin<2,frames = 1:obj.simLength;end
            
            % reset abort flag, set up callback for key press events
            this.plotAbortPlotting = false;
            set(gcf,'KeyPressFcn',@this.privPauseOnKeyPressCallback)
                        
            % display frame in specified axes
            for i=frames
                if this.plotAbortPlotting
                    % user pressed button to quit plotting
                    this.plotAbortPlotting = false;
                    close;
                    return
                end
                
                obj.plotHeatMap(obj.spkData(:,:,i))
                if dispFrameNr
                    text(2,size(obj.spkData,1)-1,num2str(i), ...
                        'FontSize',10,'BackgroundColor','white')
                end
                drawnow
                pause(0.1)
            end
        end

        function plotHeatMap(obj, data2D)
            imagesc(data2D)
            axis image
            title(obj.name)
            xlabel('nrX')
            ylabel('nrY')
            set(gca, 'XTick', 0:obj.grid3D(2):obj.grid3D(2)*obj.grid3D(3))
        end

        function loadData(obj, frameDur)
            if nargin<2,frameDur=1000;end

            spkFile = obj.getSpikeFileName();
            obj.Spk = SpikeReader(spkFile, obj.errorMode);
            obj.grid3D = obj.Spk.getGrid3D();

            spkBuffer = obj.Spk.readSpikes(frameDur);
            numFrames = size(spkBuffer,1);

            % reshape according to population dimensions
            spkBuffer = reshape(spkBuffer, numFrames, ...
                obj.grid3D(1), obj.grid3D(2), ...
                obj.grid3D(3));
            spkBuffer = permute(spkBuffer,[3 2 4 1]); % Matlab: Y, X
            spkBuffer = reshape(spkBuffer, obj.grid3D(2), [], numFrames);

            obj.spkData = spkBuffer;
            obj.simLength = numFrames;
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
            
            obj.spkFilePrefix=prefix;
            obj.spkFileSuffix=suffix;
        end
    end
    
    %% PRIVATE METHODS
    methods (Hidden, Access = private)
        function isSupported = isErrorModeSupported(obj, errMode)
            % determines whether an error mode is currently supported
            isSupported = sum(ismember(obj.supportedErrorModes,errMode))>0;
        end

        function loadDefaultParams(obj)
            obj.Spk = [];
            obj.setSpikeFileAttributes()

            obj.grid3D = -1;
            obj.spkData = [];
            obj.simLength = -1;

            obj.supportedErrorModes = {'standard', 'warning', 'silent'};
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

        function unsetError(obj)
            % unsets error message and flag
            obj.errorFlag = false;
            obj.errorMsg = '';
        end
    end
end