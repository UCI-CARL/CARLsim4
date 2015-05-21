classdef SpikeReader < handle
    % A SpikeReader can be used to read spike files that were generated
    % with the SpikeMonitor utility in CARLsim. The user can then directly
    % act on the returned spike data, be it in AER format or binned in
    % suitable time windows.
    %
    % To conveniently plot group activity, please refer to GroupMonitor.
    %
    % Example usage:
    % >> SR = SpikeReader('results/spk_group1.dat');
    % >> binWinMs = 100;
    % >> spkData = SR.readSpikes(binWinMs);
    % >> % analyze spike data ...
    % >> stimLengthMs = SR.getSimDurMs();
    % >> % etc.
    %
    % Version 5/21/2015
    % Author: Michael Beyeler <mbeyeler@uci.edu>
    
    %% PROPERTIES
    % public
    properties (SetAccess = private)
        fileStr;             % path to spike file
        errorMode;           % program mode for error handling
        supportedErrorModes; % supported error modes
    end
    
    % private
    properties (Hidden, Access = private)
        fileId;              % file ID of spike file
        fileSignature;       % int signature of all spike files
        fileVersionMajor;    % required major version number
        fileVersionMinor;    % required minimum minor version number
        fileSizeByteHeader;  % byte size of header section
        
        grid3D;              % 3D grid dimensions of group
        binWindow;           % binning window for spike times (ms)
        stimLengthMs;        % estimated simulation duration (determined by
        % last spike time)
        storeSpikes;         % flag whether to store/buffer spike data
        spkData;             % the buffered spike data
        
        errorFlag;           % error flag (true if error occured)
        errorMsg;            % error message
    end
    
    
    %% PUBLIC METHODS
    methods
        function obj = SpikeReader(spikeFile, storeSpikes, errorMode)
            % SR = SpikeReader(spikeFile) creates a new instance of class
            % SpikeReader, which can be used to read spike files generated
            % by the SpikeMonitor utility in CARLsim.
            %
            % SPIKEFILE   - Path to spike file (expects data to be in
            %               Address-Event-Representation AER: spike time
            %               (ms) followed by neuron ID (both int32)), like
            %               the ones created by the CARLsim SpikeMonitor
            %               utility.
            % STORESPIKES - Flag whether to store/buffer the retrieved
            %               spike data. If enabled, repeated calls to
            %               SR.readSpikes will returned the buffered data
            %               instead of repeatedly re-reading the spike
            %               file. Default: true.
            % ERRORMODE   - Error Mode in which to run SpikeReader. The
            %               following modes are supported:
            %                 - 'standard' Errors will be fatal (returned
            %                              via Matlab function error())
            %                 - 'warning'  Errors will be warnings
            %                              returned via Matlab function
            %                              warning())
            %                 - 'silent'   No exceptions will be thrown,
            %                              but object will populate the
            %                              properties errorFlag and
            %                              errorMsg.
            %               Default: 'standard'.
            obj.fileStr = spikeFile;
            obj.unsetError()
            obj.loadDefaultParams();
            
            if nargin<3
                obj.errorMode = 'standard';
            else
                if ~obj.isErrorModeSupported(errorMode)
                    obj.throwError(['errorMode "' errorMode '" is ' ...
                        ' currently not supported. Choose from the ' ...
                        'following: ' ...
                        strjoin(obj.supportedErrorModes, ', ') '.'], ...
                        'standard')
                    return
                end
                obj.errorMode = errorMode;
            end
            if nargin<2
                obj.storeSpikes = true;
            else
                obj.storeSpikes = storeSpikes;
            end
            if nargin<1
                obj.throwError('Path to spike file needed.');
                return
            end
            
            [~,~,fileExt] = fileparts(spikeFile);
            if strcmpi(fileExt,'')
                obj.throwError(['Parameter spikeFile must be a file ' ...
                    'name, directory found.'])
            end
            
            % move unsafe code out of constructor
            obj.openFile()
        end
        
        function delete(obj)
            % destructor, implicitly called to fclose file
            if obj.fileId ~= -1
                fclose(obj.fileId);
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
        
        function grid3D = getGrid3D(obj)
            % grid3D = SR.getGrid3D() returns the 3D grid dimensions of the
            % group as three-element vector <width x height x depth>. This
            % property is equal to the one set in CARLsim::createGroup.
            grid3D = obj.grid3D;
        end
        
        function simDurMs = getSimDurMs(obj)
            % simDurMs = SR.getSimDurMs() returns the estimated simulation
            % duration in milliseconds. This is equivalent to the time
            % stamp of the last spike that occurred. Other than that, a
            % SpikeReader does know about any simulation procedures.
            %
            % The total simulation duration is usually stored in a
            % "sim_{simName}.dat" file and can be retrieved by using a
            fseek(obj.fileId, -8, 'eof'); % jump to penultimate int
            simDurMs = fread(obj.fileId, 1, 'int32');
        end
        
        function spk = readSpikes(obj, binWindowMs)
            % spk = SR.readSpikes(binWindow) reads the spike file and
            % arranges spike times into bins of binWindow millisecond
            % length.
            %
            % Returns a 2-D matrix (spike times x neuron IDs), 1-indexed.
            %
            % BINWINDOWMS - Size of binning window for spike times (ms).
            %               Set binWindow to -1 in order to get the spikes
            %               in AER format [times;nIDs].
            %               Default: 1000.
            if nargin<2,binWindowMs=1000;end
            obj.unsetError()
            
            % if storeSpikes flag is set, we may not have to read the data
            % again (given that the user requires reading with the same
            % frame duration)
            if obj.storeSpikes
                if sum(obj.binWindow==binWindowMs) && ~isempty(obj.spkData)
                    % we don't need to re-read the data
                    spk = obj.spkData;
                    return;
                end
            end
            obj.binWindow = binWindowMs;
            
            % rewind file pointer, skip header
            fseek(obj.fileId, obj.fileSizeByteHeader, 'bof');
            
            nrRead=1e6;
            d=zeros(0,nrRead);
            spk=[];
            
            while size(d,2)==nrRead
                % D is a 2xNRREAD matrix.  Row 1 contains the times that
                % the neuron spiked. Row 2 contains the neuron id that
                % spiked at this corresponding time.
                d = fread(obj.fileId, [2 nrRead], 'int32');

                if ~isempty(d)
                    if obj.binWindow<0
                        % Return data in AER format, i.e.: [time;nID]
                        % Note: Using SPARSE on large matrices that mostly
                        % contain 0 is inefficient (-> "big sparse matrix")
                        spk = [spk, d];
                    else
                        % Resulting matrix s will have rows corresponding
                        % to time values with a minimum value of 1 and
                        % columns organized by neuron ids that are indexed
                        % starting with 1.  FRAMEDUR effectively bins the
                        % data. FRAMEDUR=1 bins at 1 ms, FRAMEDUR=1000 bins
                        % at 1000 ms, etc.

                        % Use sparse matrix to create a matrix S with
                        % correct dimensions. All firing events for each
                        % neuron id and time bin are summed automatically
                        % with ACCUMARRAY.  Finally the matrix is resized
                        % to include all the zero entries with the correct
                        % matrix dimensions. ACCUMARRAY is supposed to be
                        % faster than full(sparse(...)). Make sure the
                        % first two arguments are column vectors.
                        subs = [floor(d(1,:)/obj.binWindow)'+1,d(2,:)'+1];

                        % Make sure spk has the right dimensions (defined
                        % by the max values in subs)
                        maxDim = max(subs);
                        if size(spk,1)<maxDim(1) || size(spk,2)<maxDim(2)
                            spk(maxDim(1),maxDim(2))=0;
                        end
                        spk = spk + accumarray(subs, 1, size(spk));
                    end
                end
            end
            
            % grow to right size
            if size(spk,2) ~= prod(obj.grid3D)
                spk(max(1,end),prod(obj.grid3D))=0;
            end
            
            % store spike data
            if obj.storeSpikes
                obj.spkData = spk;
            end
            
            % extract stimulus length
            obj.stimLengthMs = obj.getSimDurMs();
        end
    end
    
    %% PRIVATE METHODS
    methods (Hidden, Access = private)
        function isSupported = isErrorModeSupported(obj, errMode)
            % determines whether an error mode is currently supported
            isSupported = sum(ismember(obj.supportedErrorModes,errMode))>0;
        end
        
        function loadDefaultParams(obj)
            % loads default parameter values for class properties
            obj.fileId = -1;
            obj.fileSignature = 206661989;
            obj.fileVersionMajor = 0;
            obj.fileVersionMinor = 2;
            obj.fileSizeByteHeader = -1; % to be set in openFile
            
            obj.grid3D = -1; % to be set in openFile
            
            obj.spkData = []; % to be set in readSpikes
            obj.stimLengthMs = -1;
            
            obj.supportedErrorModes = {'standard', 'warning', 'silent'};

			% disable backtracing for warnings and errors
			warning off backtrace
        end
        
        function openFile(obj)
            % SR.openFile() reads the header section of the spike file and
            % sets class properties appropriately.
            obj.unsetError()
            
            % try to open spike file, try little-endian
            obj.fileId = fopen(obj.fileStr, 'r', 'l');
            if obj.fileId==-1
                obj.throwError(['Could not open file "' obj.fileStr ...
                    '" with read permission'])
                return
            end
            
            % read signature
            sign = fread(obj.fileId, 1, 'int32');
            if feof(obj.fileId)
                obj.throwError('File is empty.');
            else
                if sign~=obj.fileSignature
                    % try big-endian instead
                    fclose(obj.fileId);
                    obj.fileId = fopen(obj.fileStr, 'r', 'b');
                    sign = fread(obj.fileId, 1, 'int32');
                    if sign~=obj.fileSignature
                        obj.throwError(['Unknown file type: ' num2str(sign)]);
                        return
                    end
                end
            end          

            % read version number
            version = fread(obj.fileId, 1, 'float32');
            if feof(obj.fileId) || floor(version) ~= obj.fileVersionMajor
                % check major number: must match
                obj.throwError(['File must be of version ' ...
                    num2str(obj.fileVersionMajor) '.x (Version ' ...
                    num2str(version) ' found'])
                return
            end
            if feof(obj.fileId) ...
					|| floor((version-obj.fileVersionMajor)*10.01)<obj.fileVersionMinor
                % check minor number: extract first digit after decimal
                % point
                % multiply 10.01 instead of 10 to avoid float rounding
                % errors
                obj.throwError(['File version must be >= ' ...
                    num2str(obj.fileVersionMajor) '.' ...
                    num2str(obj.fileVersionMinor) ' (Version ' ...
                    num2str(version) ' found)'])
                return
            end
            
            % read Grid3D
            obj.grid3D = fread(obj.fileId, [1 3], 'int32');
            if feof(obj.fileId) || prod(obj.grid3D)<=0
                obj.throwError(['Could not find valid Grid3D ' ...
					'dimensions (grid=[' num2str(obj.grid3D(1)) ' ' ...
					num2str(obj.grid3D(2)) ' ' num2str(obj.grid3D(3)) ...
					'])'])
                return
            end
            
            % store the size of the header section, so that we can skip it
            % when re-reading spikes
            obj.fileSizeByteHeader = ftell(obj.fileId);
        end
        
        function throwError(obj, errorMsg, errorMode)
            % SR.throwError(errorMsg, errorMode) throws an error with a
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