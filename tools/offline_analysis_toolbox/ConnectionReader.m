classdef ConnectionReader < handle
    % A ConnectionReader can be used to read a connection file that was
    % generated with the ConnectionMonitor utility in CARLsim. The user can
    % directly act on the returned connection data, to access weights at
    % specific times.
    %
    % To conveniently plot connection properties, please refer to
    % ConnectionMonitor.
    %
    % Example usage:
    % >> CR = ConnectionReader('results/conn_grp1_grp2.dat');
    % >> [allTimeStamps, allWeights] = CR.readWeights();
    % >> hist(allWeights(end,:))
    % >> % etc.
    %
    % Version 5/21/2015
    % Author: Michael Beyeler <mbeyeler@uci.edu>
    
    %% PROPERTIES
    % public
    properties (SetAccess = private)
        fileStr;             % path to connect file
        errorMode;           % program mode for error handling
        supportedErrorModes; % supported error modes
    end
    
    % private
    properties (Hidden, Access = private)
        fileId;                % file ID of spike file
        fileSignature;         % int signature of all spike files
        fileVersionMajor;      % required major version number
        fileVersionMinor;      % required minimum minor version number
        fileSizeByteHeader;    % byte size of header section
        fileSizeByteSnapshot;  % byte size of a single snapshot

        weights;
        timeStamps;
        nSnapshots;            % number of weight matrix snapshots

        minWt;                 % minimum weight magnitude of the connection
        maxWt;                 % maximum weight magnitude of the connection
        
        connId;
        grpIdPre;
        grpIdPost;
		gridPre;
		gridPost;
        nNeurPre;
        nNeurPost;
        nSynapses;
        isPlastic;
        
        errorFlag;           % error flag (true if error occured)
        errorMsg;            % error message
    end
    
    %% PUBLIC METHODS
    methods
        function obj = ConnectionReader(connectFile, errorMode)
            obj.fileStr = connectFile;
            obj.unsetError();
            obj.loadDefaultParams();
            
            if nargin<2
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
            if nargin<1
                obj.throwError('Path to connect file needed.');
                return
            end
            
            [~,~,fileExt] = fileparts(connectFile);
            if strcmpi(fileExt,'')
                obj.throwError(['Parameter connectFile must be a file ' ...
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
            % [errFlag,errMsg] = CR.getError() returns the current error
            % status.
            % If an error has occurred, errFlag will be true, and the
            % message can be found in errMsg.
            errFlag = obj.errorFlag;
            errMsg = obj.errorMsg;
		end
		
		function grid3D = getGrid3DPre(obj)
			% grid3D = CR.getGrid3DPre() returns the 3D grid dimensions for
			% the pre-synaptic group (1x3 vector)
			grid3D = obj.gridPre;
		end
        
		function grid3D = getGrid3DPost(obj)
			% grid3D = CR.getGrid3DPost() returns the 3D grid dimensions
			% for the post-synaptic group (1x3 vector)
			grid3D = obj.gridPost;
		end

        function maxWt = getMaxWeight(obj)
            % minWt = getMaxWeight() returns the maximum weight magnitude
            % of the connection
            maxWt = obj.maxWt;
        end

        function minWt = getMinWeight(obj)
            % minWt = getMinWeight() returns the minimum weight magnitude
            % of the connection
            minWt = obj.minWt;
        end
		
		function nNeurPre = getNumNeuronsPre(obj)
            % nNeurPre = CR.getNumNeuronsPre() returns the number of
            % neurons in the presynaptic group.
            nNeurPre = obj.nNeurPre;
        end
        
        function nNeurPost = getNumNeuronsPost(obj)
            % nNeurPre = CR.getNumNeuronsPost() returns the number of
            % neurons in the postsynaptic group.
            nNeurPost = obj.nNeurPost;
        end
        
        function nSnapshots = getNumSnapshots(obj)
            % nSnapshots = CR.getNumSnapshots() returns the number of
            % weight matrix snapshots.
            nSnapshots = obj.nSnapshots;
        end
        
        function [timeStamps, weights] = readWeights(obj, snapShots)
            if nargin<2 || isempty(snapShots) || snapShots==-1
                snapShots = 1:obj.nSnapshots;
            end
            
            if snapShots==0
                obj.throwError('snapShots must be a list of snapshots.')
                return
            end
            
            obj.timeStamps = [];
            obj.weights = [];
            
            for i=1:numel(snapShots)
                frame = snapShots(i);

                % rewind file pointer, skip header
                fseek(obj.fileId, obj.fileSizeByteHeader, 'bof');
                
                if frame>1
                    % skip (frame-1) snapshots
                    szByteToSkip = obj.fileSizeByteSnapshot*(frame-1);
                    status = fseek(obj.fileId, szByteToSkip, 'cof');
                    if status==-1
                        obj.throwError(ferror(obj.fileId))
                        return
                    end
                end
                
                % read data and append  to member
                obj.timeStamps = [obj.timeStamps fread(obj.fileId, 1, 'int64')];
                obj.weights(end+1,:) = fread(obj.fileId, obj.nNeurPre*obj.nNeurPost, 'float32');
            end
            timeStamps = obj.timeStamps;
            weights = obj.weights;
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
            obj.fileSignature = 202029319;
            obj.fileVersionMajor = 0;
            obj.fileVersionMinor = 3;
            obj.fileSizeByteHeader = -1;   % to be set in openFile
            obj.fileSizeByteSnapshot = -1; % to be set in openFile
            
            obj.timeStamps = [];  % to be set in readWeights
            obj.weights = [];     % to be set in readWeights
            obj.connId = -1;
            obj.grpIdPre = -1;
            obj.grpIdPost = -1;
            obj.nNeurPre = -1;
            obj.nNeurPost = -1;
            obj.nSynapses = -1;
            obj.isPlastic = false;
            obj.nSnapshots = -1;
            
            obj.supportedErrorModes = {'standard', 'warning', 'silent'};

			% disable backtracing for warnings and errors
			warning off backtrace
        end
        
        function openFile(obj)
            % SR.openFile() reads the header section of the spike file and
            % sets class properties appropriately.
            obj.unsetError()
            
            % try to open connect file, use little-endian
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
            
            % read connection ID
            obj.connId = fread(obj.fileId, 1, 'int16');
            if feof(obj.fileId) || obj.connId<0
                obj.throwError(['Could not find valid connection ID.'])
                return
            end
            
            % read pre-group info
            obj.grpIdPre = fread(obj.fileId, 1, 'int32');
			obj.gridPre  = fread(obj.fileId, [1 3],'int32');
            obj.nNeurPre = prod(obj.gridPre);
            if feof(obj.fileId) || obj.grpIdPre<0 || obj.nNeurPre<=0 || sum(obj.gridPre<=0)>0
                obj.throwError(['Could not find valid pre-group info ' ...
					'(grpId=' num2str(obj.grpIdPre) ', nNeur=' ...
					num2str(obj.nNeurPre) ', grid=[' ...
					num2str(obj.gridPre(1)) ' ' num2str(obj.gridPre(2)) ...
					' ' num2str(obj.gridPre(3)) '])'])
                return
            end
            
            % read post-group info
            obj.grpIdPost = fread(obj.fileId, 1, 'int32');
			obj.gridPost  = fread(obj.fileId, [1 3],'int32');
            obj.nNeurPost = prod(obj.gridPost);
            if feof(obj.fileId) || obj.grpIdPost<0 || obj.nNeurPost<=0 || sum(obj.gridPost<=0)>0
                obj.throwError(['Could not find valid post-group info ' ...
					'(grpId=' num2str(obj.grpIdPost) ', nNeur=' ...
					num2str(obj.nNeurPost) ', grid=[' ...
					num2str(obj.gridPost(1)) ' ' num2str(obj.gridPost(2)) ...
					' ' num2str(obj.gridPost(3)) '])'])
                return
            end
            
            % read number of synapses
            obj.nSynapses = fread(obj.fileId, 1, 'int32');
            if feof(obj.fileId) || obj.nSynapses<0
                obj.throwError(['Could not find valid number of ' ...
					'synapses (' num2str(obj.nSynapses) ')'])
                return
            end
            
            % read isPlastic
            obj.isPlastic = fread(obj.fileId, 1, 'bool');

            % read minWt and maxWt
            obj.minWt = fread(obj.fileId, 1, 'float32');
            obj.maxWt = fread(obj.fileId, 1, 'float32');
            if (obj.minWt<0) || (obj.maxWt<0)
                obj.throwError(['Could not find valid minWt/maxWt ' ...
                    'magnitudes (min=' num2str(obj.minWt) ',max=' ...
                        num2str(obj.maxWt) ')'])
            end
            
            % store the size of the header section, so that we can skip it
            % when re-reading spikes
            obj.fileSizeByteHeader = ftell(obj.fileId);
            
            % find size of each snapshot: #weights * sizeof(float32) +
            % sizeof(long int)
            obj.fileSizeByteSnapshot = obj.nNeurPre*obj.nNeurPost*4+8;

            % compute number of snapshots present in the file
            % find byte size from here on until end of file, divide it by
            % byte size of each snapshot -> number of snapshots
            fseek(obj.fileId, 0, 'eof');
            szByteTot = ftell(obj.fileId);
            obj.nSnapshots = floor( (szByteTot-obj.fileSizeByteHeader) ...
                / obj.fileSizeByteSnapshot );
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