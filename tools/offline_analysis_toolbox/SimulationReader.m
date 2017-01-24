classdef SimulationReader < handle
    % SR = SimulationReader(simFile) creates a new instance of class
    % SimulationReader.
    %
    % SimulationReader can be used to read a simulation log file that is
    % created by default ("results/sim_[netName].dat") or by specifically
    % calling method CARLsim::saveSimulation.
    % The returned object contains essential information about the
    % simulation and the network setup.
    %
    % Usage example:
    % >> SR = SimulationReader('results/sim_random.dat');
    % >> SR.sim.simTimeSec % print number of simulated seconds
    % >> SR.sim.exeTimeSec % print execution time in seconds
    %
    % If the simulation file was created with saveSynapseInfo set to true
    % in CARLsim::saveSimulation, then the file will also contain all
    % synapse information (weights, delays, maxWeights, etc.).
    %
    % To read all the synapse information in MATLAB, use an optional input
    % argument loadSynapseInfo:
    % >> SR = SimulationReader('results/sim_random'dat',true);
    % >> hist(SR.sim.syn_weights)
    %
    % Note: Use of syn_* properties is deprecated. Use ConnectionMonitor or
    % ConnectionReader instead.
    %
    % Version 5/21/2015
    % Author: Michael Beyeler <mbeyeler@uci.edu>
    
    %% PROPERTIES
    % public
    properties (SetAccess = private)
        fileStr;            % path to spike file
        
        sim;                % a struct that contains general information
                            % about the simulation and the network
        groups;             % a struct that contains all group info
        syn_preIDs;
        syn_postIDs;
        syn_weights;
        syn_maxWeights;
        syn_plastic;
        syn_delays;
    end

    % private
    properties (Hidden, Access = private)
        fileId;             % file ID of spike file
        fileSignature;      % int signature of all spike files
        fileVersionMajor;   % required major version number
        fileVersionMinor;   % required minimum minor version number
        fileSizeByteHeader; % byte size of header section        
    end
    
    
    %% PUBLIC METHODS
    methods
        function obj = SimulationReader(simFile, loadSynapseInfo)
            % SR = SimulationReader(simFile, loadSynapseInfo) creates a
            % new instance of class SimulationReader, which can be used
            % to read CARLsim simulation log files.
            %
            % SIMFILE          - Path to simulation log file
            % LOADSYNAPSEINFO  - A flag indicating whether to read all
            %                    detailed synapse information. If set to
            %                    false, only the general network structure
            %                    will be extracted. Default: false.
            if nargin<1,error('Path to spike file needed'),end
            if nargin<2,loadSynapseInfo=false;end
            
            obj.fileStr = simFile;
            obj.privLoadDefaultParams();
            
            % move unsafe code out of constructor
            obj.privOpenFile(loadSynapseInfo)
        end
        
        function delete(obj)
            % destructor, implicitly called to fclose file
            if obj.fileId ~= -1
                fclose(obj.fileId);
            end
        end
    end
    
        
    %% PRIVATE METHODS
    methods (Hidden, Access = private)
        function privLoadDefaultParams(obj)
            obj.fileId = -1;
            obj.fileSignature = 294338571;
            obj.fileVersionMajor = 0;
            obj.fileVersionMinor = 2;

			% disable backtracing for warnings and errors
			warning off backtrace
        end
        
        function privOpenFile(obj, loadSynapseInfo)
            % try to open sim file, use little-endian
            fid = fopen(obj.fileStr, 'r', 'l');
            if fid==-1
                error(['Could not open file "' obj.fileStr ...
                    '" with read permission'])
            end
            
            % read signature
            sign = fread(fid, 1, 'int32');
            if feof(fid)
                error('File is empty')
            else
                if sign~=obj.fileSignature
                    % try big-endian instead
                    fclose(fid);
                    fid = fopen(obj.fileStr, 'r', 'b');
                    sign = fread(fid, 1, 'int32');
                    if sign~=obj.fileSignature
                        obj.throwError(['Unknown file type: ' num2str(sign)]);
                        return
                    end
                end
            end
            
            % read version number
            version = fread(fid, 1, 'float32');
            if feof(fid) || floor(version) ~= obj.fileVersionMajor
                % check major number: must match
                error(['File must be of version ' ...
                    num2str(obj.fileVersionMajor) '.x (Version ' ...
                    num2str(version) ' found'])
            end
            if feof(fid) ...
					|| floor((version-obj.fileVersionMajor)*10.01)<obj.fileVersionMinor
                % check minor number: extract first digit after decimal point
                % multiply 10.01 instead of 10 to avoid float rounding errors
                error(['File version must be >= ' ...
                    num2str(obj.fileVersionMajor) '.' ...
                    num2str(obj.fileVersionMinor) ' (Version ' ...
                        num2str(version) ' found)'])
            end
            
            % read simulation info
            sim = struct();
            sim.simTimeSec = fread(fid,1,'float'); % in seconds
            sim.exeTimeSec = fread(fid,1,'float'); % in seconds
            % \TODO more params could be added:
            % read sim mode
            % read logger mode
            % ithGPU
            % nconfig
            % randSeed
            
            % read network info
            sim.nNeurons      = fread(fid,1,'int32');
            sim.nSynapsesPre  = fread(fid,1,'int32');
            sim.nSynapsesPost = fread(fid,1,'int32');
            sim.nGroups       = fread(fid,1,'int32');
            % \TODO more params could be added:
            % sim_with_fixedwts
            % sim_with_conductances
            % sim_with_NMDA_rise
            % sim_with_GABAb_rise
            % sim_with_stdp
            % sim_with_modulated_stdp
            % sim_with_homeostasis
            % sim_with_stp
            obj.sim = sim;
			
            
            %% READ GROUPS
            groups = struct('name',{},'startN',{},'endN',{},'sizeN',{},'grid3D',{});
            for g=1:sim.nGroups
                groups(g).startN = fread(fid,1,'int32'); % start index at 0
                groups(g).endN = fread(fid,1,'int32');
                groups(g).sizeN = groups(g).endN-groups(g).startN+1;

                sizeX = fread(fid,1,'int32');
                sizeY = fread(fid,1,'int32');
                sizeZ = fread(fid,1,'int32');
                groups(g).grid3D = [sizeX sizeY sizeZ];

                groups(g).name = char(fread(fid,100,'int8')');
                groups(g).name = groups(g).name(groups(g).name>0);
            end
            obj.groups = groups;
            
            %% READ SYNAPSES
            % reading synapse info is optional
            if loadSynapseInfo
                weightData = cell(sim.nNeurons,1);
                nrSynTot = 0;
                for i=1:sim.nNeurons
                    nrSyn = fread(fid,1,'int32');
                    nrSynTot = nrSynTot + nrSyn;
                    if nrSyn>0
                        weightData{i} = fread(fid,[18 nrSyn],'uint8=>uint8');
                    end
				end
				
                alldata = cat(2,weightData{:});
                clear weightData;
                
                obj.syn_preIDs = typecast(reshape(alldata(1:4,:),[],1),'uint32');
                obj.syn_postIDs = typecast(reshape(alldata(5:8,:),[],1),'uint32');
                obj.syn_weights = typecast(reshape(alldata(9:12,:),[],1),'single');
                obj.syn_maxWeights = typecast(reshape(alldata(13:16,:),[],1),'single');
                obj.syn_delays = alldata(17,:);
                obj.syn_plastic = alldata(18,:);
            end
            
            obj.fileId = fid;
            
        end
    end
end