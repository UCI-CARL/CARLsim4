function [sim,groups,preIDs,postIDs,weights,delays,plastic,maxWeights] = loadSimulation(filename, loadSynapseInfo)
% FILENAME       - relative or absolute path to a binary file
%                  containing a VisualStimulus object.
if nargin<2,loadSynapseInfo=false;end
if nargin<1,error('No filename given');end

fid = fopen(filename,'r');
if fid==-1
	error(['Could not open "' fileName '" with read permission.']);
end


%% READ HEADER SECTION

% read signature
sign = fread(fid,1,'int');
if sign~=294338571
    error('Unknown file type')
end

% read version number
version = fread(fid,1,'float');
if (version ~= 1.0)
    error(['Unknown file version, must have Version 1.0 (Version ' ...
        num2str(version) ' found)'])
end

% read simulation info
sim = struct();
sim.simTimeSec = fread(fid,1,'float'); % in seconds
sim.exeTimeSec = fread(fid,1,'float'); % in seconds
% more params could be added:
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
% more params could be added:
% sim_with_fixedwts
% sim_with_conductances
% sim_with_NMDA_rise
% sim_with_GABAb_rise
% sim_with_stdp
% sim_with_modulated_stdp
% sim_with_homeostasis
% sim_with_stp


%% READ GROUPS
groups = struct('name',{},'startN',{},'endN',{},'sizeN',{});
for g=1:sim.nGroups
	groups(g).startN = fread(fid,1,'int32'); % start index at 0
	groups(g).endN = fread(fid,1,'int32');
    groups(g).sizeN = groups(g).endN-groups(g).startN+1;
	groups(g).name = char(fread(fid,100,'int8')');
	groups(g).name = groups(g).name(groups(g).name>0);
end

%% READ SYNAPSES
% reading synapse info is optional
if loadSynapseInfo
    weightData = cell(nrCells,1);
    nrSynTot = 0;
    for i=1:sim.nNeurons
        nrSyn = fread(fid,1,'int32');
        nrSynTot = nrSynTot + nrSyn;
        if nrSyn>0
            weightData{i} = fread(fid,[18 nrSyn],'uint8=>uint8');
        end
    end
    if ischar(filename)
        fclose(fid);
    end
    
    alldata = cat(2,weightData{:});
    clear weightData;
    preIDs = typecast(reshape(alldata(1:4,:),[],1),'uint32');
    postIDs = typecast(reshape(alldata(5:8,:),[],1),'uint32');
    weights = typecast(reshape(alldata(9:12,:),[],1),'single');
    maxWeights = typecast(reshape(alldata(13:16,:),[],1),'single');
    delays = alldata(17,:);
    plastic = alldata(18,:);
end

end