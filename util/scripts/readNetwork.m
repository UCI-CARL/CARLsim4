function [groups, preIDs, postIDs, weights, delays, plastic, maxWeights] = readNetwork(filename)

if ischar(filename)
    nid = fopen(filename,'r');
else
    nid = filename;
end
version = fread(nid,1,'uint32');
if version>1
    error(['Unknown version number ' num2str(version)]);
end

nrGroups = fread(nid,1,'int32');
groups = struct('name',{},'startN',{},'endN',{});
for g=1:nrGroups
    groups(g).startN = fread(nid,1,'int32'); // start index at 0
    groups(g).endN = fread(nid,1,'int32');
    groups(g).name = char(fread(nid,100,'int8')');
    groups(g).name = groups(g).name(groups(g).name>0);
end

nrCells = fread(nid,1,'int32');
weightData = cell(nrCells,1);
nrSynTot = 0;
for i=1:nrCells
    nrSyn = fread(nid,1,'int32');
    nrSynTot = nrSynTot + nrSyn;
    if nrSyn>0
        weightData{i} = fread(nid,[18 nrSyn],'uint8=>uint8');
    end
end
if ischar(filename)
    fclose(nid);
end

alldata = cat(2,weightData{:});
weightData = {};

% fwrite(&i,sizeof(int),1,fid);
% fwrite(&p_i,sizeof(int),1,fid);
% fwrite(&(wt[pos_i]),sizeof(float),1,fid);
% fwrite(&(maxSynWt[pos_i]),sizeof(float),1,fid);
% fwrite(&delay,sizeof(uint8_t),1,fid);
% fwrite(&plastic,sizeof(uint8_t),1,fid);

preIDs = typecast(reshape(alldata(1:4,:),[],1),'uint32');
postIDs = typecast(reshape(alldata(5:8,:),[],1),'uint32');
weights = typecast(reshape(alldata(9:12,:),[],1),'single');
maxWeights = typecast(reshape(alldata(13:16,:),[],1),'single');
delays = alldata(17,:);
plastic = alldata(18,:);
