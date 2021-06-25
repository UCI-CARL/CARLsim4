function [connMat,preconnMat,postconnMat,nTypeCounts,synCountMat] = computeSynapseCount(xlName)

% Load in the sheets corresponding to the connection probabilities and the
% population sizes of each of the connection and neuron types, respectively
[~,~,connMat] = xlsread(xlName,'internal_to_internal_conn_prob');

[~,~,nTypeCounts] = xlsread(xlName,'pop_size_neuron_type_internal');

% Pre-process the matrices corresponding to the connection probabilities
% and the neuron type population sizes
connMat = connMat(1:9,1:9);
preconnMat = connMat(2:end,1);
postconnMat = connMat(1,2:end);
postconnMat = postconnMat';

nTypeCounts = nTypeCounts(1:9,1:2);
nTypeCounts = nTypeCounts(2:end,:);

for i = 1:length(preconnMat)
    preconnMat{i} = regexprep(preconnMat{i}, '-','_');
    preconnMat{i} = regexprep(preconnMat{i}, ' ','_');
    preconnMat{i} = regexprep(preconnMat{i}, '+','');
    postconnMat{i} = regexprep(postconnMat{i}, '-','_');
    postconnMat{i} = regexprep(postconnMat{i}, ' ','_');
    postconnMat{i} = regexprep(postconnMat{i}, '+','');
end

% Append the population sizes for each neuron type to the pre and post
% synaptic neuron types to be used for computing the number of synapses for
% each type
preconnMat = [preconnMat nTypeCounts(:,2)];
postconnMat = [postconnMat nTypeCounts(:,2)];

[~,postOrder] = ismember(postconnMat(:,1),preconnMat(:,1));
postconnMat = preconnMat(postOrder,:);

% [~,postOrder] = ismember(preconnMat(:,1),postconnMat(:,1));
% postconnMat = 

synCountMat = connMat;


for i = 2:size(synCountMat,1)
    for j = 2:size(synCountMat,2)
        if synCountMat{i,j} ~= 0 
            synCountMat{i,j} = floor(preconnMat{i-1,2}*postconnMat{j-1,2}*synCountMat{i,j});
        end
    end
end


writecell(synCountMat,xlName,'Sheet','internal_to_internal_conn_prob','Range','A22:I31');