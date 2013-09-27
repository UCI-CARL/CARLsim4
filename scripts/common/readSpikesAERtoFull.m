function s = readSpikesAERtoFull( time,nIDs,frameDur,dim )
%READSPIKESAERTOFULL expects spike data in AER format as input, i.e. spiketimes
%(TIME) and neuron IDs (NIDS), and converts to a FULL representation.
%Indexing in the input should start at 0, and will be converted to start at
%1.
%   time:       all spike times
%   nIDs:       all neuron IDs
%   frameDur:   frame duration, ms per stim period (default 1000)
%   dim:        [max limit of TIME, max limit of NIDS]. If omitted, extract
%               from input params
%
% Created by: Michael Beyeler <mbeyeler@uci.edu>
% Ver 05/31/12

if nargin<4
    % dimensions not given, extract from input params
    dim = [max(floor(time/frameDur)+1) max(nIDs)+1];
end
if nargin<3,frameDur=1000;end
if nargin<2,error('spike times and neuron IDs needed');end

% grow S to right dimensions
s = zeros(dim);

% make sure TIME and NIDS have size Nx1 (important for ACCUMARRAY)
time = reshape(time,[],1);
nIDs = reshape(nIDs,[],1);

% ACCUMARRAY is supposed to be faster than full(sparse(...)). Make
% sure the first two arguments are column vectors.
% Convert time and nID to start indexing at 1
s=accumarray([floor(time/frameDur)+1,nIDs+1],1,dim);

end

