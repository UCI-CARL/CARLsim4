function writeFramesToRgbFile(createFile,frames,appendIt)
% writeFramesToRgbFile(createFile,frames,appendIt)
%
%WRITEFRAMESTORGBFILE: writes all the frames given in FRAMES to a file.
%FRAMES should be a cell array of RGB images dim1xdim2x3, with elements in
%the range e[0,255]. FRAMES can be created using GETFRAMESINFOLDER or
% GETFRAMESFROMRGBFILE. If you set APPENDIT to false, all the contents in
% CREATEFILE will be discarded.
%   createFile: where to write the data. The data will be a string of uchar
%               in order R1 G1 B1 R2 G2 B2 ... e[0,255]
%   frames:     all image frames in standard RGB-format [dim1xdim2x3]
%   appendIt:   if true, new data will be appended to "createFile".
%               Default: false.
%
% Author: Michael Beyeler <mbeyeler@uci.edu>
% Ver 04/16/2013

if nargin<3,appendIt=false;end
if nargin<2,error('frames or folder not set');end

if appendIt
    fid = fopen(createFile,'a+');
else
    fid = fopen(createFile,'w');
end
if fid==-1
    error(['could not create file "' createFile '"']);
end

for i=1:length(frames)
    if mod(i,100)==0
        disp(['processing frame ' num2str(i)]);
    end
    
    I = frames{i};
    if max(I(:))<=1
        % NOTE: finding elements>1 is a good indicator that the range is
        % intended to be [0,255]. However, assuming that max<1 means the
        % range is intended to be [0,1] fails for e.g. black/dark-gray
        % scenes and such
        disp(['frame ' num2str(i) ': are you sure elements are [0,255]?']);
    end
    
    % An image frame is dim1xdim2x3: reshape to 3 rows (every row is one of
    % the R,G,B-channels). Note that FWRITE processes in column order
    % (which means iterating through all rows of column 1 first), so we
    % want our image to look like:
    % [R1 R2 R3 ... ; G1 G2 G3 ... ; B1 B2 B3]
    % because then it gets written like this: [R1 G1 B1 R2 G2 B2 ...]
    I = reshape(I,[],3)';
    cnt = fwrite(fid,I,'uchar');
    if numel(I)~=cnt
        error('writing to file failed!');
    end
end

% notify the user a file has been created
disp(['created file "' createFile '"']);
fclose(fid);
end