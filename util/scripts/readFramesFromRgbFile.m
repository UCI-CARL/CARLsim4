function I = readFramesFromRgbFile( filename,dim,frames,plotIt,movieIt )
%I = readFramesFromRgbFile(filename,dim,frames,plotIt)
%READFRAMESFROMRGBFILE returns all frames found in a file of RGB images. The
%file data should be organized as img1R1 img1G1 img1B1 img1R2 img1G2 img1B2
%... img2R1 img2G1 img2B1 etc. in uchar format e[0,255]. Such a file can
%be created with WRITEFRAMESTOFILE. The function returns an image I with
%values e[0,1] (required by IMAGESC etc.).
%   filename:   filepath to RGB images, organized as R1 G1 B1 R2 G2 B2...,
%               frame after frame.
%   dim:        [nrRows nrCols]
%   frames:     vector of all frames to read, or set to "-1" to read out
%               all frames. Default: -1.
%   plotIt:     set to true to plot each frame. Default: false.
%   movieIt:    set to true to output frames to './movie.avi'.
%               Default: false.
%
% Author: Michael Beyeler <mbeyeler@uci.edu>
% Ver 07/23/12

if nargin<5,movieIt=false;end
if nargin<4,plotIt=false;end
if nargin<3,frames=-1;end % -1 means: read all frames
if nargin<2,error('filename or dim not set');end

if length(dim)~=2 || min(dim)<0
    error('invalid image dimension, must be [nrRows nrCols]');
end

fid = fopen(filename,'r');
if fid==-1
    error(['could not open file "' filename '"']);
end

if frames==-1
    frames=1:1e6;
end

if plotIt
    fig=figure(1);
    set(fig,'DoubleBuffer','on');
    set(fig,'Position',[1 100 800 600]);
    set(fig,'PaperPositionMode','auto');
end
if movieIt
    outputVideo='movie.avi';
    Mov = avifile(outputVideo,'fps',10,'quality',100,'compression','none');
end

I={};
lastRow=0;
for row=frames
    % if we're skipping frames we have to adjust fp
    if row-lastRow>1
        %disp(['row is ' num2str(row) ', lastRow is ' num2str(lastRow) '...
        %     , skipping ' num2str(row-lastRow-1) ' rows']);
        jumpAhead = fseek(fid,(row-lastRow-1)*prod(dim)*3,'cof');
    else
        jumpAhead = 0; % no error
    end
    
    % read another frame and update last row
    w=fread(fid,prod(dim)*3,'uchar');
    lastRow=row;
    
    % check for eof through fseek or fread
    if (jumpAhead==-1) || (length(w)~=prod(dim)*3)
        break;
    end
    
    % w is supposed to be organized as R1 G1 B1 R2 G2 B2 ...
    % 1. reshape into [R1 R2 R3 ..;G1 G2 G3 ..;B1 B2 B3 ..]
    % 2. transpose (now all R, G, and B are in a separate column)
    % 3. reshape each column to the pic dimension in DIM
    % 4. normalize values so that imshow() works
    I{end+1} = reshape(reshape(w,3,[])',dim(1),dim(2),[])./255;
    
    if plotIt
        imagesc(I{end});
        origSize = size(I{end}(1,:,:));
        text(2,dim(1)-3,num2str(find(frames==row)),'FontSize',10,'BackgroundColor','white');
        getframe(fig);
    end
    
    if movieIt
        F=getframe(gcf);
        Mov = addframe(Mov,F);
    end
end

fclose(fid);
if movieIt
    Mov = close(Mov);
end

end