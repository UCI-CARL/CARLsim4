function createStimFromImage(inFile, outFile, outPxSize, mode)
% createStimFromImage(inFile, outFile, mode) creates a stimulus file from
% an image file using the VisualStimulusToolbox.
%
% INFILE     - An image file, such as .jpg or .png
% OUTFILE    - Name of the output file to create (of type .dat). Can be
%              read by CARLsim using the VisualStimulus bindings.
%              Default: '../input/image.dat'.
% OUTPXSIZE  - Size of output image in pixels: [width x height].
%              Default: [128 128].
% MODE       - Either 'gray' for grayscale images or 'rgb' for RGB images.
%              Default: 'gray'
if nargin<2,outFile='../input/image.dat';end
if nargin<3,outPxSize=[128 128];end
if nargin<4,mode='gray';end

pic = PictureStim(inFile);
pic.resize(outPxSize);

switch lower(mode)
	case 'gray'
		pic.rgb2gray();
	case 'rgb'
		pic.gray2rgb();
end

pic.save(outFile);

end