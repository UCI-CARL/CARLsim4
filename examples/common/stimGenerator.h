//Kris Carlson
// Creates a Gabor filter for orientation and spatial frequency
// selectivity of orientation OR (in radians) and spatial frequency SF.
// everything is in radians

void gabor(double** &_gfilter, int _xdim, int _ydim, double _OR, double _SF);

void gabor(double _gfilter[], int size, double _OR, double _SF);

//_dim refers to square dimension.  If array is of size 10x10, _dim is 10.  Of course
//this is only valid for square matrices.
void getRate(double _gaborPattern[], double _rateMatrix[], int _dim, double _maxRate, double _minRate, int _OnOrOff);

//just meant to output the filters to file 
void outputFilter(double _filter[], int _size);

//outputs the angles chosen randomly that are presented to the network
//during the training and testing phase.
void outputAngle(int i);

//counter phase sinusoidal grating generator for single array by reference
//_OR=orientation, _SF=spatial frequency, _SP=spatial phase, _A=contrast amplitude
void CPSGrating(double _gratingFilter[], int _size, double _OR, double _SF, double _SP, double A);


