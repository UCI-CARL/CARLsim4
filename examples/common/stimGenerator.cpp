//Kris Carlson
// Creates a Gabor filter for orientation and spatial frequency
// selectivity of orientation OR (in radians) and spatial frequency SF.
// everything is in radians
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define PI 3.1415926535897


//gabor generator for double array by reference
void gabor(double** &_gfilter, int _xdim, int _ydim, double _OR, double _SF)
{
  FILE *fp_gabor;
  fp_gabor = fopen("gfilter.txt","w");
  
  // set parameters
  double sigma_x=7/2;// standard deviation of 2D Gaussian along x-dir
  double sigma_y=17/2;// standard deviation of 2D Gaussian along y-dir
  
  // create filter
  // [x,y]=meshgrid(-8:1:7,-8:1:7);
  double X, Y;
  double x[_xdim];
  double y[_ydim];
  _gfilter=new double*[_xdim];
  for(int k=0;k<_ydim;k++)
    {
      _gfilter[k]=new double[_ydim];
    }


  for(int i = 0;i<_xdim;i++)
    {
      x[i]=-1*(_xdim/2)+i;
      //DEBUGGING
      //printf("x[i]=%f \n",x[i]);
    }
  //printf("\n");
  
  for(int i = 0;i<_ydim;i++)
    {
      y[i]=-1*(_ydim/2)+i;
      //DEBUGGING
      //printf("y[i]=%f \n",y[i]);
    }
  
  for(int i=0;i<_xdim;i++)
    {
      for(int j=0;j<_ydim;j++)
	{
	  X=(x[i])*cos(_OR)+y[j]*sin(_OR); //rotate axes
	  Y=-(x[i])*sin(_OR)+y[j]*cos(_OR);
	  
	  _gfilter[i][j]=(1/(2*PI*sigma_x*sigma_y))*exp( -0.5*(pow(X/sigma_x,2)+pow(Y/sigma_y,2) ))*sin(2*PI*_SF*X);
	  //DEBUGGING
	  //printf("_gfilter[%d][%d]=%e\n",i,j,_gfilter[i][j]);
	  //fprintf(fp_gabor,"%d %d %e\n",i,j,_gfilter[i][j]);
	}
      //printf("\n");
    }
  
  fclose(fp_gabor);

  return;

}

//gabor generator for single array by reference
void gabor(double _gfilter[], int _size, double _OR, double _SF)
{
  int xdim=sqrt(_size);
  int ydim=sqrt(_size);

  //DEBUGGING
  //FILE *fp_gabor;
  //fp_gabor = fopen("gfilter.txt","a");
  
  // set parameters
  double sigma_x=7/2;// standard deviation of 2D Gaussian along x-dir
  double sigma_y=17/2;// standard deviation of 2D Gaussian along y-dir
  
  // create filter
  // [x,y]=meshgrid(-8:1:7,-8:1:7);
  double X, Y;
  double x[xdim];
  double y[ydim];

 
  for(int i = 0;i<xdim;i++)
    {
      x[i]=-1*(xdim/2)+i;
      //DEBUGGING
      //printf("x[i]=%f \n",x[i]);
    }
  //printf("\n");
  
  for(int i = 0;i<ydim;i++)
    {
      y[i]=-1*(ydim/2)+i;
      //DEBUGGING
      //printf("y[i]=%f \n",y[i]);
    }
  
  
  
  for(int i=0;i<xdim;i++)
    {
      for(int j=0;j<ydim;j++)
	{
	  X=(x[i])*cos(_OR)+y[j]*sin(_OR); //rotate axes
	  Y=-(x[i])*sin(_OR)+y[j]*cos(_OR);
	  
	  _gfilter[i*ydim+j]=(1/(2*PI*sigma_x*sigma_y))*exp( -0.5*(pow(X/sigma_x,2)+pow(Y/sigma_y,2) ))*sin(2*PI*_SF*X);
	  //DEBUGGING
	  //printf("_gfilter[%d][%d]=%e\n",i,j,_gfilter[i*ydim+j]);
	  //fprintf(fp_gabor,"%d %d %e\n",i,j,_gfilter[i*ydim+j]);
	}
      //printf("\n");
    }

  //DEBUGGING
  //fclose(fp_gabor);

  return;

}



void outputFilter(double _filter[], int _size)
{
  FILE *fp;
  
  fp = fopen("filter_data.dat","ab");
  
  fwrite(_filter,sizeof(double),_size,fp);

  fflush(fp);
  fclose(fp);

  return;
}

void outputAngle(int i)
{
  FILE *fp;
  
  fp = fopen("angle_data.dat","ab");
  double angle[]={1*PI/4,2*PI/4,3*PI/4,PI,5*PI/4,6*PI/4,7*PI/4,2*PI};

  fwrite(&i,sizeof(int),1,fp);

  fflush(fp);
  fclose(fp);

  return;
}


// Counter phase sinusoidal grating generator for single array by reference
// _OR=orientation, _SF=spatial frequency, _SP=spatial phase, _A=contrast amplitude
// Because Dayan and Abbott measured _OR from the y-axis (_OR=0 at y-axis), 
// we add PI/2 to the orientation.  This makes it so that _OR=0 corresponds to vertical
// orientation.
void CPSGrating(double _gratingFilter[], int _size, double _OR, double _SF, double _SP, double A)
{
  int xdim=sqrt(_size);
  int ydim=sqrt(_size);

  //FILE *fp_grating;
  //fp_grating = fopen("grating.txt","a");
  
  //Perhaps _SF~5-15 to fit?
  //or 0.01 as before?
 
  double arg1;
  double x[xdim];
  double y[ydim];

  // create filter
  // [x,y]=meshgrid(-8:1:7,-8:1:7);
  for(int i = 0;i<xdim;i++)
    {
      x[i]=-1*(xdim/2)+i;
      //x[i]=i+xdim/2-xdim/2+1;//0 to 15
      //DEBUGGING
      //printf("x[i]=%f \n",x[i]);
    }
  //printf("\n");
  
  for(int i = 0;i<ydim;i++)
    {
      y[i]=-1*(ydim/2)+i;
      //y[i]=i+ydim/2-ydim/2+1;//0 to 15
      //DEBUGGING
      //printf("y[i]=%f \n",y[i]);
    }
    
  for(int i=0;i<xdim;i++)
    {
      for(int j=0;j<ydim;j++)
	{
	  arg1=_SF*(x[i])*cos(_OR)+_SF*(y[j])*sin(_OR)-_SP;
	  
	   _gratingFilter[i*ydim+j]=A*cos(arg1);
	  //_gratingFilter[i*ydim+j]=3*x[i]+y[j];
	  //DEBUGGING
	  //printf("_gratingFilter[%d][%d]=%e\n",i,j,_gratingFilter[i*ydim+j]);
	  //fprintf(fp_grating,"%d %d %e\n",i,j,_gratingFilter[i*ydim+j]);
	}
      //printf("\n");
    }

  
  //fclose(fp_grating);

  return;

}


void getRate(double _filter[], double _rateMatrix[], int _dim, double _maxRate, double _minRate, int _OnOrOff)
{
  int i,j;
  float maxVal, minVal, maxMinVal;

  //OnOrOff, if =1, then on, else then off
  if(_OnOrOff == 1) 
    {
      //DEBGUGGING
      //FILE *fp = fopen("CPSGratingRateOn.txt","w");
      maxVal = minVal = _filter[0];
      for(i = 0; i < _dim; i++ )
	{
	  for(j=0; j< _dim; j++)
	    {
	      if( _filter[i*_dim+j] >=  0.0 ) 
		{
		  if( _filter[i*_dim+j] > maxVal ) 
		    maxVal = _filter[i*_dim+j];
		  if( _filter[i*_dim+j] < minVal ) 
		    minVal = _filter[i*_dim+j];
		}
	    }
	}
      
      maxMinVal = maxVal - minVal;
    
      for(i=0;i<_dim;i++ ) 
	{
	  for(j=0;j<_dim;j++ )	
	    {
	      if( _filter[i*_dim+j] >=  0.0 ) 
		{
		  //y=m(x-x0)+b (y=rateMatrix, x=gaborPattern)
		  _rateMatrix[i*_dim+j] = _minRate + (_maxRate-_minRate)*(_filter[i*_dim+j]-minVal)/maxMinVal;
		}
	      else
		_rateMatrix[i*_dim+j] = _minRate;
	      //DEBUGGING
	      //fprintf(fp, "[%d,%d] %f ==> %f\n", i, j, _filter[i][j], *ptr);
	    }
	}
    }
  else
    {
      //DEBUGGING
      //FILE *fp = fopen("gaborRateOff.txt","w");
    maxVal = minVal = _filter[0];
    for(i = 0; i < _dim; i++ )
      {
	for(j=0; j< _dim; j++)
	{
	  if( _filter[i*_dim+j] <=  0.0 ) 
	    {
	      if( _filter[i*_dim+j] > maxVal ) 
		maxVal = _filter[i*_dim+j];
	      if( _filter[i*_dim+j] < minVal ) 
		minVal = _filter[i*_dim+j];
	    }
	}
      }
    maxMinVal = maxVal - minVal;
    
    for(i=0;i<_dim;i++ ) 
      {
	for(j=0;j<_dim;j++ )	
	  {
	    if( _filter[i*_dim+j] <=  0.0 )
	      {
		//y=m(x-x0)+b (y=rateMatrix, x=gaborPattern)
		_rateMatrix[i*_dim+j] = _maxRate - (_maxRate-_minRate)*(_filter[i*_dim+j]-minVal)/maxMinVal;
	      }
	    else
	      _rateMatrix[i*_dim+j] = _minRate;
	    //DEBUGGING
	    //fprintf(fp, "[%d,%d] %f ==> %f\n", i, j, _filter[i][j], *ptr);
	  }
      }
    }
  
  return;
}
