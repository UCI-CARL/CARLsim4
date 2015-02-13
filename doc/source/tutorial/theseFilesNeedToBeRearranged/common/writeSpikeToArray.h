// Interface and implementation for WriteSpikeToArray class/callback function that inherits
// from SpikeMonitor.

#include "snn.h"

class WriteSpikeToArray: public SpikeMonitor {
private: 
  bool firstRun;
  int matrixSize;
  int oldMatrixSize;
  int newMatrixSize;
  unsigned int** oldArray;
  unsigned int** tempArray;
  unsigned int** array;
public:
  WriteSpikeToArray() {
    matrixSize=0;
    oldMatrixSize=0;
    newMatrixSize=0;
    firstRun=true;
    oldArray=NULL;
  }
  
  // takes a reference to a int** (2D array) and a reference to an int.  Returns
  // the size of dimension 1 of _answerArray in variable _size.  Returns _answerArray
  // which is a 2D array.  The first column has the time the neuron spiked, the
  // second column returns the corresponding NID of the neuron that spiked at that
  // time.  DIM 1 is of size _size and DIM2 is of size 2.
  void getArrayInfo(unsigned int**& _answerArray, int& _size){
    assert(array!=NULL);
    _answerArray=array;
    _size=matrixSize;
  }
  
  void resetSpikeCounter(){
    firstRun=true;
  }

  // callback function
  void update(CpuSNN* s, int grpId, unsigned int* Nids, unsigned int* timeCnts)
  {
    // first calculate the size of the current 1 second of data
    matrixSize = 0;
    int pos = 0;
    for(int t=0; t <1000; t++){
      for(int i=0; i<timeCnts[t];i++,pos++) {
	matrixSize=matrixSize+1;
      }
    }

    //create the correct size array
    tempArray = new unsigned int* [matrixSize];
    for(int i=0;i<matrixSize;i++){
      tempArray[i]=new unsigned int[2];
    }

    pos = 0;
    // get the data
    for (int t=0; t < 1000; t++) {
      for(int i=0; i<timeCnts[t];i++,pos++) {
    	int time = t + s->getSimTime() - 1000;
    	int id   = Nids[pos];
	tempArray[pos][0]=time;
    	tempArray[pos][1]=id;
      }
    }

    // adjust array pointers in case the function is called again
    if(firstRun){
      oldMatrixSize=matrixSize;
      array=tempArray;
      oldArray=array;
      firstRun=false;
    } // copy data from previous runs into a single array
    else{
      newMatrixSize=matrixSize+oldMatrixSize;
      array = new unsigned int*[newMatrixSize];
      for(int i=0;i<newMatrixSize;i++){
    	array[i]=new unsigned int[2];
      } 
      // copy the data from the old matrix first
      for(int i=0;i<oldMatrixSize;i++){
    	array[i][0]=oldArray[i][0];
    	array[i][1]=oldArray[i][1];
      }
      // copy the data from the new matrix next
      for(int i=0;i<matrixSize;i++){
    	array[oldMatrixSize+i][0]=tempArray[i][0];
	array[oldMatrixSize+i][1]=tempArray[i][1];
      }
      // now fix the pointers and free the memory
      // first delete the oldArray
      for(int i=0;i<oldMatrixSize;i++){
	delete[] oldArray[i];
      }
      delete[] oldArray;
      // now point the oldArray to the current array
      oldArray=array;
      // free the memory for temp array
      for(int i=0;i<matrixSize;i++){
	delete[] tempArray[i];
      }
      delete[] tempArray;
      tempArray=NULL;
      // set the correct size to return
      matrixSize=newMatrixSize;
      oldMatrixSize=matrixSize;
    }
  }
};
