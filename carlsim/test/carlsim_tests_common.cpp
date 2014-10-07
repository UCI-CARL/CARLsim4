#include "carlsim_tests.h"

#include <stdio.h>			// fopen, fseek, fclose, etc.
#include <cassert>			// assert
#include <string.h>			// std::string


/// ****************************************************************************
/// Function to read and return a 1D array with time and nid (in that order.
/// ****************************************************************************
// \TODO this should probably be a utility function or something more standard...
// Reason is that whenever we change the structure of the spike file (for example, by
// adding more meta-data to the header section of the binary file), then this function
// here needs to be adjusted...
// Same goes for readAndPrintSpikeFile.
void readAndReturnSpikeFile(const std::string fileName, int*& AERArray, long &arraySize){
	FILE* pFile;
	long lSize;
	size_t result;
	pFile = fopen ( fileName.c_str() , "rb" );
	if (pFile==NULL) {fputs ("File error",stderr); exit (1);}

	// \FIXME: this is a hack...to get the size of the header section
	// needs to be updated every time header changes
	int szByteHeader = 4*sizeof(int)+1*sizeof(float);

	// get data size
	fseek(pFile, 0, SEEK_END);
	lSize = ftell(pFile) - szByteHeader;
	arraySize = lSize/sizeof(int);

	// jump back to end of header
	fseek(pFile, szByteHeader, SEEK_SET);
		
//	fprintf(stderr,"lSize = %d, arraySize = %d\n",lSize,arraySize);

	AERArray = new int[lSize];
	memset(AERArray,0,sizeof(int)*lSize);
	// allocate memory to contain the whole file:
	int* buffer = (int*) malloc (sizeof(int)*lSize);
	if (buffer == NULL) {fputs ("Memory error",stderr); exit (2);}
		
	// copy the file into the buffer:
	result = fread (buffer,1,lSize,pFile);
	if (result != lSize) {fputs ("Reading error",stderr); exit (3);}
		
	// the whole file is now loaded in the memory buffer.
	for (int i=0; i<lSize; i++) {
		int tmp = buffer[i];
		AERArray[i]=tmp;
	}

	// terminate
	fclose (pFile);
	free(buffer);
}

/// ****************************************************************************
/// Function for reading and printing spike data written to a file
/// ****************************************************************************
// \FIXME: same as above
void readAndPrintSpikeFile(const std::string fileName){
	int* arrayAER;
	long arraySize;
	readAndReturnSpikeFile(fileName, arrayAER, arraySize);

	for (int i=0; i<arraySize; i+=2)
		printf("time = %d, nid = %d\n",arrayAER[i],arrayAER[i+1]);
}