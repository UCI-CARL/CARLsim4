#include "carlsim_tests.h"

#include <stdio.h>			// fopen, fseek, fclose, etc.
#include <cassert>			// assert
#include <string.h>			// std::string


/// ****************************************************************************
/// Function to read and return a 1D array with time and nid (in that order.
/// ****************************************************************************
void readAndReturnSpikeFile(const std::string fileName, int*& AERArray, long &arraySize){
	FILE* pFile;
	long lSize;
	size_t result;
	pFile = fopen ( fileName.c_str() , "rb" );
	if (pFile==NULL) {fputs ("File error",stderr); exit (1);}
		
	// obtain file size:
	fseek (pFile , 0 , SEEK_END);
	lSize = ftell(pFile);
	arraySize = lSize/sizeof(int);
	rewind (pFile);
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
void readAndPrintSpikeFile(const std::string fileName){
	FILE * pFile;
	long lSize;
	int* buffer;
	size_t result;
	pFile = fopen ( fileName.c_str() , "rb" );
	if (pFile==NULL) {fputs ("File error",stderr); exit (1);}
			
	// obtain file size:
	fseek (pFile , 0 , SEEK_END);
	lSize = ftell (pFile);
	rewind (pFile);
		
	// allocate memory to contain the whole file:
	buffer = (int*) malloc (sizeof(int)*lSize);
	if (buffer == NULL) {fputs ("Memory error",stderr); exit (2);}
		
	// copy the file into the buffer:
	result = fread (buffer,1,lSize,pFile);
	if (result != lSize) {fputs ("Reading error",stderr); exit (3);}
		
	// the whole file is now loaded in the memory buffer.
	for(int i=0;i<lSize/sizeof(int);i=i+2){
		printf("time = %d, nid = %d\n",buffer[i],buffer[i+1]);
	}

	// terminate
	fclose (pFile);
	free (buffer);
}