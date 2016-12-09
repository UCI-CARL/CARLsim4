/* * Copyright (c) 2016 Regents of the University of California. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*
* 1. Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*
* 3. The names of its contributors may not be used to endorse or promote
*    products derived from this software without specific prior written
*    permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
* A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
* LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
* NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* *********************************************************************************************** *
* CARLsim
* created by: (MDR) Micah Richert, (JN) Jayram M. Nageswaran
* maintained by:
* (MA) Mike Avery <averym@uci.edu>
* (MB) Michael Beyeler <mbeyeler@uci.edu>,
* (KDC) Kristofor Carlson <kdcarlso@uci.edu>
* (TSC) Ting-Shuo Chou <tingshuc@uci.edu>
* (HK) Hirak J Kashyap <kashyaph@uci.edu>
*
* CARLsim v1.0: JM, MDR
* CARLsim v2.0/v2.1/v2.2: JM, MDR, MA, MB, KDC
* CARLsim3: MB, KDC, TSC
* CARLsim4: TSC, HK
*
* CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
* Ver 12/31/2016
*/
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