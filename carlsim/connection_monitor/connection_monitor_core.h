/*
 * Copyright (c) 2014 Regents of the University of California. All rights reserved.
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
 * *************************************************************************
 * CARLsim
 * created by: 		(MDR) Micah Richert, (JN) Jayram M. Nageswaran
 * maintained by:	(MA) Mike Avery <averym@uci.edu>, (MB) Michael Beyeler <mbeyeler@uci.edu>,
 *					(KDC) Kristofor Carlson <kdcarlso@uci.edu>
 *					(TSC) Ting-Shuo Chou <tingshuc@uci.edu>
 *
 * CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
 * Ver 7/29/2014
 */

#ifndef _CONN_MON_CORE_H_
#define _CONN_MON_CORE_H_

#include <stdio.h>					// FILE
#include <vector>					// std::vector
#include <carlsim_definitions.h>	// ALL

class CpuSNN; // forward declaration of CpuSNN class


/*
 * \brief ConnectionMonitor private core implementation
 *
 * Naming convention for methods:
 * - getPop*: 		a population metric (single value) that applies to the entire group; e.g., getPopMeanFiringRate.
 * - getNeuron*: 	a neuron metric (single value), about a specific neuron (requires neurId); e.g., getNeuronNumSpikes. 
 * - getAll*: 		a metric (vector) that is based on all neurons in the group; e.g. getAllFiringRates.
 * - getNum*:		a number metric, returns an int
 * - getPercent*:	a percentage metric, returns a double
 * - get*:			all the others
 */
class ConnectionMonitorCore {
public: 
	/*! 
	 * \brief constructor
	 *
	 * Creates a new instance of the analysis class. 
	 * Takes a CpuSNN pointer to an object as an input and assigns it to a CpuSNN* member variable for easy reference.
	 * \param[in] snn pointer to current CpuSNN object
	 * \param[in] grpId the ID of the group we're monitoring
	 */
	ConnectionMonitorCore(CpuSNN* snn, int monitorId, short int connId, int grpIdPre, int grpIdPost); 

	//! destructor, cleans up all the memory upon object deletion
	~ConnectionMonitorCore();
	
	// +++++ PUBLIC METHODS: +++++++++++++++++++++++++++++++++++++++++++++++//

	std::vector< std::vector<float> > calcWeightChanges();

	//! deletes data from the 2D spike vector
	void clear();

	int getFanIn(int neurPostId);

	int getFanOut(int neurPreId);

	int getNumWeightsChanged(double minAbsChanged=1e-5);

	double getPercentWeightsChanged(double minAbsChanged=1e-5);


	double getTotalAbsWeightChange();

	//! initialization method
	//! depends on several SNN data structures, so it has be to called at the end of setConnectionMonitor (or later)
	void init();

	void print();

	//! blah
	void printSparse(int neurPostId=ALL, int maxConn=100, int connPerLine=4);

	std::vector< std::vector<float> > takeSnapshot();

	void updateWeight(int preId, int postId, float wt);

	void updateTime(unsigned int simTimeMs);

	
	// +++++ PUBLIC SETTERS/GETTERS: +++++++++++++++++++++++++++++++++++++++//

	short int getConnectId() { return connId_; }
	int getMonitorId() { return monitorId_; }
	FILE* getConnFileId() { return connFileId_; }
	long int getTimeMsCurrentSnapshot() { return simTimeMs_; }
	long int getTimeMsLastSnapshot() { return (simTimeMs_ - simTimeSinceLastMs_); }
	long int getTimeMsSinceLastSnapshot() { return simTimeSinceLastMs_; }


	/*!
	 * \brief returns the monBufferFid
	 * \param void
	 * \return unsigned int
	 */
	void setConnectFileId(FILE* connFileId);
	
private:
	bool needToWriteSnapshot();

	//! writes the header section (file signature, version number) of a connect file
	void writeConnectFileHeader();

	//! writes each snapshot to connect file
	void writeConnectFileSnapshot();

	CpuSNN* snn_;	//!< private CARLsim implementation
	int monitorId_;	//!< current ConnectionMonitor ID
	short int connId_;	//!< current connection ID
	int grpIdPre_;
	int grpIdPost_;
	int nNeurPre_;
	int nNeurPost_;
	int nSynapses_;

	long int simTimeMs_;
	long int simTimeSinceLastMs_;
	long int simTimeMsLastWrite_;

	bool isPlastic_; //!< whether this connection has plastic synapses

	std::vector< std::vector<float> > wtMat_, wtLastMat_;

	//! whether we have to write header section of conn file
	bool needToWriteFileHeader_;

	FILE* connFileId_;		//!< file pointer to the conn file or NULL
	int connFileSignature_; //!< int signature of conn file
	double connFileVersion_; //!< version number of conn file

	// file pointers for error logging
	const FILE* fpInf_;
	const FILE* fpErr_;
	const FILE* fpDeb_;
	const FILE* fpLog_;
};

#endif