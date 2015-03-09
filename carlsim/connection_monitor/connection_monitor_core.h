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
 * Ver 2/10/2015
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
 * - getNum*:		a number metric, returns an int
 * - getTime*:      a time metric, returns long int
 * - getPercent*:	a percentage metric, returns a double
 * - get*:			all the others
 */
class ConnectionMonitorCore {
public: 
	//! constructor, created by CARLsim::setConnectionMonitor
	ConnectionMonitorCore(CpuSNN* snn, int monitorId, short int connId, int grpIdPre, int grpIdPost); 

	//! destructor, cleans up all the memory upon object deletion
	~ConnectionMonitorCore();
	

	// +++++ PUBLIC METHODS: +++++++++++++++++++++++++++++++++++++++++++++++//

	//! calculates weight changes since last snapshot and reports them in 2D weight change matrix
	std::vector< std::vector<float> > calcWeightChanges();

	//! returns connection ID
	short int getConnectId() { return connId_; }

	//! returns pointer to connection file
	FILE* getConnectFileId() { return connFileId_; }

	//! returns number of incoming synapses to post-synaptic neuron
	int getFanIn(int neurPostId);

	//! returns number of outgoing synapses of pre-synaptic neuron
	int getFanOut(int neurPreId);

	//! returns ConnectionMonitor ID
	int getMonitorId() { return monitorId_; }

	//! returns number of neurons in pre-synaptic group
	int getNumNeuronsPre() { return nNeurPre_; }

	//! returns number of neurons in post-synaptic group
	int getNumNeuronsPost() { return nNeurPost_; }

	//! returns number of synapses that exist in the connection
	int getNumSynapses() { return nSynapses_; }

	//! returns number of weights with >=minAbsChanged weight change since last snapshot
	int getNumWeightsChanged(double minAbsChanged=1e-5);

	//! returns percentage of weights with >=minAbsChanged weight change since last snapshot
	double getPercentWeightsChanged(double minAbsChanged=1e-5);

	//! returns the timestamp of the current snapshot (not necessarily CARLsim::getSimTime)
	long int getTimeMsCurrentSnapshot() { return simTimeMs_; }

	//! returns the timestamp of the last snapshot
	long int getTimeMsLastSnapshot() { return (simTimeMs_ - simTimeSinceLastMs_); }

	//! returns the time passed between current and last snapshot
	long int getTimeMsSinceLastSnapshot() { return simTimeSinceLastMs_; }

	//! returns absolute sum of all weight changes since last snapshot
	double getTotalAbsWeightChange();

	//! prints current weight state as 2D matrix (non-existent synapses: NAN, existent but zero weigth: 0.0f)
	void print();

	//! prints current weight state as sparse list of (only allocated, existent) synapses
	void printSparse(int neurPostId=ALL, int maxConn=100, int connPerLine=4);

	//! takes snapshot of current weight state and returns 2D matrix (non-existent synapses: NAN, existent but zero
	//! weight: 0.0f).
	std::vector< std::vector<float> > takeSnapshot();


	// +++++ PUBLIC METHODS THAT SHOULD NOT BE EXPOSED TO INTERFACE +++++++++//

	//! deletes data from the 2D weight matrix
	void clear();

	//! initialization method
	//! depends on several SNN data structures, so it has be to called at the end of setConnectionMonitor (or later)
	void init();

	//! updates an entry in the current weight matrix (called by CARLsim::updateConnectionMonitor)
	void updateWeight(int preId, int postId, float wt);

	//! updates timestamp of the snapshots, returns true if update was needed
	bool updateTime(unsigned int simTimeMs);

	//! sets pointer to connection file
	void setConnectFileId(FILE* connFileId);

	//! sets time update interval (seconds) for periodically storing weights to file
	void setUpdateTimeIntervalSec(int intervalSec);
	
private:
	//! indicates whether writing the current snapshot is necessary (false it has already been written)
	bool needToWriteSnapshot();

	//! writes the header section (file signature, version number) of a connect file
	void writeConnectFileHeader();

	//! writes each snapshot to connect file
	void writeConnectFileSnapshot();

	CpuSNN* snn_;                   //!< private CARLsim implementation
	int monitorId_;                 //!< current ConnectionMonitor ID
	short int connId_;              //!< current connection ID
	int grpIdPre_;                  //!< pre-synaptic group ID
	int grpIdPost_;                 //!< post-synaptic group ID
	int nNeurPre_;                  //!< number of neurons in pre
	int nNeurPost_;                 //!< number of neurons in post
	int nSynapses_;                 //!< number of synapses in connection

	float minWt_;					//!< minimum weight magnitude of the connection
	float maxWt_;					//!< maximum weight magnitude of the connection

	long int simTimeMs_;            //!< timestamp of current snapshot
	long int simTimeSinceLastMs_;   //!< time between current and last snapshot
	long int simTimeMsLastWrite_;   //!< timestamp of last file write

	bool isPlastic_; //!< whether this connection has plastic synapses

	std::vector< std::vector<float> > wtMat_;      //!< current snapshot of weight matrix
	std::vector< std::vector<float> > wtLastMat_;  //!< last snapshot of weight matrix

	bool needToInit_;				//!< whether we have to initialize first
	bool needToWriteFileHeader_;    //!< whether we have to write header section of conn file

	bool tookSnapshot_;

	FILE* connFileId_;              //!< file pointer to the conn file or NULL
	int connFileSignature_;         //!< int signature of conn file
	float connFileVersion_;         //!< version number of conn file
	int connFileTimeIntervalSec_;   //!< time update interval (seconds) for storing weights to file

	const FILE* fpInf_;             //!< file pointer for info logging
	const FILE* fpErr_;             //!< file pointer for error logging
	const FILE* fpDeb_;             //!< file pointer for debug logging
	const FILE* fpLog_;             //!< file pointer of debug.log
};

#endif
