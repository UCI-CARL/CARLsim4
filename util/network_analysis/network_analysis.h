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
 * Ver 2/21/2014
 */

// paradigm shift: run this on spikes. 

#ifndef _ANALYSIS_H_
#define _ANALYSIS_H_

// we need the AER data structure
#include <carlsim_datastructures.h>
#include <algorithm>
#include <vector>

class network_analysis {
 public: 
	/*! 
	 * \brief analysis constructor.
	 *
	 * Creates a new instance of the analysis class. 
	 *
	 * Takes a carlsim object as an argument.
	 */
	network_analysis(const std::vector<AER>& spkVector); // Maybe setSpikeCounter for all groups in sim. // by calling an initAnalysis
	/*! 
	 * \brief network_analysis destructor.
	 *
	 * Cleans up all the memory upon object deletion.
	 *
	 */
	~network_analysis();
	
	// +++++ PUBLIC METHODS: +++++++++++++++++++++++++++++++++++++++++++++++//
	/*!
	 * \brief return the average firing rate for a certain group averaged
	 * over timeDuration.
	 * \param timeDuration is the time duration over which the spikes should
	 * be averaged.
	 * \return float value for the average firing rate of the whole group. 
	 */
	float getGrpFiringRate(int timeDuration, int sizeN);
	
	/*!
	 * \brief print the time/nid values to stdout. CAUTION: this can be
	 * a lot of data.
	 * \param void
	 * \return void
	 */
	void printAER();

 private:
	// will be turned into a data structure
	std::vector<AER>::const_iterator it_begin_;
	std::vector<AER>::const_iterator it_end_;
	int vectorSize_;
};


#endif
