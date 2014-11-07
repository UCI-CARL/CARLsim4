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

// paradigm shift: run this on spikes.

#ifndef _CONN_MON_H_
#define _CONN_MON_H_

#include <carlsim_datastructures.h> // spikeMonMode_t
#include <vector>					// std::vector

//class CpuSNN; 			// forward declaration of CpuSNN class
class ConnectionMonitorCore; // forward declaration of implementation

/*!
 * \brief Class ConnectionMonitor
 *
 *
 * \TODO finish documentation
 */
class ConnectionMonitor {
 public:
	/*!
	 * \brief ConnectionMonitor constructor
	 *
	 * Creates a new instance of the ConnectionMonitor class.
	 *
	 */
	ConnectionMonitor(ConnectionMonitorCore* connMonCorePtr);

	/*!
	 * \brief ConnectionMonitor destructor.
	 *
	 * Cleans up all the memory upon object deletion.
	 *
	 */
	~ConnectionMonitor();


	// +++++ PUBLIC METHODS: +++++++++++++++++++++++++++++++++++++++++++++++//


 private:
  //! This is a pointer to the actual implementation of the class. The user should never directly instantiate it.
  ConnectionMonitorCore* connMonCorePtr_;

};

#endif
