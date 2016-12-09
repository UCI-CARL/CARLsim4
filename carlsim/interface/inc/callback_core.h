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

#ifndef _CALLBACK_CORE_H_
#define _CALLBACK_CORE_H_

class CARLsim;
class SNN;

class ConnectionGenerator;
class SpikeGenerator;

/// **************************************************************************************************************** ///
/// Classes for relay callback
/// **************************************************************************************************************** ///

//! used for relaying callback to SpikeGenerator
/*!
 * \brief The class is used to store user-defined callback function and to be registered in core (i.e., snn_cpu.cpp)
 * Once the core invokes the callback method of the class, the class relays all parameter and invokes user-defined
 * callback function.
 * \sa SpikeGenerator
 */
class SpikeGeneratorCore {
public:
	SpikeGeneratorCore(CARLsim* c, SpikeGenerator* s);
	//! controls spike generation using a callback mechanism
	/*! \attention The virtual method should never be called directly
	 */
	virtual int nextSpikeTime(SNN* s, int grpId, int i, int currentTime, int lastScheduledSpikeTime, int endOfTimeSlice);

private:
	CARLsim* carlsim;
	SpikeGenerator* sGen;
};

//! used for relaying callback to ConnectionGenerator
/*!
 * \brief The class is used to store user-defined callback function and to be registered in core (i.e., snn_cpu.cpp)
 * Once the core invokes the callback method of the class, the class relays all parameter and invokes user-defined
 * callback function.
 * \sa ConnectionGenerator
 */
class ConnectionGeneratorCore {
public:
	ConnectionGeneratorCore(CARLsim* c, ConnectionGenerator* cg);
	//! specifies which synaptic connections (per group, per neuron, per synapse) should be made
	/*! \attention The virtual method should never be called directly */
	virtual void connect(SNN* s, int srcGrpId, int i, int destGrpId, int j, float& weight, float& maxWt,
		float& delay, bool& connected);

private:
	CARLsim* carlsim;
	ConnectionGenerator* cGen;
};

#endif
