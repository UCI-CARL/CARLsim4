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
 * *********************************************************************************************** *
 * CARLsim
 * created by: 		(MDR) Micah Richert, (JN) Jayram M. Nageswaran
 * maintained by:	(MA) Mike Avery <averym@uci.edu>, (MB) Michael Beyeler <mbeyeler@uci.edu>,
 *					(KDC) Kristofor Carlson <kdcarlso@uci.edu>
 *					(TSC) Ting-Shuo Chou <tingshuc@uci.edu>
 *
 * CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
 * Ver 2/21/2014
 */

#ifndef _CALLBACK_H_
#define _CALLBACK_H_

//! CARLsim user interface classes
class CARLsim; //!< forward-declaration

//! used for fine-grained control over spike generation, using a callback mechanism
/*! Spike generation can be performed using spike generators. Spike generators are dummy-neurons that have their spikes
 * specified externally either defined by a Poisson firing rate or via a spike injection mechanism. Spike generators can
 * have post-synaptic connections with STDP and STP, but unlike Izhikevich neurons, they do not receive any pre-synaptic
 * input. For more information on spike generators see Section Neuron groups: Spike generators in the Tutorial.
 *
 * For fine-grained control over spike generation, individual spike times can be specified per neuron in each group.
 * This is accomplished using a callback mechanism, which is called at each time step, to specify whether a neuron has
 * fired or not. */
class SpikeGenerator {
public:
	//SpikeGenerator() {};

	//! controls spike generation using a callback mechanism
	/*! \attention The virtual method should never be called directly
	 *  \param s pointer to the simulator object
	 *  \param grpId the group id
	 *  \param i the neuron index in the group
	 *  \param currentTime the current simluation time
	 *  \param lastScheduledSpikeTime the last spike time which was scheduled
	 */
	/*! \attention The virtual method should never be called directly */
	virtual unsigned int nextSpikeTime(CARLsim* s, int grpId, int i,
											unsigned int currentTime, unsigned int lastScheduledSpikeTime) = 0;
};

//! used for fine-grained control over spike generation, using a callback mechanism
/*!
 * The user can choose from a set of primitive pre-defined connection topologies, or he can implement a topology of
 * their choice by using a callback mechanism. In the callback mechanism, the simulator calls a method on a user-defined
 * class in order to determine whether a connection should be made or not. The user simply needs to define a method that
 * specifies whether a connection should be made between a pre-synaptic neuron and a post-synaptic neuron, and the
 * simulator will automatically call the method for all possible pre- and post-synaptic pairs. The user can then specify
 * the connection's delay, initial weight, maximum weight, and whether or not it is plastic.
 */
class ConnectionGenerator {
public:
	//ConnectionGenerator() {};

	//! specifies which synaptic connections (per group, per neuron, per synapse) should be made
	/*! \attention The virtual method should never be called directly */
	virtual void connect(CARLsim* s, int srcGrpId, int i, int destGrpId, int j, float& weight, float& maxWt,
							float& delay, bool& connected) = 0;
};


//! can be used to create a custom spike monitor
/*! To retrieve outputs, a spike-monitoring callback mechanism is used. This mechanism allows the user to calculate
 * basic statistics, store spike trains, or perform more complicated output monitoring. Spike monitors are registered
 * for a group and are called automatically by the simulator every second. Similar to an address event representation
 * (AER), the spike monitor indicates which neurons spiked by using the neuron ID within a group (0-indexed) and the
 * time of the spike. Only one spike monitor is allowed per group.*/
class SpikeMonitor {
public:
	//SpikeMonitor() {};

	//! Controls actions that are performed when certain neurons fire (user-defined).
	/*! \attention The virtual method should never be called directly */
	virtual void update(CARLsim* s, int grpId, unsigned int* Nids, unsigned int* timeCnts) = 0;
};

//! can be used to create a custom group monitor
/*! To retrieve group status, a group-monitoring callback mechanism is used. This mechanism allows the user to monitor
 * basic status of a group (currently support concentrations of neuromodulator). Group monitors are registered
 * for a group and are called automatically by the simulator every second. The parameter would be the group ID, an
 * array of data, number of elements in that array.
 */
class GroupMonitor {
public:
	//GroupMonitor() {};

	virtual void update(CARLsim* s, int grpID, float* grpDA, int numData) = 0;
};

//! can be used to create a custom connection monitor
/*! To retrieve connection status, a connection-monitoring callback mechanism is used. This mechanism allows the user to
 * monitor connection status between groups (currently support weight distributions). Connection monitors are registered
 * for two groups (i.e., pre- and post- synaptic groups) and are called automatically by the simulator every second.
 * The parameter would be the pre- and post- synaptic group IDs, an array of data, number of elements in that array.
 */
class ConnectionMonitor {
public:
	//NetworkMonitor() {};

	virtual void update(CARLsim* s, int grpIdPre, int grpIdPost, float* weight, int numData) = 0;
};

#endif