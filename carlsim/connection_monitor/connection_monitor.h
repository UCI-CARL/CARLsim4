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
 * Ver 11/12/2014
 */

#ifndef _CONN_MON_H_
#define _CONN_MON_H_

#include <vector>					// std::vector
#include <carlsim_definitions.h>	// ALL

class ConnectionMonitorCore; // forward declaration of implementation

/*!
 * \brief Class ConnectionMonitor
 *
 * The ConnectionMonitor class allows a user record weights from a particular connection. First, the user must call the
 * method CARLsim::setConnectionMonitor must be called on a specific pair of pre-synaptic and post-synaptic group. This
 * method then returns a pointer to a ConnectionMonitor object, which can be queried for connection data.
 *
 * By default, a snapshot of all the weights will be taken once per second and dumped to file. The default path to this
 * file is "results/conn_{name of pre-group}_{name of post-group}.dat". Weights can be visualized int the Matlab Offline
 * Analysis Toolbox using the ConnectionMonitor utility.
 *
 * Additionally, during a CARLsim simulation, the ConnectionMonitor object returned by CARLsim::setConnectionMonitor can
 * be queried for connection data. The user may take a snapshot of the weights at any point in time using the method
 * ConnectionMonitor::takeSnapshot. If at least two snapshots have been taken, the method
 * ConnectionMonitor::calcWeightChanges will calculate the weight changes since the last snapshot.
 * Weights can also be visualized using ConnectionMonitor::print and ConnectionMonitor::printSparse.
 *
 * Example usage:
 * \code
 * // configure a network etc. ...
 *
 * sim.setupNetwork();
 *
 * // create a ConnectionMonitor pointer to grab the pointer from setConnectionMonitor.
 * ConnectionMonitor* connMon = sim.setConnectionMonitor(grp0,grp1);
 * // run the network for a bit
 * sim.runNetwork(4,200);
 * // take a snapshot of the weights
 * wt = connMon->takeSnapshot();
 * // run network some more and take another snapshot
 * sim.runNetwork(1,500);
 * connMon->takeSnapshot();
 * // calculate and store the weight changes
 * std::vector< std::vector<float> > wtChanges = connMon->calcWeightChanges();
 * printf("The weight from pre-neuron ID 3 to post-neuron ID 7 is: %f\n",wtChanges[3][7]);
 * // or print all the weights and weight changes
 * connMon->printSparse();
 * \endcode

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

	/*!
	 * \brief Reports the weight changes since the last snapshot in a 2D weight matrix (pre x post)
	 *
	 * This function calculates the difference between the current state of the weight matrix and what it was when
	 * taking the last snapshot. Weight change is reported for every synapse, in a 2D vector where the first dimension
	 * corresponds to pre-synaptic neuron ID and the second dimension corresponds to post-synaptic neuron ID.
	 * For example, element wtChange[3][8] of the return argumentr will indicate the signed weight change since the last
	 * snapshot for the synapse that connects preNeurId=3 to postNeurId=8.
	 * Synapses that are not allocated (i.e., that do not exist) are marked as float value NAN in the weight matrix.
	 * Synapses that do exist, but have zero weight, are marked as 0.0f in the weight matrix.
	 *
	 * In order to get the current state of the weight matrix, this function will take a snapshot itself.
	 *
	 * \returns a 2D vector of weight changes, where the first dimension is pre-synaptic neuron ID and the second
	 * dimension is post-synaptic neuron ID. Non-existent synapses are marked with NAN.
	 */
	std::vector< std::vector<float> > calcWeightChanges();

	/*!
	 * \brief Returns the connection ID that this ConnectionMonitor is managing
	 *
	 * This function returns the connection ID that this ConnectionMonitor is managing. It is equivalent to the return
	 * argument of CARLsim::connect.
	 */
	short int getConnectId();

	/*!
	 * \brief Returns the number of incoming synapses for a specific post-synaptic neuron
	 *
	 * This function returns the number of incoming synapses for a specific post-synaptic neuron ID.
	 */
	int getFanIn(int neurPostId);

	/*!
	 * \brief Returns the number of outgoing synapses for a specific pre-synaptic neuron
	 *
	 * This function returns the number of outgoing synapses for a specific pre-synaptic neuron ID.
	 */
	int getFanOut(int neurPreId);

	/*!
	 * \brief Returns the number of pre-synaptic neurons 
	 *
	 * This function returns the number of neurons in the pre-synaptic group.
	 */
	int getNumNeuronsPre();

	/*!
	 * \brief Returns the number of post-synaptic neurons 
	 *
	 * This function returns the number of neurons in the post-synaptic group.
	 */
	int getNumNeuronsPost();

	/*!
	 * \brief Returns the number of allocated synapses
	 *
	 * This function returns the number of allocated synapses in the connection.
	 */
	int getNumSynapses();

	/*!
	 * \brief Returns the number of weights that have changed since the last snapshot
	 *
	 * This function returns the number of weights whose absolute has changed at least minAbsChanged (inclusive) since
	 * the last snapshot was taken.
	 *
	 * In order to get the current state of the weight matrix, this function will take a snapshot itself.
	 *
	 * \param[in]  minAbsChanged  the minimal value (inclusive) a weight has to have changed in order for it to be
	 *                            counted towards the number of changed synapses
	 */
	int getNumWeightsChanged(double minAbsChanged=1e-5);

	/*!
	 * \brief Returns the percentage of weights that have changed since the last snapshot
	 *
	 * This function returns the percentage of weights whose absolute has changed at least minAbsChanged (inclusive)
	 * since the last snapshot was taken.
	 *
	 * In order to get the current state of the weight matrix, this function will take a snapshot itself.
	 *
	 * This is a convenience function whose result is equivalent to getNumWeightsChanged()*100.0/getNumSynapses().
	 *
	 * \param[in]  minAbsChanged  the minimal value (inclusive) a weight has to have changed in order for it to be
	 *                            counted towards the percentage of changed synapses
	 */
	double getPercentWeightsChanged(double minAbsChanged=1e-5);

	/*!
	 * \brief Returns the timestamp of the current snapshot (ms since beginning of simulation)
	 *
	 * This function returns the timestamp of the current weight snapshot, reported as the amount of time that has
	 * passed since the beginning of the simulation (in milliseconds). It will not take a snapshot by itself, so the
	 * time reported here is not necessarily equal to the time reported by CARLsim::getSimTime.
	 */
	long int getTimeMsCurrentSnapshot();

	/*!
	 * \brief Returns the timestamp of the last snapshot (ms since beginning of simulation)
	 *
	 * This function returns the timestamp of the last weight snapshot, reported as the amount of time that has
	 * passed since the beginning of the simulation (in milliseconds).
	 */
	long int getTimeMsLastSnapshot();

	/*!
	 * \brief Returns the timestamp difference of the current and last snapshot
	 *
	 * This function returns the amount of time that has been passed between the current and last weight snapshot
	 * (reported in ms).
	 *
	 * This is a convenience function whose result is equivalent to getTimeMsCurrentSnapshot()-getTimeMsLastSnapshot().
	 */
	long int getTimeMsSinceLastSnapshot();

	/*!
	 * \brief Returns the absolute sum of all the weight changes since the last snapshot
	 *
	 * This function calculates the absolute sum of weight changes since the last snapshot was taken.
	 *
	 * In order to get the current state of the weight matrix, this function will take a snapshot itself.
	 */
	double getTotalAbsWeightChange();

	/*!
	 * \brief Prints the current weight state as a 2D matrix (pre x post)
	 *
	 * This function prints the current state of the weights in a readable 2D weight matrix, where the first dimension
	 * corresponds to pre-synaptic neuron ID and the second dimension corresponds to post-synaptic neuron ID.
	 * Synapses that are not allocated (i.e., that do not exist) are marked as float value NAN in the weight matrix.
	 * Synapses that do exist, but have zero weight, are marked as 0.0f in the weight matrix.
	 *
	 * In order to get the current state of the weight matrix, this function will take a snapshot itself.
	 *
	 * \note Please note that this will visualize a getNumNeuronsPre() x getNumNeuronsPost() matrix on screen. For
	 * connections between large neuronal groups, use ConnectionMonitor::printSparse.
	 */
	void print();

	/*!
	 * \brief Prints the current weight state as a sparse list of weights
	 *
	 * This function prints the current state of the weights as a readable sparse list of weights. This is also the
	 * standard format used for ConnectionMonitor to report a run summary (see CARLsim::runNetwork). A weight will be
	 * reported as [preId,postId] wt (+/- weight change in the last x ms). User can control for which post-synaptic
	 * neurons the list should be generated, and set limits on how many connections to print in total and per line.
	 *
	 * In order to get the current state of the weight matrix, this function will take a snapshot itself.
	 *
	 * \note Please note that this is the preferred way to visualize connections between large neuronal groups. The
	 * method ConnectionMonitor::print should primarily be used for small-sized groups.
	 *
	 * \param[in] neurPostId   The neuron ID of the post-synaptic group for which to generate the sparse weight list.
	 *                         Set to ALL to generate the list for all post-synaptic neurons.
	 * \param[in] maxConn      The maximum number of weights to print.
	 * \param[in] connPerLine  The number of weights to print per line.
	 */
	void printSparse(int neurPostId=ALL, int maxConn=100, int connPerLine=4);

	/*!
	 * \brief Takes a snapshot of the current weight state
	 *
	 * This function takes a snapshot of the current weight matrix, and returns it as a 2D vector where the first
	 * dimension corresponds to pre-synaptic neuron ID and the second dimension corresponds to post-synaptic neuron ID.
	 * For example, element wtChange[3][8] of the return argumentr will indicate the signed weight change since the last
	 * snapshot for the synapse that connects preNeurId=3 to postNeurId=8.
	 * Synapses that are not allocated (i.e., that do not exist) are marked as float value NAN in the weight matrix.
	 * Synapses that do exist, but have zero weight, are marked as 0.0f in the weight matrix.
	 *
	 * \returns a 2D vector of weights, where the first dimension is pre-synaptic neuron ID and the second dimension
	 * is post-synaptic neuron ID. Non-existent synapses are marked with NAN.
	 */
	std::vector< std::vector<float> > takeSnapshot();

private:
	//! This is a pointer to the actual implementation of the class. The user should never directly instantiate it.
	ConnectionMonitorCore* connMonCorePtr_;
};

#endif