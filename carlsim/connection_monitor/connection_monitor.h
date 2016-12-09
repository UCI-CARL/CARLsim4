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

#ifndef _CONN_MON_H_
#define _CONN_MON_H_

#include <vector>					// std::vector
#include <carlsim_definitions.h>	// ALL

class ConnectionMonitorCore; // forward declaration of implementation

/*!
 * \brief Class ConnectionMonitor
 *
 * The ConnectionMonitor class allows a user record weights from a particular connection. First, the method
 * CARLsim::setConnectionMonitor must be called on a specific pair of pre-synaptic and post-synaptic group. This
 * method then returns a pointer to a ConnectionMonitor object, which can be queried for connection data.
 *
 * By default, a snapshot of all the weights will be taken once per second and dumped to file.
 * The easiest way to use a ConnectionMonitor is to call CARLsim::setConnectionMonitor with file string "default".
 * This will create a file with path "results/conn_{name of pre-group}_{name of post-group}.dat".
 * It is also possible to specify a custom file string instead.
 * Alternatively, the user may suppress creation of the binary file by using file string "null" instead.
 *
 * Periodic storing can be disabled by calling ConnectionMonitor::setUpdateTimeInterval with argument intervalSec=-1.
 *
 * Additionally, during a CARLsim simulation, the ConnectionMonitor object returned by CARLsim::setConnectionMonitor can
 * be queried for connection data. The user may take a snapshot of the weights at any point in time using the method
 * ConnectionMonitor::takeSnapshot.
 * Note that a snapshot taken with this method will also be stored in the binary file.
 * However, the binary file will never contain the same snapshot (captured with a certain timestamp) twice.
 *
 * If at least two snapshots have been taken, the method ConnectionMonitor::calcWeightChanges will calculate the weight
 * changes since the last snapshot.
 * To make sure you are comparing the right snapshots, compare the timestamps returend by
 * ConnectionMonitor::getTimeMsCurrentSnapshot and ConnectionMonitor::getTimeMsLastSnapshot.
 *
 * Weights can be visualized in the Matlab Offline Analysis Toolbox (OAT) using the ConnectionMonitor utility.
 * The OAT offers ways to plot 2D weight matrices, as well as receptive fields and response fields.
 *
 * Weights can also be visualized in C++ using ConnectionMonitor::print and ConnectionMonitor::printSparse.
 *
 * Example to store weights in binary every second:
 * \code
 * // configure a network, etc. ...
 * sim.setupNetwork();
 *
 * // direct storing of snapshots to default file "results/conn_{name of pre}_{name of post}.dat".
 * sim.setConnectionMonitor(grp0, grp1, "default");
 *
 * // run the network for 10 seconds
 * sim.runNetwork(10,0);
 * \endcode
 * 
 * Example that stores weight at the beginning and end of a simulation:
 * \code
 * // configure a network, etc. ...
 * sim.setupNetwork();
 *
 * // direct storing of snapshots to default file "results/conn_{name of pre}_{name of post}.dat".
 * // additionally, grab the pointer to the monitor object
 * ConnectionMonitor* CM = sim.setConnectionMonitor(grp0, grp1, "default");
 *
 * // disable periodid storing of snapshots in binary ...
 * CM->setUpdateTimeIntervalSec(-1);
 *
 * // ... and instead take a snapshot yourself and put it in the binary
 * CM->takeSnapshot();
 *
 * // run the network for 10 seconds
 * sim.runNetwork(10,0);
 *
 * // take another snapshot at the end and put it in the binary
 * CM->takeSnapshot();
 * \endcode
 *
 * Example that periodically stores to binary and does some analysis on returned weight vector
 * \code
 * // configure a network, etc. ...
 * sim.setupNetwork();
 *
 * // direct storing of snapshots to default file "results/conn_{name of pre}_{name of post}.dat".
 * // additionally, grab the pointer to the monitor object
 * ConnectionMonitor* CM = sim.setConnectionMonitor(grp0, grp1, "default");
 *
 * // retrieve initial weights (this snapshot will also end up in the binary)
 * std::vector< std::vector<float> > wt = CM->takeSnapshot();
 * printf("The weight from pre-neuron ID 3 to post-neuron ID 7 is: %f\n",wt[3][7]);
 *
 * // run the network for 5 seconds and print 2D weight matrix
 * sim.runNetwork(5,0);
 * CM->print();
 *
 * // run the network for an additional 5 seconds and retrieve weight changes between last snapshot
 * // (at the beginning) and now (after 10 seconds)
 * sim.runNetwork(5,0);
 * std::vector< std::vector<float> > wtChanges = CM->calcWeightChanges();
 * \endcode
 *
 * \note A snapshot taken programmatically with ConnectionMonitor::takeSnapshot can be put in the binary file by
 * setting an optional input flag <tt>writeToFile</tt> to true.
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
	 * In order to get the current state of the weight matrix, this function will take a snapshot itself, but will
	 * not write it to file.
	 *
	 * \returns a 2D vector of weight changes, where the first dimension is pre-synaptic neuron ID and the second
	 * dimension is post-synaptic neuron ID. Non-existent synapses are marked with NAN.
	 * \since v3.0
	 */
	std::vector< std::vector<float> > calcWeightChanges();

	/*!
	 * \brief Returns the connection ID that this ConnectionMonitor is managing
	 *
	 * This function returns the connection ID that this ConnectionMonitor is managing. It is equivalent to the return
	 * argument of CARLsim::connect.
	 * \since v3.0
	 */
	short int getConnectId();

	/*!
	 * \brief Returns the number of incoming synapses for a specific post-synaptic neuron
	 *
	 * This function returns the number of incoming synapses for a specific post-synaptic neuron ID.
	 * \since v3.0
	 */
	int getFanIn(int neurPostId);

	/*!
	 * \brief Returns the number of outgoing synapses for a specific pre-synaptic neuron
	 *
	 * This function returns the number of outgoing synapses for a specific pre-synaptic neuron ID.
	 * \since v3.0
	 */
	int getFanOut(int neurPreId);

	/*!
	 * \brief Returns the max weight in the connection
	 *
	 * This function returns the maximum weight value of all synapses in the connection.
	 *
	 * If getCurrent is set to true, then the function will return the *currently* largest weight value. In a
	 * plastic connection, this value might be different from the upper bound of the weight range specified
	 * when setting up the network (i.e., the max field of the ::RangeWeight struct).
	 * In order to get the current state of the weight matrix, this function will take a snapshot itself, but will
	 * not write it to file.
	 *
	 * If getCurrent is set to false, then the upper bound of the configured weight range will be returned.
	 *
	 * \param[in] getCurrent whether to return the currently largest weight value (true) or the upper bound of
	 *                       the weight range specified during setup (false). Default: false.
	 * \since v3.1
	 */
	double getMaxWeight(bool getCurrent=false);

	/*!
	 * \brief Returns the min weight in the connection
	 *
	 * This function returns the minimum weight value of all synapses in the connection.
	 *
	 * If getCurrent is set to true, then the function will return the *currently* smallest weight value. In a
	 * plastic connection, this value might be different from the lower bound of the weight range specified
	 * when setting up the network (i.e., the min field of the ::RangeWeight struct).
	 * In order to get the current state of the weight matrix, this function will take a snapshot itself, but will
	 * not write it to file.
	 *
	 * If getCurrent is set to false, then the lower bound of the configured weight range will be returned.
	 *
	 * \param[in] getCurrent whether to return the currently smallest weight value (true) or the lower bound of
	 *                       the weight range specified during setup (false). Default: false.
	 * \since v3.1
	 */
	double getMinWeight(bool getCurrent=false);

	/*!
	 * \brief Returns the number of pre-synaptic neurons 
	 *
	 * This function returns the number of neurons in the pre-synaptic group.
	 * \since v3.0
	 */
	int getNumNeuronsPre();

	/*!
	 * \brief Returns the number of post-synaptic neurons 
	 *
	 * This function returns the number of neurons in the post-synaptic group.
	 * \since v3.0
	 */
	int getNumNeuronsPost();

	/*!
	 * \brief Returns the number of allocated synapses
	 *
	 * This function returns the number of allocated synapses in the connection.
	 * \since v3.0
	 */
	int getNumSynapses();

	/*!
	 * \brief Returns the number of weights that have changed since the last snapshot
	 *
	 * This function returns the number of weights whose absolute value has changed at least minAbsChanged (inclusive)
	 * since the last snapshot was taken.
	 *
	 * In order to get the current state of the weight matrix, this function will take a snapshot itself, but will
	 * not write it to file.
	 *
	 * \param[in]  minAbsChanged  the minimal value (inclusive) a weight has to have changed in order for it to be
	 *                            counted towards the number of changed synapses
	 * \since v3.0
	 */
	int getNumWeightsChanged(double minAbsChanged=1e-5);

	/*!
	 * \brief Returns the number of weights in the connection whose values are within some range (inclusive)
	 *
	 * This function returns the number of synaptic weights whose values are within some specific range
	 * e[minVal,maxVal] (inclusive).
	 *
	 * In order to get the current state of the weight matrix, this function will take a snapshot itself, but will
	 * not write it to file.
	 *
	 * \param[in] minValue the lower bound of the weight range (inclusive)
	 * \param[in] maxValue the upper bound of the weight range (inclusive)
	 * \note CM.getNumWeightsInRange(CM.getMinWeight(false),CM.getMaxWeight(false)) is the same as
	 * CM.getNumSynapses().
	 * \see ConnectionMonitor::getPercentWeightsInRange
	 * \see ConnectionMonitor::getNumWeightsWithValue
	 * \since v3.1
	 */
	int getNumWeightsInRange(double minValue, double maxValue);

	/*!
	 * \brief Returns the number of weights in the connection with a particular value
	 *
	 * This function returns the number of synaptic weights that have exactly some specific value.
	 * It could be used to determine the sparsity of the connection matrix (wtValue==0.0f).
	 *
	 * In order to get the current state of the weight matrix, this function will take a snapshot itself, but will
	 * not write it to file.
	 *
	 * Machine epsilon (FLT_EPSILON) is used for floating point equality. That is, the weight value is
	 * considered equal to the input value if fabs(wt-value)<=FLT_EPSILON (inclusive).
	 *
	 * This is a convenience function whose result is equivalent to
	 * getNumWeightsInRange(value-FLT_EPSILON,value+FLT_EPSILON).
	 *
	 * \param[in] value the exact weight value to look for
	 * \see ConnectionMonitor::getPercentWeightsWithValue
	 * \see ConnectionMonitor::getNumWeightsInRange
	 * \since v3.1
	 */
	int getNumWeightsWithValue(double value);

	/*!
	 * \brief Returns the percentage of weights whose values are within some range (inclusive)
	 *
	 * This function returns the percentage of synaptic weights whose values are within some specific range
	 * e[minVal,maxVal] (inclusive).
	 *
	 * In order to get the current state of the weight matrix, this function will take a snapshot itself, but will
	 * not write it to file.
	 *
	 * This is a convenience function whose result is equivalent to
	 * getNumWeightsInRange(minValue,maxValue)*100.0/getNumSynapses().
	 *
	 * \param[in] minValue the lower bound of the weight range (inclusive)
	 * \param[in] maxValue the upper bound of the weight range (inclusive)
	 * \note CM.getNumWeightsInRange(CM.getMinWeight(false),CM.getMaxWeight(false)) is the same as
	 * CM.getNumSynapses().
	 * \see ConnectionMonitor::getNumWeightsInRange
	 * \see ConnectionMonitor::getNumWeightsWithValue
	 * \since v3.1
	 */
	double getPercentWeightsInRange(double minValue, double maxValue);

	/*!
	 * \brief Returns the percentage of weights in the connection with a particular value
	 *
	 * This function returns the percentage of synaptic weights that have exactly some specific value.
	 * It could be used to determine the sparsity of the connection matrix (wtValue==0.0f).
	 *
	 * In order to get the current state of the weight matrix, this function will take a snapshot itself, but will
	 * not write it to file.
	 *
	 * Machine epsilon (FLT_EPSILON) is used for floating point equality. That is, the weight value is
	 * considered equal to the input value if fabs(wt-value)<=FLT_EPSILON (inclusive).
	 *
	 * This is a convenience function whose result is equivalent to
	 * getNumWeightsWithValue(value)*100.0/getNumSynapses().
	 *
	 * \param[in] value the exact weight value to look for
	 * \see ConnectionMonitor::getNumWeightsWithValue
	 * \see ConnectionMonitor::getNumWeightsInRange
	 * \since v3.1
	 */
	double getPercentWeightsWithValue(double value);


	/*!
	 * \brief Returns the percentage of weights that have changed since the last snapshot
	 *
	 * This function returns the percentage of weights whose absolute has changed at least minAbsChanged (inclusive)
	 * since the last snapshot was taken.
	 *
	 * In order to get the current state of the weight matrix, this function will take a snapshot itself, but will
	 * not write it to file.
	 *
	 * This is a convenience function whose result is equivalent to getNumWeightsChanged()*100.0/getNumSynapses().
	 *
	 * \param[in]  minAbsChanged  the minimal value (inclusive) a weight has to have changed in order for it to be
	 *                            counted towards the percentage of changed synapses
	 * \since v3.0
	 */
	double getPercentWeightsChanged(double minAbsChanged=1e-5);

	/*!
	 * \brief Returns the timestamp of the current snapshot (ms since beginning of simulation)
	 *
	 * This function returns the timestamp of the current weight snapshot, reported as the amount of time that has
	 * passed since the beginning of the simulation (in milliseconds). It will not take a snapshot by itself, so the
	 * time reported here is not necessarily equal to the time reported by CARLsim::getSimTime.
	 * \since v3.0
	 */
	long int getTimeMsCurrentSnapshot();

	/*!
	 * \brief Returns the timestamp of the last snapshot (ms since beginning of simulation)
	 *
	 * This function returns the timestamp of the last weight snapshot, reported as the amount of time that has
	 * passed since the beginning of the simulation (in milliseconds).
	 * \since v3.0
	 */
	long int getTimeMsLastSnapshot();

	/*!
	 * \brief Returns the timestamp difference of the current and last snapshot
	 *
	 * This function returns the amount of time that has been passed between the current and last weight snapshot
	 * (reported in ms).
	 *
	 * This is a convenience function whose result is equivalent to getTimeMsCurrentSnapshot()-getTimeMsLastSnapshot().
	 * \since v3.0
	 */
	long int getTimeMsSinceLastSnapshot();

	/*!
	 * \brief Returns the absolute sum of all the weight changes since the last snapshot
	 *
	 * This function calculates the absolute sum of weight changes since the last snapshot was taken.
	 *
	 * In order to get the current state of the weight matrix, this function will take a snapshot itself, but will
	 * not write it to file.
	 * \since v3.0
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
	 * In order to get the current state of the weight matrix, this function will take a snapshot itself, but will
	 * not write it to file.
	 *
	 * \note Please note that this will visualize a getNumNeuronsPre() x getNumNeuronsPost() matrix on screen. For
	 * connections between large neuronal groups, use ConnectionMonitor::printSparse.
	 * \since v3.0
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
	 * In order to get the current state of the weight matrix, this function will take a snapshot itself, but will
	 * not write it to file.
	 *
	 * \note Please note that this is the preferred way to visualize connections between large neuronal groups. The
	 * method ConnectionMonitor::print should primarily be used for small-sized groups.
	 *
	 * \param[in] neurPostId   The neuron ID of the post-synaptic group for which to generate the sparse weight list.
	 *                         Set to ALL to generate the list for all post-synaptic neurons.
	 * \param[in] maxConn      The maximum number of weights to print.
	 * \param[in] connPerLine  The number of weights to print per line.
	 * \since v3.0
	 */
	void printSparse(int neurPostId=ALL, int maxConn=100, int connPerLine=4);

	/*!
	 * \brief Sets the time interval (seconds) for writing snapshots to file
	 *
	 * This function sets the time interval (seconds) for writing weight snapshots to file.
	 * The first snapshot will be written at time t=0, and every intervalSec seconds later.
	 *
	 * In order to disable the periodic storing of weights to file alltogether, set intervalSec to -1.
	 * In this case, only weight snapshots acquired via ConnectionMonitor::takeSnapshots will end up in the binary.
	 *
	 * \param[in] intervalSec  The update time interval (number of seconds) for writing snapshots to file.
	 *                         Set to -1 to disable periodic weight storing. Default: 1 (every second).
	 *
	 * \since v3.0
	 */
	void setUpdateTimeIntervalSec(int intervalSec);

	/*!
	 * \brief Takes a snapshot of the current weight state
	 *
	 * This function takes a snapshot of the current weight matrix, stores it in the binary as well as returns it as a
	 * 2D vector where the first dimension corresponds to pre-synaptic neuron ID and the second dimension corresponds 
	 * to post-synaptic neuron ID.
	 * For example, element wtChange[3][8] of the return argumentr will indicate the signed weight change since the last
	 * snapshot for the synapse that connects preNeurId=3 to postNeurId=8.
	 * Synapses that are not allocated (i.e., that do not exist) are marked as float value NAN in the weight matrix.
	 * Synapses that do exist, but have zero weight, are marked as 0.0f in the weight matrix.
	 *
	 * \returns a 2D vector of weights, where the first dimension is pre-synaptic neuron ID and the second dimension
	 * is post-synaptic neuron ID. Non-existent synapses are marked with NAN.
	 * \note Every snapshot taken will also be stored in the binary file.
	 * \since v3.0
	 */
	std::vector< std::vector<float> > takeSnapshot();

private:
	//! This is a pointer to the actual implementation of the class. The user should never directly instantiate it.
	ConnectionMonitorCore* connMonCorePtr_;
};

#endif