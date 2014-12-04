#ifndef _SIMPLE_WEIGHT_TUNER_H_
#define _SIMPLE_WEIGHT_TUNER_H_

class CARLsim;
class SpikeMonitor;
struct RangeWeight;

/*!
 * \brief Class SimpleWeightTuner
 *
 * The SimpleWeightTuner utility is a class that implements a simple search algorithm inspired by the bisection method.
 *
 * The usage scenario is to tune the weights of a specific connection (collection of synapses) so that a specific neuron
 * group fires at a predefined target firing rate—without having to recompile the network.
 * A complete code example can be found in the Tutorial subfolder 12_advanced_topics/simple_weight_tuner.
 *
 * Example usage:
 * \code
 *  // assume CARLsim object exists with the following components:
 *	int gOut=sim->createGroup("out", nNeur, EXCITATORY_NEURON);
 *	sim->setNeuronParameters(gOut, 0.02f, 0.2f, -65.0f, 8.0f);
 *	int gIn=sim->createSpikeGeneratorGroup("in", nNeur, EXCITATORY_NEURON);
 *	int c0=sim->connect(gIn, gOut, "random", RangeWeight(initWt), 0.1f, RangeDelay(1,10));
 *  // etc.
 *  sim->setupNetwork();
 *
 * 	// use weight tuner to find the weights that give 27.4 Hz spiking in gOut
 *	SimpleWeightTuner SWT(sim, 0.01, 100);
 *	SWT.setConnectionToTune(c0, 0.0);
 *	SWT.setTargetFiringRate(gOut, 27.4);
 *
 *	while (!SWT.done()) {
 *		SWT.iterate();
 *	}
 * \endcode
 *
 * \see Tutorial: \ref ch12s3s1_simple_weight_tuner
 * \see Code example: <tt>tutorial/12_advanced_topics/simple_weight_tuner/main_simple_weight_tuner.cpp</tt>
 * \since v3.0
 */
class SimpleWeightTuner {
public:
	/*!
	 * \brief Creates a new instance of class SimpleWeightTuner
	 *
	 * This method creates a new instance of class SimpleWeightTuner. A SimpleWeightTuner can be used to tune
	 * weights on the fly; that is, without having to recompile and build a network.
	 * This is useful especially for tuning feedforward weights in large-scale networks that would otherwise take
	 * a long time to repeatedly build.
	 * For tuning in more complex situations please refer to ECJ (Parameter Tuning Interface).
	 *
	 * \param[in] sim               pointer to CARLsim object
	 * \param[in] errorMargin       termination condition for error margin on target firing rate
	 * \param[in] maxIter           termination condition for max number of iterations
	 * \param[in] stepSizeFraction  step size for increasing/decreasing weights per iteration
	 * \since v3.0
	 * \see \ref ch10_ecj
	 */
	SimpleWeightTuner(CARLsim *sim, double errorMargin=1e-3, int maxIter=100, double stepSizeFraction=0.5);

	/*!
	 * \brief Destructor
	 *
	 * Cleans up all objects related to SimpleWeightTuner.
	 * \since v3.0
	 */
	~SimpleWeightTuner();

	/*!
	 * \brief Sets up the connection to tune
	 *
	 * This method sets up the connection ID to tune. The algorithm will repeatedely change the synaptic weights of
	 * this connection until the firing rate of a group (speficied via setTargetFiringRate)
	 * matches a certain target firing rate (specified in #SimpleWeightTuner).
	 *
	 * If the initial weight is set to a negative value, the algorithm will start with whatever weights have been
	 * specified when setting up the connection in CARLsim::connect. Otherwise a bias will be applied to all weights
	 * such that <tt>initWt</tt> matches the field <tt>initWt</tt> of the connection's RangeWeight struct.
	 *
	 * If adjustRange is set to true, the [minWt,maxWt] ranges will be adjusted automatically should the weight go
	 * out of bounds.
	 * \see CARLsim::connect
	 * \see RangeWeight
	 * \see CARLsim::biasWeights
	 * \since v3.0
	 */
	void setConnectionToTune(short int connId, double initWt=-1.0, bool adjustRange=true);

	/*!
	 * \brief Sets up the target firing rate of a specific group
	 *
	 * This method sets up the target firing rate (Hz) of a specific group to achieve via changing the weights of the
	 * connection specified in setConnectionToTune.
	 *
	 * A SpikeMonitor will be set up for the group if it does not already exist. SpikeMonitor::getPopMeanFiringRate
	 * will be used to determine the group's firing activity, and compare it to <tt>targetRate</tt> in order to compute
	 * the error margin.
	 *
	 * \param[in] grpId       the group ID
	 * \param[in] targetRate  target firing rate (Hz) of the group
	 * \attention If a SpikeMonitor already exists for this group, SimpleWeightTuner will use the same one and turn
	 * PersistentMode off.
	 * \since v3.0
	 * \see SpikeMonitor
	 */
	void setTargetFiringRate(int grpId, double targetRate);

	/*!
	 * \brief Performs an iteration step of the tuning algorithm
	 *
	 * This method runs the CARLsim network for a time period of <tt>runDurationMs</tt> milliseconds, throughout which a
	 * SpikeMonitor is recording the firing rate of the group ID specified in setTargetFiringRate.
	 *
	 * At the end of the iteration step the recorded firing rate is compared to the target firing rate, and the
	 * relative error is computed.
	 * If the error is smaller than the specified error margin, the algorithm terminates.
	 * If the maximum number of iteration steps is reached, the algorithm terminates.
	 * Otherwise the weights of the connection ID specified in setConnectionToTune() are updated, and the next
	 * iteration step is ready to be performed.
	 *
	 * \param[in] runDurationMs time to run the CARLsim network (ms)
	 * \param[in] printStatus   whether to print stats at the end of the iteration
	 * \since v3.0
	 * \see CARLsim::runNetwork
	 */
	void iterate(int runDurationMs=1000, bool printStatus=true);

	/*!
	 * \brief Determines whether a termination criterion has been met
	 *
	 * This method checks whether a termination criterion has been met, in which case <tt>true</tt> is returned.
	 *
	 * The algorithm will terminate if any of the following criteria have been met:
	 * - The firing rate is close enough to the target: `currentFiring - targetFiring < errorMargin'
	 * - The maximum number of iteration step has been reached: `numberOfIter > maxIter`
	 *
	 * \param[in] printMessage flag whether to print message upon termination
	 * \since v3.0
	 */
	bool done(bool printMessage=false);

	/*!
	 * \brief Resets the algorithm to initial conditions
	 *
	 * This method resets the algorithm to the initial conditions. It is implicitly called at the beginning and
	 * whenever setTargetFiringRate or setConnectionToTune has been called.
	 * \since v3.0
	 */
	void reset();


private:
	//! private method to initialize algorithm to initial conditions
	void initAlgo();

	// flags that manage state
	bool needToInitConnection_;     //!< flag indicating whether to initialize connection params
	bool needToInitTargetFiring_;   //!< flag indicating whether to initialize target firing params
	bool needToInitAlgo_;           //!< flag indicating whether to initialize algorithm

	// CARLsim data structures
	CARLsim *sim_;                  //!< pointer to CARLsim object
	SpikeMonitor *SM_;              //!< pointer to SpikeMonitor object
	int grpId_;                     //!< CARLsim group ID
	short int connId_;              //!< CARLsim connection ID
	RangeWeight* wtRange_;          //!< pointer to CARLsim RangeWeight struct

	// termination condition params
	int maxIter_;                   //!< maximum number of iterations (termination condition)
	double errorMargin_;            //!< error margin for firing rate (termination condition)
	double targetRate_;             //!< target firing rate specified in setTargetFiringRate

	// params that are updated every iteration step
	int cntIter_;                   //!< current count of iteration number
	double wtStepSize_;             //!< current weight step size
	bool wtShouldIncrease_;         //!< flag indicating the direction of weight change (increase=true, decrease=false)
	double currentError_;           //!< current firing error

	// options
	bool adjustRange_;              //!< flag indicating whether to update [minWt,maxWt] when weight goes out of bounds
	double wtInit_;                 //!< initial weight specified in setConnectionToTune
	double stepSizeFraction_;       //!< initial weight step size
};

#endif