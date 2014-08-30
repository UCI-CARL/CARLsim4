#include <vector>
class CARLsim; // forward-declaration

/*!
 * \brief a periodic SpikeGenerator (constant ISI) creating spikes at a certain rate
 *
 * This class implements a period SpikeGenerator that schedules spikes with a constant inter-spike-interval.
 * For example, a PeriodicSpikeGenerator with rate=10Hz will schedule spikes for each neuron in the group at t=100,
 * t=200, t=300, etc. If spikeAtZero is set to true, then the first spike will be scheduled at t=0.
 */
class PeriodicSpikeGenerator : public SpikeGenerator {
public:
	/*!
	 * \brief PeriodicSpikeGenerator constructor
	 * \param[in] rate the firing rate (Hz) at which to schedule spikes
	 * \param[in] spikeAtZero a boolean flag to indicate whether to insert the first spike at t=0
	 */
	PeriodicSpikeGenerator(float rate, bool spikeAtZero=true);

	//! PeriodicSpikeGenerator destructor
	~PeriodicSpikeGenerator {};

	/*!
	 * \brief schedules the next spike time
	 *
	 * This function schedules the next spike time, given the currentTime and the lastScheduledSpikeTime. It implements
	 * the virtual function of the base class.
	 * \param[in] sim pointer to a CARLsim object
	 * \param[in] grpId current group ID for which to schedule spikes
	 * \param[in] nid current neuron ID for which to schedule spikes
	 * \param[in] currentTime current time (ms) at which spike scheduler is called
	 * \param[in] lastScheduledSpikeTime the last time (ms) at which a spike was scheduled for this nid, grpId
	 * \returns the next spike time (ms)
	 */
	unsigned int nextSpikeTime(CARLsim* sim, int grpId, int nid, unsigned int currentTime, 
		unsigned int lastScheduledSpikeTime);

private:
	float rate_;		//!< spike rate (Hz)
	int isi_;			//!< inter-spike interval that results in above spike rate
	std::vector<int> nIdFiredAtZero_; //!< keep track of all neuron IDs for which a spike at t=0 has been scheduled
	bool spikeAtZero_; //!< whether to emit a spike at t=0
};