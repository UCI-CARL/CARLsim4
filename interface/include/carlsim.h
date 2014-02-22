//#include "../src/snn.h"
#include <snn.h>
#include <string>


class CARLsim {
public:
	CARLsim(std::string netName = "SNN", int numConfig = 1);
	~CARLsim();

	//! shortcut to create SYN_FIXED connections with just one weight and one delay value
	// returns connection id
	int connect(int grpId1, int grpId2, const std::string& connType, float wt, float connProb, uint8_t delay);

	//! make connection from each neuron in grpId1 to 'numPostSynapses' neurons in grpId2
	// returns connection id
	int connect(int grpId1, int grpId2, const std::string& connType, float initWt, float maxWt, float connProb,
					uint8_t minDelay, uint8_t maxDelay, bool synWtType);


	//! creates a group of Izhikevich spiking neurons
	int createGroup(const std::string grpName, unsigned int nNeur, int neurType, int configId=ALL);

	//! creates a spike generator group
	int createSpikeGeneratorGroup(const std::string grpName, unsigned int nNeur, int neurType, int configId=ALL);


	void runNetwork();

	//! Sets default values for conduction decays or disables COBA if enable==false
	void setConductances(int grpId, bool isSet, int configId=ALL);

	/*!
	 * \brief Sets custom values for conduction decays or disables COBA if enable==false
	 * 
	 */
	void setConductances(int grpId, bool isSet, float tdAMPA, float tdNMDA, float tdGABAa, float tdGABAb, int configId=ALL);


	void setNeuronParameters(int grpId, float izh_a, float izh_a_sd, float izh_b, float izh_b_sd,
							 float izh_c, float izh_c_sd, float izh_d, float izh_d_sd, int configId=ALL);

	//! Sets the Izhikevich parameters a, b, c, and d of a neuron group. 
	void setNeuronParameters(int grpId, float izh_a, float izh_b, float izh_c, float izh_d, int configId=ALL);






	// GETTER / SETTERS //

	//! sets default values for conductance decays
	void setDefaultConductanceDecay(float tdAMPA, float tdNMDA, float tdGABAa, float tdGABAb);


private:
	CpuSNN* snn_;			//!< an instance of CARLsim core class
	int numConfig_;				//!< number of configurations

	bool hasConnectBegun_;	//!< flag to inform that connection setup has begun
	bool hasSetConductALL_; //!< flag to inform that conductances have been set for ALL groups (can't add more groups)
	bool hasRunNetwork_;	//!< flag to inform that network has been run

	std::vector<std::string> userWarnings_; // !< an accumulated list of user warnings
	float def_tdAMPA_;		//!< default value for AMPA decay (ms)
	float def_tdNMDA_;		//!< default value for NMDA decay (ms)
	float def_tdGABAa_;		//!< default value for GABAa decay (ms)
	float def_tdGABAb_;		//!< default value for GABAb decay (ms)
};