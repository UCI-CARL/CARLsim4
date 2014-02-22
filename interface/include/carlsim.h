//#include "../src/snn.h"
#include <snn.h>
#include <string>


class CARLsim {
public:
	CARLsim(std::string netName = "SNN", int numConfig = 1);
	~CARLsim();

	//! creates a group of Izhikevich spiking neurons
	int createGroup(std::string grpName, unsigned int nNeur, int neurType, int configId=ALL);

	//! creates a spike generator group
	int createSpikeGeneratorGroup(std::string grpName, unsigned int nNeur, int neurType, int configId=ALL);


	//! Sets default values for conduction decays or disables COBA if enable==false
	void setConductances(int grpId, bool isSet, int configId=ALL);

	//! Sets custom values for conduction decays or disables COBA if enable==false
	void setConductances(int grpId, bool isSet, float tdAMPA, float tdNMDA, float tdGABAa, float tdGABAb, int configId=ALL);


	void setNeuronParameters(int grpId, float izh_a, float izh_a_sd, float izh_b, float izh_b_sd,
							 float izh_c, float izh_c_sd, float izh_d, float izh_d_sd, int configId=ALL);

	//! Sets the Izhikevich parameters a, b, c, and d of a neuron group. 
	void setNeuronParameters(int grpId, float izh_a, float izh_b, float izh_c, float izh_d, int configId=ALL);




	void connect_dummy();



	// GETTER / SETTERS //

	//! sets default values for conductance decays
	void setDefaultConductanceDecay(float tdAMPA, float tdNMDA, float tdGABAa, float tdGABAb);


private:
	CpuSNN* snn_;			//!< an instance of CARLsim core class
	int numConfig_;				//!< number of configurations
	bool hasConnectBegun_;	//!< flag to inform that connection setup has begun

	float def_tdAMPA_;		//!< default value for AMPA decay (ms)
	float def_tdNMDA_;		//!< default value for NMDA decay (ms)
	float def_tdGABAa_;		//!< default value for GABAa decay (ms)
	float def_tdGABAb_;		//!< default value for GABAb decay (ms)
};