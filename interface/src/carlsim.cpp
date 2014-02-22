#include "../include/carlsim.h"

#include "../../src/snn.h"
#include <string>



/// **************************************************************************************************************** ///
/// CONSTRUCTOR / DESTRUCTOR
/// **************************************************************************************************************** ///

CARLsim::CARLsim(std::string netName, int numConfig) {

	snn_ = new CpuSNN(netName.c_str());
	numConfig_ = numConfig;
	hasConnectBegun_ = false;

	// set default time constants for synaptic current decay
	// TODO: add ref
	def_tdAMPA_  = 5.0f;	// default decay time for AMPA (ms)
	def_tdNMDA_  = 150.0f;	// default decay time for NMDA (ms)
	def_tdGABAa_ = 6.0f;	// default decay time for GABAa (ms)
	def_tdGABAb_ = 150.0f;	// default decay time for GABAb (ms)
}

CARLsim::~CARLsim() {
	// deallocate all dynamically allocated structures
	if (snn_!=NULL)
		delete snn_;
}


/// **************************************************************************************************************** ///
/// PUBLIC METHODS
/// **************************************************************************************************************** ///

// create group of Izhikevich spiking neurons
int CARLsim::createGroup(std::string grpName, unsigned int nNeur, int neurType, int configId) {
	assert(!hasConnectBegun_); // TODO: make nice error message
	return snn_->createGroup(grpName.c_str(),nNeur,neurType,configId);
}


// create group of spike generators
int CARLsim::createSpikeGeneratorGroup(std::string grpName, unsigned int nNeur, int neurType, int configId) {
	assert(!hasConnectBegun_); // TODO: make nicer
	return snn_->createSpikeGeneratorGroup(grpName.c_str(),nNeur,neurType,configId);
}


// set conductance values, use defaults
void CARLsim::setConductances(int grpId, bool isSet, int configId) {
	if (isSet) { // enable conductances, use default values
		snn_->setConductances(grpId,true,def_tdAMPA_,def_tdNMDA_,def_tdGABAa_,def_tdGABAb_,configId);
	} else { // discable conductances
		snn_->setConductances(grpId,false,0.0f,0.0f,0.0f,0.0f,configId);
	}
}

// set conductances values, custom
void CARLsim::setConductances(int grpId, bool isSet, float tdAMPA, float tdNMDA, float tdGABAa, float tdGABAb,
								int configId) {
	if (isSet) { // enable conductances, use custom values
		snn_->setConductances(grpId,true,tdAMPA,tdNMDA,tdGABAa,tdGABAb,configId);
	} else { // discable conductances
		snn_->setConductances(grpId,false,0.0f,0.0f,0.0f,0.0f,configId);
	}
}


// set neuron parameters for Izhikevich neuron, with standard deviations
void CARLsim::setNeuronParameters(int grpId, float izh_a, float izh_a_sd, float izh_b, float izh_b_sd,
							 		float izh_c, float izh_c_sd, float izh_d, float izh_d_sd, int configId) {
	// wrapper identical to core func
	snn_->setNeuronParameters(grpId, izh_a, izh_a_sd, izh_b, izh_b_sd, izh_c, izh_c_sd, izh_d, izh_d_sd, configId);
}

// set neuron parameters for Izhikevich neuron
void CARLsim::setNeuronParameters(int grpId, float izh_a, float izh_b, float izh_c, float izh_d, int configId) {
	// set standard deviations of Izzy params to zero
	snn_->setNeuronParameters(grpId, izh_a, 0.0f, izh_b, 0.0f, izh_c, 0.0f, izh_d, 0.0f, configId);
}



void CARLsim::connect_dummy() {
	hasConnectBegun_ = true; // inform class that creating groups etc. is no longer allowed
}



/// **************************************************************************************************************** ///
/// PUBLIC GETTERS / SETTERS
/// **************************************************************************************************************** ///

// set default values for conductance decay times
void CARLsim::setDefaultConductanceDecay(float tdAMPA, float tdNMDA, float tdGABAa, float tdGABAb) {
	def_tdAMPA_  = tdAMPA;
	def_tdNMDA_  = tdNMDA;
	def_tdGABAa_ = tdGABAa;
	def_tdGABAb_ = tdGABAb;
}

/// **************************************************************************************************************** ///
/// PRIVATE METHODS
/// **************************************************************************************************************** ///
