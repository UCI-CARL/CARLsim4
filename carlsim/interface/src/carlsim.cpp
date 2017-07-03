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

#include <carlsim.h>
#include <user_errors.h>

//#include <callback.h>



#include <callback_core.h>

#include <iostream>		// std::cout, std::endl
#include <sstream>		// std::stringstream
#include <algorithm>	// std::find, std::transform

#include <snn.h>

// includes for mkdir
#if CREATE_SPIKEDIR_IF_NOT_EXISTS
	#if defined(WIN32) || defined(WIN64)
	#else
		#include <sys/stat.h>
		#include <errno.h>
		#include <libgen.h>
	#endif
#endif

// NOTE: Conceptual code documentation should go in carlsim.h. Do not include extensive high-level documentation here,
// but do document your code.



class CARLsim::Impl {
public:
	// +++++ PUBLIC METHODS: SETUP / TEAR-DOWN ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	Impl(CARLsim* sim, const std::string& netName, SimMode prferredSimMode, LoggerMode loggerMode, int randSeed) {
		netName_ 					= netName;
		loggerMode_ 				= loggerMode;
		preferredSimMode_			= prferredSimMode;
		randSeed_					= randSeed;
		enablePrint_ = false;
		copyState_ = false;

		numConnections_				= 0;

		hasSetHomeoALL_ 			= false;
		hasSetHomeoBaseFiringALL_ 	= false;
		hasSetSTDPALL_ 				= false;
		hasSetSTPALL_ 				= false;
		hasSetConductances_			= false;
		carlsimState_				= CONFIG_STATE;

		sim_ = sim;
		snn_ = NULL;

		CARLsimInit(); // move everything else out of constructor
	}

	~Impl() {
		// save simulation
		if (carlsimState_ == SETUP_STATE || carlsimState_ == RUN_STATE)
			saveSimulation(def_save_fileName_,def_save_synapseInfo_);

		// deallocate all dynamically allocated structures
		for (int i=0; i<spkGen_.size(); i++) {
			if (spkGen_[i]!=NULL)
				delete spkGen_[i];
			spkGen_[i]=NULL;
		}
		for (int i=0; i<connGen_.size(); i++) {
			if (connGen_[i]!=NULL)
				delete connGen_[i];
			connGen_[i]=NULL;
		}
		if (snn_!=NULL)
			delete snn_;
		snn_=NULL;
	}


	// +++++++++ PUBLIC METHODS: SETTING UP A SIMULATION ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	// Connects a presynaptic to a postsynaptic group using one of the primitive types
	short int connect(int grpId1, int grpId2, const std::string& connType, const RangeWeight& wt, float connProb,
			const RangeDelay& delay, const RadiusRF& radRF, bool synWtType, float mulSynFast, float mulSynSlow)
	{
		std::string funcName = "connect(\""+getGroupName(grpId1)+"\",\""+getGroupName(grpId2)+"\")";
		std::stringstream grpId1str; grpId1str << "Group Id " << grpId1;
		std::stringstream grpId2str; grpId2str << "Group Id " << grpId2;
		UserErrors::assertTrue(grpId1!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, grpId1str.str()); // grpId can't be ALL
		UserErrors::assertTrue(grpId2!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, grpId2str.str());
		UserErrors::assertTrue(!isPoissonGroup(grpId2), UserErrors::WRONG_NEURON_TYPE, funcName, grpId2str.str() +
			" is PoissonGroup, connect");
		UserErrors::assertTrue(wt.max>=0.0f, UserErrors::CANNOT_BE_NEGATIVE, funcName, "wt.max");
		UserErrors::assertTrue(wt.min>=0.0f, UserErrors::CANNOT_BE_NEGATIVE, funcName, "wt.min");
		UserErrors::assertTrue(wt.init>=0.0f, UserErrors::CANNOT_BE_NEGATIVE, funcName, "wt.init");
		UserErrors::assertTrue(connProb>=0.0f && connProb<=1.0f, UserErrors::MUST_BE_IN_RANGE, funcName,
			"Connection Probability connProb", "[0,1]");
		UserErrors::assertTrue(delay.min>0, UserErrors::MUST_BE_POSITIVE, funcName, "delay.min");
		UserErrors::assertTrue(connType.compare("one-to-one")!=0
			|| connType.compare("one-to-one")==0 && getGroupNumNeurons(grpId1) == getGroupNumNeurons(grpId2),
			UserErrors::MUST_BE_IDENTICAL, funcName, "For type \"one-to-one\", number of neurons in pre and post");
		UserErrors::assertTrue(connType.compare("gaussian")!=0
			|| connType.compare("gaussian")==0 && (radRF.radX>-1 || radRF.radY>-1 || radRF.radZ>-1),
			UserErrors::CANNOT_BE_NEGATIVE, funcName, "Receptive field radius for type \"gaussian\"");
		UserErrors::assertTrue(synWtType==SYN_PLASTIC || synWtType==SYN_FIXED && wt.init==wt.max,
			UserErrors::MUST_BE_IDENTICAL, funcName, "For fixed synapses, initWt and maxWt");
		UserErrors::assertTrue(mulSynFast>=0.0f, UserErrors::CANNOT_BE_NEGATIVE, funcName, "mulSynFast");
		UserErrors::assertTrue(mulSynSlow>=0.0f, UserErrors::CANNOT_BE_NEGATIVE, funcName, "mulSynSlow");

		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName,
			"CONFIG.");
		assert(++numConnections_ <= MAX_CONN_PER_SNN);

		// throw a warning if "one-to-one" is used in combination with a non-zero RF
		if (connType.compare("one-to-one")==0 && (radRF.radX>0 || radRF.radY>0 || radRF.radZ>0)) {
			userWarnings_.push_back("RadiusRF>0 will be ignored for connection type \"one-to-one\"");
		}

		// TODO: enable support for non-zero min
		if (fabs(wt.min)>1e-15) {
			std::cerr << funcName << ": " << wt << ". Non-zero minimum weights are not yet supported.\n" << std::endl;
			assert(false);
		}

		// groups cannot be both chemically (synaptically) and electrically (compartmentally) connected
		UserErrors::assertTrue(std::find(connComp_[grpId1].begin(), connComp_[grpId1].end(), grpId2) ==
			connComp_[grpId1].end(), UserErrors::CANNOT_BE_CONN_SYN_AND_COMP, funcName,
			grpId1str.str() + " and " + grpId2str.str());

		UserErrors::assertTrue(std::find(connComp_[grpId2].begin(), connComp_[grpId2].end(), grpId1) ==
			connComp_[grpId2].end(), UserErrors::CANNOT_BE_CONN_SYN_AND_COMP, funcName,
			grpId1str.str() + " and " + grpId2str.str());

		// add synaptic connection to 2D matrix
		connSyn_[grpId1].push_back(grpId2);

		return snn_->connect(grpId1, grpId2, connType, wt.init, wt.max, connProb, delay.min, delay.max,
			radRF, mulSynFast, mulSynSlow, synWtType);
	}

	// custom connectivity profile
	short int connect(int grpId1, int grpId2, ConnectionGenerator* conn, bool synWtType) {
		std::string funcName = "connect(\""+getGroupName(grpId1)+"\",\""+getGroupName(grpId2)+"\")";
		std::stringstream grpId1str; grpId1str << ". Group Id " << grpId1;
		std::stringstream grpId2str; grpId2str << ". Group Id " << grpId2;
		UserErrors::assertTrue(grpId1!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, grpId1str.str()); // grpId can't be ALL
		UserErrors::assertTrue(grpId2!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, grpId2str.str());
		UserErrors::assertTrue(!isPoissonGroup(grpId2), UserErrors::WRONG_NEURON_TYPE, funcName, grpId2str.str() +
			" is PoissonGroup, connect");
		UserErrors::assertTrue(conn!=NULL, UserErrors::CANNOT_BE_NULL, funcName, "ConnectionGenerator* conn");

		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");
		assert(++numConnections_ <= MAX_CONN_PER_SNN);

		// groups cannot be both chemically (synaptically) and electrically (compartmentally) connected
		UserErrors::assertTrue(std::find(connComp_[grpId1].begin(), connComp_[grpId1].end(), grpId2) ==
			connComp_[grpId1].end(), UserErrors::CANNOT_BE_CONN_SYN_AND_COMP, funcName,
			grpId1str.str() + " and " + grpId2str.str());
		UserErrors::assertTrue(std::find(connComp_[grpId2].begin(), connComp_[grpId2].end(), grpId1) ==
			connComp_[grpId2].end(), UserErrors::CANNOT_BE_CONN_SYN_AND_COMP, funcName,
			grpId1str.str() + " and " + grpId2str.str());

		// add synaptic connection to 2D matrix
		connSyn_[grpId1].push_back(grpId2);

		// TODO: check for sign of weights
		// ConnectionGeneratorCore* CGC = new ConnectionGeneratorCore(this, conn);
		ConnectionGeneratorCore* CGC = new ConnectionGeneratorCore(sim_, conn);
		connGen_.push_back(CGC);
		return snn_->connect(grpId1, grpId2, CGC, 1.0f, 1.0f, synWtType);
	}

	// custom connectivity profile
	short int connect(int grpId1, int grpId2, ConnectionGenerator* conn, float mulSynFast, float mulSynSlow,
		bool synWtType)
	{
		std::string funcName = "connect(\""+getGroupName(grpId1)+"\",\""+getGroupName(grpId2)+"\")";
		std::stringstream grpId1str; grpId1str << ". Group Id " << grpId1;
		std::stringstream grpId2str; grpId2str << ". Group Id " << grpId2;
		UserErrors::assertTrue(grpId1!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, grpId1str.str()); // grpId can't be ALL
		UserErrors::assertTrue(grpId2!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, grpId2str.str());
		UserErrors::assertTrue(!isPoissonGroup(grpId2), UserErrors::WRONG_NEURON_TYPE, funcName, grpId2str.str() +
			" is PoissonGroup, connect");
		UserErrors::assertTrue(conn!=NULL, UserErrors::CANNOT_BE_NULL, funcName);
		UserErrors::assertTrue(mulSynFast>=0.0f, UserErrors::CANNOT_BE_NEGATIVE, funcName, "mulSynFast");
		UserErrors::assertTrue(mulSynSlow>=0.0f, UserErrors::CANNOT_BE_NEGATIVE, funcName, "mulSynSlow");
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, 
			funcName, "CONFIG.");
		assert(++numConnections_ <= MAX_CONN_PER_SNN);

		// groups cannot be both chemically (synaptically) and electrically (compartmentally) connected
		UserErrors::assertTrue(std::find(connComp_[grpId1].begin(), connComp_[grpId1].end(), grpId2) ==
			connComp_[grpId1].end(), UserErrors::CANNOT_BE_CONN_SYN_AND_COMP, funcName,
			grpId1str.str() + " and " + grpId2str.str());
		UserErrors::assertTrue(std::find(connComp_[grpId2].begin(), connComp_[grpId2].end(), grpId1) ==
			connComp_[grpId2].end(), UserErrors::CANNOT_BE_CONN_SYN_AND_COMP, funcName,
			grpId1str.str() + " and " + grpId2str.str());

		// add synaptic connection to 2D matrix
		connSyn_[grpId1].push_back(grpId2);

		// ConnectionGeneratorCore* CGC = new ConnectionGeneratorCore(this, conn);
		ConnectionGeneratorCore* CGC = new ConnectionGeneratorCore(sim_, conn);
		connGen_.push_back(CGC);
		return snn_->connect(grpId1, grpId2, CGC, mulSynFast, mulSynSlow, synWtType);
	}

	short int connectCompartments(int grpIdLower, int grpIdUpper) {
		std::stringstream funcName; funcName << "connectCompartments(" << grpIdLower << "," << grpIdUpper << ")";

		// grpIDs must be valid, cannot be identical
		std::stringstream grpIdLowerStr; grpIdLowerStr << "Group Id " << grpIdLower;
		std::stringstream grpIdUpperStr; grpIdUpperStr << "Group Id " << grpIdUpper;
		UserErrors::assertTrue(grpIdLower >= 0 && grpIdLower<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName.str(),
			grpIdLowerStr.str(), "[0,getNumGroups()]");
		UserErrors::assertTrue(grpIdUpper >= 0 && grpIdUpper<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName.str(),
			grpIdUpperStr.str(), "[0,getNumGroups()]");
		UserErrors::assertTrue(grpIdLower != grpIdUpper, UserErrors::CANNOT_BE_IDENTICAL, funcName.str(),
			grpIdLowerStr.str() + " and " + grpIdUpperStr.str());

		// groups cannot be spike generators
		UserErrors::assertTrue(!isPoissonGroup(grpIdLower), UserErrors::WRONG_NEURON_TYPE, funcName.str(),
			grpIdLowerStr.str() + " is PoissonGroup, connectCompartments");
		UserErrors::assertTrue(!isPoissonGroup(grpIdUpper), UserErrors::WRONG_NEURON_TYPE, funcName.str(),
			grpIdUpperStr.str() + " is PoissonGroup, connectCompartments");

		// groups must have the same size
		UserErrors::assertTrue(getGroupNumNeurons(grpIdLower) == getGroupNumNeurons(grpIdUpper),
			UserErrors::MUST_BE_IDENTICAL, funcName.str(), "Sizes of " + grpIdLowerStr.str() + " and " +
			grpIdUpperStr.str());

		// groups must be located on the same partition
		UserErrors::assertTrue(groupPrefNetIds_.at(grpIdLower) == groupPrefNetIds_.at(grpIdUpper),
			UserErrors::MUST_BE_IDENTICAL, funcName.str(), "Preferred partions of " + grpIdLowerStr.str() + " and " +
			grpIdUpperStr.str());

		// groups cannot be both chemically (synaptically) and electrically (compartmentally) connected
		UserErrors::assertTrue(std::find(connSyn_[grpIdLower].begin(), connSyn_[grpIdLower].end(), grpIdUpper) ==
			connSyn_[grpIdLower].end(), UserErrors::CANNOT_BE_CONN_SYN_AND_COMP, funcName.str(),
			grpIdLowerStr.str() + " and " + grpIdUpperStr.str());
		UserErrors::assertTrue(std::find(connSyn_[grpIdUpper].begin(), connSyn_[grpIdUpper].end(), grpIdLower) ==
			connSyn_[grpIdUpper].end(), UserErrors::CANNOT_BE_CONN_SYN_AND_COMP, funcName.str(),
			grpIdLowerStr.str() + " and " + grpIdUpperStr.str());

		// groups cannot be connected twice (order doesn't matter)
		UserErrors::assertTrue(std::find(connComp_[grpIdLower].begin(), connComp_[grpIdLower].end(), grpIdUpper) ==
			connComp_[grpIdLower].end(), UserErrors::CANNOT_BE_CONN_TWICE, funcName.str(),
			grpIdLowerStr.str() + " and " + grpIdUpperStr.str());
		UserErrors::assertTrue(std::find(connComp_[grpIdUpper].begin(), connComp_[grpIdUpper].end(), grpIdLower) ==
			connComp_[grpIdUpper].end(), UserErrors::CANNOT_BE_CONN_TWICE, funcName.str(),
			grpIdLowerStr.str() + " and " + grpIdUpperStr.str());

		// groups can have at most getMaxNumCompConnections() connections
		UserErrors::assertTrue(connComp_[grpIdLower].size() < getMaxNumCompConnections(), UserErrors::MUST_BE_IN_RANGE,
			funcName.str(), "Number of compartmental connections for group " + grpIdLowerStr.str(),
			"[0,getMaxNumCompConnections()");
		UserErrors::assertTrue(connComp_[grpIdUpper].size() < getMaxNumCompConnections(), UserErrors::MUST_BE_IN_RANGE,
			funcName.str(), "Number of compartmental connections for group " + grpIdUpperStr.str(),
			"[0,getMaxNumCompConnections()");

		// add compartment connection to 2D matrix (both ways)
		connComp_[grpIdLower].push_back(grpIdUpper);
		connComp_[grpIdUpper].push_back(grpIdLower);

		return snn_->connectCompartments(grpIdLower, grpIdUpper);
	}

	// create group of Izhikevich spiking neurons on 1D grid
	int createGroup(const std::string& grpName, int nNeur, int neurType, int preferredPartition, ComputingBackend preferredBackend) {
		return createGroup(grpName, Grid3D(nNeur,1,1), neurType, preferredPartition, preferredBackend);
	}

	// create group of LIF spiking neurons on 1D grid
	int createGroupLIF(const std::string& grpName, int nNeur, int neurType, int preferredPartition = ANY, ComputingBackend preferredBackend = CPU_CORES){
		return createGroupLIF(grpName, Grid3D(nNeur,1,1), neurType, preferredPartition, preferredBackend);
	}

	// create a group of Izhikevich spiking neurons on 3D grid
	int createGroup(const std::string& grpName, const Grid3D& grid, int neurType, int preferredPartition, ComputingBackend preferredBackend) {
		std::string funcName = "createGroup(\""+grpName+"\")";
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, 
			funcName, "CONFIG.");
		UserErrors::assertTrue(grid.numX>0, UserErrors::CANNOT_BE_NEGATIVE, funcName, "grid.numX");
		UserErrors::assertTrue(grid.numY>0, UserErrors::CANNOT_BE_NEGATIVE, funcName, "grid.numY");
		UserErrors::assertTrue(grid.numZ>0, UserErrors::CANNOT_BE_NEGATIVE, funcName, "grid.numZ");

		// if user has called any set functions with grpId=ALL, and is now adding another group, previously set properties
		// will not apply to newly added group
		if (hasSetSTPALL_)
			userWarnings_.push_back("Make sure to call setSTP on group "+grpName);
		if (hasSetSTDPALL_)
			userWarnings_.push_back("Make sure to call setSTDP on group "+grpName);
		if (hasSetHomeoALL_)
			userWarnings_.push_back("Make sure to call setHomeostasis on group "+grpName);
		if (hasSetHomeoBaseFiringALL_)
			userWarnings_.push_back("Make sure to call setHomeoBaseFiringRate on group "+grpName);

		int grpId = snn_->createGroup(grpName.c_str(), grid, neurType, preferredPartition, preferredBackend);
		grpIds_.push_back(grpId); // keep track of all groups

		int partitionOffset = 0;
		if (preferredBackend == CPU_CORES)
			partitionOffset = MAX_NUM_CUDA_DEVICES;
		else if (preferredBackend == GPU_CORES)
			partitionOffset = 0;
		int prefPartition = preferredPartition + partitionOffset;
		groupPrefNetIds_.insert(std::pair<int, int>(grpId, prefPartition));

		// extend 2D connection matrices to number of groups
		connSyn_.resize(grpIds_.size());
		connComp_.resize(grpIds_.size());

		return grpId;
	}

	// create a group of LIF spiking neurons on 3D grid
	int createGroupLIF(const std::string& grpName, const Grid3D& grid, int neurType, int preferredPartition, ComputingBackend preferredBackend) {
		std::string funcName = "createGroupLIF(\""+grpName+"\")";
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, 
			funcName, "CONFIG.");
		UserErrors::assertTrue(grid.numX>0, UserErrors::CANNOT_BE_NEGATIVE, funcName, "grid.numX");
		UserErrors::assertTrue(grid.numY>0, UserErrors::CANNOT_BE_NEGATIVE, funcName, "grid.numY");
		UserErrors::assertTrue(grid.numZ>0, UserErrors::CANNOT_BE_NEGATIVE, funcName, "grid.numZ");

		// if user has called any set functions with grpId=ALL, and is now adding another group, previously set properties
		// will not apply to newly added group
		if (hasSetSTPALL_)
			userWarnings_.push_back("Make sure to call setSTP on group "+grpName);
		if (hasSetSTDPALL_)
			userWarnings_.push_back("Make sure to call setSTDP on group "+grpName);
		if (hasSetHomeoALL_)
			userWarnings_.push_back("Make sure to call setHomeostasis on group "+grpName);
		if (hasSetHomeoBaseFiringALL_)
			userWarnings_.push_back("Make sure to call setHomeoBaseFiringRate on group "+grpName);

		int grpId = snn_->createGroupLIF(grpName.c_str(), grid, neurType, preferredPartition, preferredBackend);
		grpIds_.push_back(grpId); // keep track of all groups

		int partitionOffset = 0;
		if (preferredBackend == CPU_CORES)
			partitionOffset = MAX_NUM_CUDA_DEVICES;
		else if (preferredBackend == GPU_CORES)
			partitionOffset = 0;
		int prefPartition = preferredPartition + partitionOffset;
		groupPrefNetIds_.insert(std::pair<int, int>(grpId, prefPartition));

		// extend 2D connection matrices to number of groups
		connSyn_.resize(grpIds_.size());
		connComp_.resize(grpIds_.size());

		return grpId;
	}

	// create group of spike generators on 1D grid
	int createSpikeGeneratorGroup(const std::string& grpName, int nNeur, int neurType, int preferredPartition, ComputingBackend preferredBackend) {
		return createSpikeGeneratorGroup(grpName, Grid3D(nNeur,1,1), neurType, preferredPartition, preferredBackend);
	}

	// create group of spike generators on 3D grid
	int createSpikeGeneratorGroup(const std::string& grpName, const Grid3D& grid, int neurType, int preferredPartition, ComputingBackend preferredBackend) {
		std::string funcName = "createSpikeGeneratorGroup(\""+grpName+"\")";
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, 
			funcName, "CONFIG.");
		UserErrors::assertTrue(grid.numX>0, UserErrors::CANNOT_BE_NEGATIVE, funcName, "grid.numX");
		UserErrors::assertTrue(grid.numY>0, UserErrors::CANNOT_BE_NEGATIVE, funcName, "grid.numY");
		UserErrors::assertTrue(grid.numZ>0, UserErrors::CANNOT_BE_NEGATIVE, funcName, "grid.numZ");

		int grpId = snn_->createSpikeGeneratorGroup(grpName.c_str(),grid,neurType, preferredPartition, preferredBackend);
		grpIds_.push_back(grpId); // keep track of all groups

		// extend 2D connection matrices to number of groups
		connSyn_.resize(grpIds_.size());
		connComp_.resize(grpIds_.size());

		return grpId;
	}

	void setCompartmentParameters(int grpId, float couplingUp, float couplingDown) {
		std::string funcName = "setCompartmentParameters(\"" + getGroupName(grpId) + "\")";
		UserErrors::assertTrue(carlsimState_ == CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName,
			"CONFIG.");

		snn_->setCompartmentParameters(grpId, couplingUp, couplingDown);
	}


	// set conductance values, use defaults
	void setConductances(bool isSet) {
		std::stringstream funcName; funcName << "setConductances(" << isSet << ")";
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName.str(),
			funcName.str(), "CONFIG.");
		hasSetConductances_ = true;

		if (isSet) { // enable conductances, use default values
			snn_->setConductances(true,def_tdAMPA_,0,def_tdNMDA_,def_tdGABAa_,0,def_tdGABAb_);
		} else { // disable conductances
			snn_->setConductances(false,0,0,0,0,0,0);
		}
	}

	// set conductances values, CUSTOM
	void setConductances(bool isSet, int tdAMPA, int tdNMDA, int tdGABAa, int tdGABAb) {
		std::stringstream funcName; funcName << "setConductances(" << isSet << "," << tdAMPA << "," << tdNMDA << ","
			<< tdGABAa << "," << tdGABAb << ")";
		UserErrors::assertTrue(!isSet||tdAMPA>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "tdAMPA");
		UserErrors::assertTrue(!isSet||tdNMDA>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "tdNMDA");
		UserErrors::assertTrue(!isSet||tdGABAa>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "tdGABAa");
		UserErrors::assertTrue(!isSet||tdGABAb>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "trGABAb");
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName.str(),
			funcName.str(),"CONFIG.");
		hasSetConductances_ = true;

		if (isSet) { // enable conductances, use custom values
			snn_->setConductances(true,tdAMPA,0,tdNMDA,tdGABAa,0,tdGABAb);
		} else { // disable conductances
			snn_->setConductances(false,0,0,0,0,0,0);
		}
	}

	// set conductances values, custom
	void setConductances(bool isSet, int tdAMPA, int trNMDA, int tdNMDA, int tdGABAa, int trGABAb, 
		int tdGABAb)
	{
		std::stringstream funcName; funcName << "setConductances(" << isSet << "," << tdAMPA << "," << trNMDA << "," <<
			tdNMDA << "," << tdGABAa << "," << trGABAb << "," << tdGABAb << ")";
		UserErrors::assertTrue(!isSet||tdAMPA>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "tdAMPA");
		UserErrors::assertTrue(!isSet||trNMDA>=0, UserErrors::CANNOT_BE_NEGATIVE, funcName.str(), "trNMDA");
		UserErrors::assertTrue(!isSet||tdNMDA>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "tdNMDA");
		UserErrors::assertTrue(!isSet||tdGABAa>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "tdGABAa");
		UserErrors::assertTrue(!isSet||trGABAb>=0, UserErrors::CANNOT_BE_NEGATIVE, funcName.str(), "trGABAb");
		UserErrors::assertTrue(!isSet||tdGABAb>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "trGABAb");
		UserErrors::assertTrue(trNMDA!=tdNMDA, UserErrors::CANNOT_BE_IDENTICAL, funcName.str(), "trNMDA and tdNMDA");
		UserErrors::assertTrue(trGABAb!=tdGABAb, UserErrors::CANNOT_BE_IDENTICAL, funcName.str(), 
			"trGABAb and tdGABAb");
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName.str(),
			funcName.str(), "CONFIG.");
		hasSetConductances_ = true;

		if (isSet) { // enable conductances, use custom values
			snn_->setConductances(true,tdAMPA,trNMDA,tdNMDA,tdGABAa,trGABAb,tdGABAb);
		} else { // disable conductances
			snn_->setConductances(false,0,0,0,0,0,0);
		}
	}

	// set default homeostasis params
	void setHomeostasis(int grpId, bool isSet) {
		std::string funcName = "setHomeostasis(\""+getGroupName(grpId)+"\")";
		UserErrors::assertTrue(!isSet || isSet && !isPoissonGroup(grpId), UserErrors::WRONG_NEURON_TYPE, funcName, 
			funcName);
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, 
			funcName, "CONFIG.");

		hasSetHomeoALL_ = grpId==ALL; // adding groups after this will not have homeostasis set

		if (isSet) { // enable homeostasis, use default values
			snn_->setHomeostasis(grpId,true,def_homeo_scale_,def_homeo_avgTimeScale_);
			if (grpId!=ALL && hasSetHomeoBaseFiringALL_)
				userWarnings_.push_back("Make sure to call setHomeoBaseFiringRate on group "
					+ getGroupName(grpId));
		} else { // disable conductances
			snn_->setHomeostasis(grpId,false,0.0f,0.0f);
		}
	}

	// set custom homeostasis params for group
	void setHomeostasis(int grpId, bool isSet, float homeoScale, float avgTimeScale) {
		std::string funcName = "setHomeostasis(\""+getGroupName(grpId)+"\")";
		UserErrors::assertTrue(!isSet || isSet && !isPoissonGroup(grpId), UserErrors::WRONG_NEURON_TYPE, funcName,
			funcName);
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName,
			"CONFIG.");

		hasSetHomeoALL_ = grpId==ALL; // adding groups after this will not have homeostasis set

		if (isSet) { // enable homeostasis, use default values
			snn_->setHomeostasis(grpId,true,homeoScale,avgTimeScale);
			if (grpId!=ALL && hasSetHomeoBaseFiringALL_)
				userWarnings_.push_back("Make sure to call setHomeoBaseFiringRate on group "
					+ getGroupName(grpId));
		} else { // disable conductances
			snn_->setHomeostasis(grpId,false,0.0f,0.0f);
		}
	}

	// set a homeostatic target firing rate (enforced through homeostatic synaptic scaling)
	void setHomeoBaseFiringRate(int grpId, float baseFiring, float baseFiringSD) {
		std::string funcName = "setHomeoBaseFiringRate(\""+getGroupName(grpId)+"\")";
		UserErrors::assertTrue(!isPoissonGroup(grpId), UserErrors::WRONG_NEURON_TYPE, funcName, funcName);
		UserErrors::assertTrue(isGroupWithHomeostasis(grpId), UserErrors::WRONG_NEURON_TYPE, funcName,
			funcName, " Must call setHomeostasis first.");
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName,
			"CONFIG.");

		hasSetHomeoBaseFiringALL_ = grpId==ALL; // adding groups after this will not have base firing set

		snn_->setHomeoBaseFiringRate(grpId, baseFiring, baseFiringSD);
	}

	// sets integration method (FORWARD_EULER, RUNGE_KUTTA4, etc.) and integration step
	void setIntegrationMethod(integrationMethod_t method, int numStepsPerMs) {
		std::string funcName = "setIntegrationMethod()";
		UserErrors::assertTrue(carlsimState_ == CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName,
			"CONFIG.");
		UserErrors::assertTrue((numStepsPerMs >= 1) && (numStepsPerMs <= 100), UserErrors::MUST_BE_IN_RANGE, funcName,
			"numStepsPerMs", "[1, 100]");

		snn_->setIntegrationMethod(method, numStepsPerMs);

		//std::cout << "numStepsPerMs is (in interface): " + numStepsPerMs << std::endl;
	}

	// set neuron parameters for Izhikevich neuron, with standard deviations
	void setNeuronParameters(int grpId, float izh_a, float izh_a_sd, float izh_b, float izh_b_sd,
		float izh_c, float izh_c_sd, float izh_d, float izh_d_sd)
	{
		std::string funcName = "setNeuronParameters(\""+getGroupName(grpId)+"\")";
		UserErrors::assertTrue(!isPoissonGroup(grpId), UserErrors::WRONG_NEURON_TYPE, funcName, funcName);
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName,
			"CONFIG.");

		// wrapper identical to core func
		snn_->setNeuronParameters(grpId, izh_a, izh_a_sd, izh_b, izh_b_sd, izh_c, izh_c_sd, izh_d, izh_d_sd);
	}

	// set neuron parameters for Izhikevich neuron
	void setNeuronParameters(int grpId, float izh_a, float izh_b, float izh_c, float izh_d) {
		std::string funcName = "setNeuronParameters(\""+getGroupName(grpId)+"\")";
		UserErrors::assertTrue(!isPoissonGroup(grpId), UserErrors::WRONG_NEURON_TYPE, funcName, funcName);
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, 
			funcName, "CONFIG.");

		// set standard deviations of Izzy params to zero
		snn_->setNeuronParameters(grpId, izh_a, 0.0f, izh_b, 0.0f, izh_c, 0.0f, izh_d, 0.0f);
	}

	void setNeuronParameters(int grpId, float izh_C, float izh_k, float izh_vr, float izh_vt,
		float izh_a, float izh_b, float izh_vpeak, float izh_c, float izh_d)
	{
		std::string funcName = "setNeuronParameters(\"" + getGroupName(grpId) + "\")";
		UserErrors::assertTrue(!isPoissonGroup(grpId), UserErrors::WRONG_NEURON_TYPE, funcName, funcName);
		UserErrors::assertTrue(carlsimState_ == CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");
		UserErrors::assertTrue(izh_C > 0, UserErrors::CANNOT_BE_NEGATIVE, funcName, "izh_C");

		// set standard deviations of Izzy params to zero
		snn_->setNeuronParameters(grpId, izh_C, 0.0f, izh_k, 0.0f, izh_vr, 0.0f, izh_vt, 0.0f,
			izh_a, 0.0f, izh_b, 0.0f, izh_vpeak, 0.0f, izh_c, 0.0f, izh_d, 0.0f);
	}


	void setNeuronParameters(int grpId, float izh_C, float izh_C_sd, float izh_k, float izh_k_sd,
		float izh_vr, float izh_vr_sd, float izh_vt, float izh_vt_sd,
		float izh_a, float izh_a_sd, float izh_b, float izh_b_sd,
		float izh_vpeak, float izh_vpeak_sd, float izh_c, float izh_c_sd,
		float izh_d, float izh_d_sd)
	{
		std::string funcName = "setNeuronParameters(\"" + getGroupName(grpId) + "\")";
		UserErrors::assertTrue(!isPoissonGroup(grpId), UserErrors::WRONG_NEURON_TYPE, funcName, funcName);
		UserErrors::assertTrue(carlsimState_ == CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");
		UserErrors::assertTrue(izh_C > 0, UserErrors::CANNOT_BE_NEGATIVE, funcName, "izh_C");

		// wrapper identical to core func
		snn_->setNeuronParameters(grpId, izh_C, izh_C_sd, izh_k, izh_k_sd, izh_vr, izh_vr_sd, izh_vt, izh_vt_sd,
			izh_a, izh_a_sd, izh_b, izh_b_sd, izh_vpeak, izh_vpeak_sd, izh_c, izh_c_sd, izh_d, izh_d_sd);
	}

	// set neuron parameters for LIF spiking neuron
	void setNeuronParametersLIF(int grpId, int tau_m, int tau_ref, float vTh, float vReset, const RangeRmem& rMem)
	{
		std::string funcName = "setNeuronParametersLIF(\"" + getGroupName(grpId) + "\")";
		UserErrors::assertTrue(!isPoissonGroup(grpId), UserErrors::WRONG_NEURON_TYPE, funcName, funcName);
		UserErrors::assertTrue(carlsimState_ == CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");

		UserErrors::assertTrue(tau_m >= 0 , UserErrors::CANNOT_BE_NEGATIVE, funcName, "tau_m");
		UserErrors::assertTrue(tau_ref >= 0 , UserErrors::CANNOT_BE_NEGATIVE, funcName, "tau_ref");

		UserErrors::assertTrue(vReset < vTh , UserErrors::CANNOT_BE_LARGER, funcName, "vReset");

		UserErrors::assertTrue(rMem.minRmem >= 0.0f , UserErrors::CANNOT_BE_NEGATIVE, funcName, "rangeRmem.minRmem");
		UserErrors::assertTrue(rMem.minRmem <= rMem.maxRmem , UserErrors::CANNOT_BE_LARGER, funcName, "rangeRmem.minRmem");

		// wrapper identical to core func
		snn_->setNeuronParametersLIF(grpId, tau_m, tau_ref, vTh, vReset,rMem.minRmem, rMem.maxRmem);
	}

	// set parameters for each neuronmodulator
	void setNeuromodulator(int grpId, float baseDP, float tauDP, float base5HT, float tau5HT, float baseACh, 
		float tauACh, float baseNE, float tauNE)
	{
		std::string funcName = "setNeuromodulator(\""+getGroupName(grpId)+"\")";
		UserErrors::assertTrue(baseDP > 0, UserErrors::MUST_BE_POSITIVE, funcName);
		UserErrors::assertTrue(tauDP > 0, UserErrors::MUST_BE_POSITIVE, funcName);
		UserErrors::assertTrue(base5HT > 0, UserErrors::MUST_BE_POSITIVE, funcName);
		UserErrors::assertTrue(tau5HT > 0, UserErrors::MUST_BE_POSITIVE, funcName);
		UserErrors::assertTrue(baseACh > 0, UserErrors::MUST_BE_POSITIVE, funcName);
		UserErrors::assertTrue(tauACh > 0, UserErrors::MUST_BE_POSITIVE, funcName);
		UserErrors::assertTrue(baseNE > 0, UserErrors::MUST_BE_POSITIVE, funcName);
		UserErrors::assertTrue(tauNE > 0, UserErrors::MUST_BE_POSITIVE, funcName);
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, 
			funcName, "CONFIG.");

		snn_->setNeuromodulator(grpId, baseDP, tauDP, base5HT, tau5HT, baseACh, tauACh, baseNE, tauNE);
	}

	void setNeuromodulator(int grpId,float tauDP, float tau5HT, float tauACh, float tauNE) {
		std::string funcName = "setNeuromodulator(\""+getGroupName(grpId)+"\")";
		UserErrors::assertTrue(tauDP > 0, UserErrors::MUST_BE_POSITIVE, funcName);
		UserErrors::assertTrue(tau5HT > 0, UserErrors::MUST_BE_POSITIVE, funcName);
		UserErrors::assertTrue(tauACh > 0, UserErrors::MUST_BE_POSITIVE, funcName);
		UserErrors::assertTrue(tauNE > 0, UserErrors::MUST_BE_POSITIVE, funcName);
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, 
			funcName, "CONFIG.");

		snn_->setNeuromodulator(grpId, 1.0f, tauDP, 1.0f, tau5HT, 1.0f, tauACh, 1.0f, tauNE);
	}

	// set STDP, default, wrapper function
	void setSTDP(int grpId, bool isSet) {
		setESTDP(grpId, isSet);
	}

	// set STDP, custom, wrapper function
	void setSTDP(int grpId, bool isSet, STDPType type, float alphaPlus, float tauPlus, float alphaMinus, 
		float tauMinus)
	{
		setESTDP(grpId, isSet, type, ExpCurve(alphaPlus, tauPlus, alphaMinus, tauMinus));
	}

	// set ESTDP, default
	void setESTDP(int grpId, bool isSet) {
		std::string funcName = "setESTDP(\""+getGroupName(grpId)+"\")";
		UserErrors::assertTrue(!isSet || isSet && !isPoissonGroup(grpId), UserErrors::WRONG_NEURON_TYPE, funcName, 
			funcName);
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, 
			funcName, "CONFIG.");

		hasSetSTDPALL_ = grpId==ALL; // adding groups after this will not have conductances set

		if (isSet) { // enable STDP, use default values and type
			snn_->setESTDP(grpId, true, def_STDP_type_, EXP_CURVE, def_STDP_alphaLTP_, def_STDP_tauLTP_, 
				def_STDP_alphaLTD_, def_STDP_tauLTD_, 0.0f);
		} else { // disable STDP
			snn_->setESTDP(grpId, false, UNKNOWN_STDP, UNKNOWN_CURVE, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f);
		}
	}

	// set ESTDP by stdp curve
	void setESTDP(int grpId, bool isSet, STDPType type, ExpCurve curve) {
		std::string funcName = "setESTDP(\""+getGroupName(grpId)+","+stdpType_string[type]+"\")";
		UserErrors::assertTrue(!isSet || isSet && !isPoissonGroup(grpId), UserErrors::WRONG_NEURON_TYPE, funcName, 
			funcName);
		UserErrors::assertTrue(type!=UNKNOWN_STDP, UserErrors::CANNOT_BE_UNKNOWN, funcName, "Mode");
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, 
			funcName, "CONFIG.");

		hasSetSTDPALL_ = grpId==ALL; // adding groups after this will not have conductances set

		if (isSet) { // enable STDP, use custom values
			snn_->setESTDP(grpId, true, type, curve.stdpCurve, curve.alphaPlus, curve.tauPlus, curve.alphaMinus, 
				curve.tauMinus, 0.0f);
		} else { // disable STDP and DA-STDP as well
			snn_->setESTDP(grpId, false, UNKNOWN_STDP, UNKNOWN_CURVE, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f);
		}
	}

	// set ESTDP by stdp curve
	void setESTDP(int grpId, bool isSet, STDPType type, TimingBasedCurve curve) {
		std::string funcName = "setESTDP(\""+getGroupName(grpId)+","+stdpType_string[type]+"\")";
		UserErrors::assertTrue(!isSet || isSet && !isPoissonGroup(grpId), UserErrors::WRONG_NEURON_TYPE, funcName, 
			funcName);
		UserErrors::assertTrue(type!=UNKNOWN_STDP, UserErrors::CANNOT_BE_UNKNOWN, funcName, "Mode");
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, 
			funcName, "CONFIG.");

		hasSetSTDPALL_ = grpId==ALL; // adding groups after this will not have conductances set

		if (isSet) { // enable STDP, use custom values
			snn_->setESTDP(grpId, true, type, curve.stdpCurve, curve.alphaPlus, curve.tauPlus, curve.alphaMinus, 
				curve.tauMinus, curve.gamma);
		} else { // disable STDP and DA-STDP as well
			snn_->setESTDP(grpId, false, UNKNOWN_STDP, UNKNOWN_CURVE, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f);
		}
	}

	// set ISTDP, default
	void setISTDP(int grpId, bool isSet) {
		std::string funcName = "setISTDP(\""+getGroupName(grpId)+"\")";
		UserErrors::assertTrue(!isSet || isSet && !isPoissonGroup(grpId), UserErrors::WRONG_NEURON_TYPE, funcName, 
			funcName);
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, 
			funcName, "CONFIG.");

		hasSetSTDPALL_ = grpId==ALL; // adding groups after this will not have conductances set

		if (isSet) { // enable STDP, use default values and types
			snn_->setISTDP(grpId, true, def_STDP_type_, PULSE_CURVE, def_STDP_betaLTP_, def_STDP_betaLTD_, 
				def_STDP_lambda_, def_STDP_delta_);
		} else { // disable STDP
			snn_->setISTDP(grpId, false, UNKNOWN_STDP, UNKNOWN_CURVE, 0.0f, 0.0f, 1.0f, 1.0f);
		}
	}

	// set ISTDP by stdp curve
	void setISTDP(int grpId, bool isSet, STDPType type, ExpCurve curve) {
		std::string funcName = "setISTDP(\""+getGroupName(grpId)+","+stdpType_string[type]+"\")";
		UserErrors::assertTrue(!isSet || isSet && !isPoissonGroup(grpId), UserErrors::WRONG_NEURON_TYPE, funcName, 
			funcName);
		UserErrors::assertTrue(type!=UNKNOWN_STDP, UserErrors::CANNOT_BE_UNKNOWN, funcName, "Mode");
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, 
			funcName, "CONFIG.");

		hasSetSTDPALL_ = grpId==ALL; // adding groups after this will not have conductances set

		if (isSet) { // enable STDP, use custom values
			snn_->setISTDP(grpId, true, type, curve.stdpCurve, curve.alphaPlus, curve.alphaMinus, curve.tauPlus, 
				curve.tauMinus);
		} else { // disable STDP and DA-STDP as well
			snn_->setISTDP(grpId, false, UNKNOWN_STDP, UNKNOWN_CURVE, 0.0f, 0.0f, 1.0f, 1.0f);
		}
	}

	// set ISTDP by stdp curve
	void setISTDP(int grpId, bool isSet, STDPType type, PulseCurve curve) {
		std::string funcName = "setISTDP(\""+getGroupName(grpId)+","+stdpType_string[type]+"\")";
		UserErrors::assertTrue(!isSet || isSet && !isPoissonGroup(grpId), UserErrors::WRONG_NEURON_TYPE, funcName, 
			funcName);
		UserErrors::assertTrue(type!=UNKNOWN_STDP, UserErrors::CANNOT_BE_UNKNOWN, funcName, "Mode");
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, 
			funcName, "CONFIG.");

		hasSetSTDPALL_ = grpId==ALL; // adding groups after this will not have conductances set

		if (isSet) { // enable STDP, use custom values
			snn_->setISTDP(grpId, true, type, curve.stdpCurve, curve.betaLTP, curve.betaLTD, curve.lambda, curve.delta);
		} else { // disable STDP and DA-STDP as well
			snn_->setISTDP(grpId, false, UNKNOWN_STDP, UNKNOWN_CURVE, 0.0f, 0.0f, 1.0f, 1.0f);
		}
	}

	// set STP, default
	void setSTP(int grpId, bool isSet) {
		std::string funcName = "setSTP(\""+getGroupName(grpId)+"\")";
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, 
			funcName, "CONFIG.");

		hasSetSTPALL_ = grpId==ALL; // adding groups after this will not have conductances set

		if (isSet) { // enable STDP, use default values
			UserErrors::assertTrue(isExcitatoryGroup(grpId) || isInhibitoryGroup(grpId), UserErrors::WRONG_NEURON_TYPE,
				funcName, "setSTP");

			if (isExcitatoryGroup(grpId))
				snn_->setSTP(grpId,true,def_STP_U_exc_,def_STP_tau_u_exc_,def_STP_tau_x_exc_);
			else if (isInhibitoryGroup(grpId))
				snn_->setSTP(grpId,true,def_STP_U_inh_,def_STP_tau_u_inh_,def_STP_tau_x_inh_);
			else {
				// some error message
			}
		} else { // disable STDP
			snn_->setSTP(grpId,false,0.0f,0.0f,0.0f);
		}
	}

	// set STP, custom
	void setSTP(int grpId, bool isSet, float STP_U, float STP_tau_u, float STP_tau_x) {
		std::string funcName = "setSTP(\""+getGroupName(grpId)+"\")";
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, 
			funcName, "CONFIG.");

		hasSetSTPALL_ = grpId==ALL; // adding groups after this will not have conductances set

		if (isSet) { // enable STDP, use default values
			UserErrors::assertTrue(isExcitatoryGroup(grpId) || isInhibitoryGroup(grpId), UserErrors::WRONG_NEURON_TYPE,
				funcName,"setSTP");

			snn_->setSTP(grpId,true,STP_U,STP_tau_u,STP_tau_x);
		} else { // disable STDP
			snn_->setSTP(grpId,false,0.0f,0.0f,0.0f);
		}
	}

	void setWeightAndWeightChangeUpdate(UpdateInterval wtANDwtChangeUpdateInterval, bool enableWtChangeDecay,
		float wtChangeDecay)
	{
		std::string funcName = "setWeightAndWeightChangeUpdate()";
		UserErrors::assertTrue(wtChangeDecay > 0.0f, UserErrors::MUST_BE_POSITIVE, funcName);
		UserErrors::assertTrue(wtChangeDecay < 1.0f, UserErrors::CANNOT_BE_LARGER, funcName);
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, 
			funcName, "CONFIG.");

		snn_->setWeightAndWeightChangeUpdate(wtANDwtChangeUpdateInterval, enableWtChangeDecay, wtChangeDecay);
	}


	// +++++++++ PUBLIC METHODS: RUNNING A SIMULATION +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	// run network with custom options
	int runNetwork(int nSec, int nMsec, bool printRunSummary) {
		std::string funcName = "runNetwork()";
		UserErrors::assertTrue(carlsimState_ == SETUP_STATE || carlsimState_ == RUN_STATE,
				UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "SETUP or RUN.");

		// run some checks before running network for the first time
		if (carlsimState_ != RUN_STATE) {
			// if user hasn't called setConductances, set to false and disp warning
			if (!hasSetConductances_) {
				userWarnings_.push_back("setConductances has not been called. Setting simulation mode to CUBA.");
			}

			// make sure user didn't provoque any user warnings
			handleUserWarnings();
		}

		carlsimState_ = RUN_STATE;

		return snn_->runNetwork(nSec, nMsec, printRunSummary);
	}

	// setup network with custom options
	void setupNetwork() {
		std::string funcName = "setupNetwork()";
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, 
			funcName, "CONFIG.");

		carlsimState_ = SETUP_STATE;
		snn_->setupNetwork();
	}


	// +++++++++ PUBLIC METHODS: LOGGING / PLOTTING +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	const FILE* getLogFpInf() { return snn_->getLogFpInf(); }
	const FILE* getLogFpErr() { return snn_->getLogFpErr(); }
	const FILE* getLogFpDeb() { return snn_->getLogFpDeb(); }
	const FILE* getLogFpLog() { return snn_->getLogFpLog(); }

	void saveSimulation(const std::string& fileName, bool saveSynapseInfo) {
		FILE* fpSave = fopen(fileName.c_str(),"wb");
		std::string funcName = "saveSimulation()";
		UserErrors::assertTrue(fpSave!=NULL,UserErrors::FILE_CANNOT_OPEN,fileName);
		UserErrors::assertTrue(carlsimState_ == SETUP_STATE || carlsimState_ == RUN_STATE,
			UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "SETUP or RUN.");

		snn_->saveSimulation(fpSave,saveSynapseInfo);

		fclose(fpSave);
	}

	void setLogFile(const std::string& fileName) {
		std::string funcName = "setLogFile("+fileName+")";
		UserErrors::assertTrue(loggerMode_!=CUSTOM,UserErrors::CANNOT_BE_SET_TO, funcName, "Logger mode", "CUSTOM");

		FILE* fpLog = NULL;
		std::string fileNameNonConst = fileName;
		std::transform(fileNameNonConst.begin(), fileNameNonConst.end(), fileNameNonConst.begin(), ::tolower);
		if (fileNameNonConst=="null") {
#if defined(WIN32) || defined(WIN64)
			fpLog = fopen("nul","w");
#else
			fpLog = fopen("/dev/null","w");
#endif
		} else {
			fpLog = fopen(fileName.c_str(),"w");
		}
		UserErrors::assertTrue(fpLog!=NULL, UserErrors::FILE_CANNOT_OPEN, funcName, fileName);

		// change only CARLsim log file pointer (use NULL as code for leaving the others unchanged)
		snn_->setLogsFp(NULL, NULL, NULL, fpLog);
	}

	// set new file pointer for all files in CUSTOM mode
	void setLogsFpCustom(FILE* fpInf, FILE* fpErr, FILE* fpDeb, FILE* fpLog) {
		UserErrors::assertTrue(loggerMode_==CUSTOM,UserErrors::MUST_BE_SET_TO,"setLogsFpCustom","Logger mode","CUSTOM");

		snn_->setLogsFp(fpInf,fpErr,fpDeb,fpLog);
	}


	// +++++++++ PUBLIC METHODS: INTERACTING WITH A SIMULATION ++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	// adds a constant bias to the weight of every synapse in the connection
	void biasWeights(short int connId, float bias, bool updateWeightRange) {
		std::stringstream funcName;	funcName << "biasWeights(" << connId << "," << bias << "," << updateWeightRange <<
			")";
		UserErrors::assertTrue(carlsimState_==SETUP_STATE || carlsimState_==RUN_STATE,
			UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName.str(), funcName.str(), "SETUP or RUN.");
		UserErrors::assertTrue(connId>=0 && connId<getNumConnections(), UserErrors::MUST_BE_IN_RANGE, funcName.str(),
			"connId", "[0,getNumConnections()]");

		snn_->biasWeights(connId, bias, updateWeightRange);
	}

	void startTesting(bool updateWeights) {
		std::string funcName = "startTesting()";
		UserErrors::assertTrue(carlsimState_==SETUP_STATE || carlsimState_==RUN_STATE, 
			UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "SETUP or RUN.");
		snn_->startTesting(updateWeights);
	}

	void stopTesting() {
		std::string funcName = "stopTesting()";
		UserErrors::assertTrue(carlsimState_==SETUP_STATE || carlsimState_==RUN_STATE, 
			UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "SETUP or RUN.");
		snn_->stopTesting();
	}

	// reads network state from file
	void loadSimulation(FILE* fid) {
		std::string funcName = "loadSimulation()";
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, 
			funcName, "CONFIG.");

		snn_->loadSimulation(fid);
	}

	// scales the weight of every synapse in the connection with a scaling factor
	void scaleWeights(short int connId, float scale, bool updateWeightRange) {
		std::stringstream funcName;	funcName << "scaleWeights(" << connId << "," << scale << "," << updateWeightRange
			<< ")";
		UserErrors::assertTrue(carlsimState_==SETUP_STATE || carlsimState_==RUN_STATE,
			UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName.str(), funcName.str(), "SETUP or RUN.");
		UserErrors::assertTrue(connId>=0 && connId<getNumConnections(), UserErrors::MUST_BE_IN_RANGE, funcName.str(),
			"connId", "[0,getNumConnections()]");
		UserErrors::assertTrue(scale>=0.0f, UserErrors::CANNOT_BE_NEGATIVE, funcName.str(), "Scaling factor");

		snn_->scaleWeights(connId, scale, updateWeightRange);
	}

	// set spike monitor for group and write spikes to file
	ConnectionMonitor* setConnectionMonitor(int grpIdPre, int grpIdPost, const std::string& fname) {
		std::string funcName = "setConnectionMonitor(\"" + getGroupName(grpIdPre) + "\",\"" + getGroupName(grpIdPost)
			+ "\",\"" + fname + "\")";
		UserErrors::assertTrue(grpIdPre!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "grpIdPre");
		UserErrors::assertTrue(grpIdPost!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "grpIdPost");
		UserErrors::assertTrue(grpIdPre>=0, UserErrors::CANNOT_BE_NEGATIVE, funcName, "grpIdPre");
		UserErrors::assertTrue(grpIdPost>=0, UserErrors::CANNOT_BE_NEGATIVE, funcName, "grpIdPost");
		UserErrors::assertTrue(carlsimState_==SETUP_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName,
			funcName, "SETUP.");

		FILE* fid;
		std::string fileName = fname;
		std::transform(fileName.begin(), fileName.end(), fileName.begin(), ::tolower);
		if (fileName  == "null") {
			// user does not want a binary file created
			fid = NULL;
		} else {
			// try to open spike file
			if (fileName == "default") {
				fileName = "results/conn_" + snn_->getGroupName(grpIdPre) + "_" + snn_->getGroupName(grpIdPost) + ".dat";
			} else {
				fileName = fname;
			}

			fid = fopen(fileName.c_str(),"wb");
			if (fid == NULL) {
				// file could not be opened
				// default case: print error and exit
				std::string fileError = " Double-check file permissions and make sure directory exists.";
				UserErrors::assertTrue(false, UserErrors::FILE_CANNOT_OPEN, funcName, fileName, fileError);
			}
		}

	// return ConnectionMonitor object
	return snn_->setConnectionMonitor(grpIdPre, grpIdPost, fid);
}

	void setExternalCurrent(int grpId, const std::vector<float>& current) {
		std::string funcName = "setExternalCurrent(\""+getGroupName(grpId)+"\")";
		UserErrors::assertTrue(grpId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "grpId");
		UserErrors::assertTrue(current.size()==getGroupNumNeurons(grpId), UserErrors::MUST_BE_IDENTICAL, funcName,
			"current.size()", "number of neurons in the group.");
		UserErrors::assertTrue(!isPoissonGroup(grpId), UserErrors::WRONG_NEURON_TYPE, funcName, funcName);
		UserErrors::assertTrue(carlsimState_==SETUP_STATE || carlsimState_==RUN_STATE,
			UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "SETUP or RUN.");

		snn_->setExternalCurrent(grpId, current);
	}

	void setExternalCurrent(int grpId, float current) {
		std::string funcName = "setExternalCurrent(\""+getGroupName(grpId)+"\")";
		UserErrors::assertTrue(grpId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "grpId");
		UserErrors::assertTrue(!isPoissonGroup(grpId), UserErrors::WRONG_NEURON_TYPE, funcName, funcName);
		UserErrors::assertTrue(carlsimState_==SETUP_STATE || carlsimState_==RUN_STATE,
			UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "SETUP or RUN.");

		std::vector<float> vecCurrent(getGroupNumNeurons(grpId), current);
		snn_->setExternalCurrent(grpId, vecCurrent);
	}

	// set group monitor for a group
	GroupMonitor* setGroupMonitor(int grpId, const std::string& fname) {
		std::string funcName = "setGroupMonitor(\""+getGroupName(grpId)+"\",\""+fname+"\")";
		UserErrors::assertTrue(grpId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "grpId");		// grpId can't be ALL
		UserErrors::assertTrue(grpId>=0, UserErrors::CANNOT_BE_NEGATIVE, funcName, "grpId"); // grpId can't be negative
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE || carlsimState_==SETUP_STATE,
			UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG or SETUP.");

		FILE* fid;
		std::string fileName = fname;
		std::transform(fileName.begin(), fileName.end(), fileName.begin(), ::tolower);
		if (fileName  == "null") {
			// user does not want a binary file created
			fid = NULL;
		} else {
			// try to open spike file
			if (fileName == "default") {
				fileName = "results/grp_" + snn_->getGroupName(grpId) + ".dat";
			} else {
				fileName = fname;
			}

			fid = fopen(fileName.c_str(),"wb");
			if (fid == NULL) {
				// file could not be opened
				// default case: print error and exit
				std::string fileError = " Double-check file permissions and make sure directory exists.";
				UserErrors::assertTrue(false, UserErrors::FILE_CANNOT_OPEN, funcName, fileName, fileError);
			}
		}

		// return GroupMonitor object
		return snn_->setGroupMonitor(grpId, fid);
	}

	// sets up a spike generator
	void setSpikeGenerator(int grpId, SpikeGenerator* spikeGenFunc) {
		std::string funcName = "setSpikeGenerator(\""+getGroupName(grpId)+"\")";
		UserErrors::assertTrue(grpId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "grpId");  // groupId can't be ALL
		UserErrors::assertTrue(isPoissonGroup(grpId), UserErrors::WRONG_NEURON_TYPE, funcName, funcName);
		UserErrors::assertTrue(spikeGenFunc!=NULL, UserErrors::CANNOT_BE_NULL, funcName);
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE,	UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, 
			funcName, "CONFIG.");

		// SpikeGeneratorCore* SGC = new SpikeGeneratorCore(this, spikeGenFunc);
		SpikeGeneratorCore* SGC = new SpikeGeneratorCore(sim_, spikeGenFunc);
		spkGen_.push_back(SGC);
		snn_->setSpikeGenerator(grpId, SGC);
	}

	// set spike monitor for group and write spikes to file
	SpikeMonitor* setSpikeMonitor(int grpId, const std::string& fileName) {
		std::string funcName = "setSpikeMonitor(\""+getGroupName(grpId)+"\",\""+fileName+"\")";
		UserErrors::assertTrue(grpId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "grpId");		// grpId can't be ALL
		UserErrors::assertTrue(grpId>=0, UserErrors::CANNOT_BE_NEGATIVE, funcName, "grpId"); // grpId can't be negative
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE || carlsimState_==SETUP_STATE, 
			UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG or SETUP.");

		FILE* fid;
		std::string fileNameLower = fileName;
		std::transform(fileNameLower.begin(), fileNameLower.end(), fileNameLower.begin(), ::tolower);
		if (fileNameLower == "null") {
			// user does not want a binary file created
			fid = NULL;
		} else {
			// try to open spike file
			if (fileNameLower == "default") {
				std::string fileNameDefault = "results/spk_" + snn_->getGroupName(grpId) + ".dat";
				fid = fopen(fileNameDefault.c_str(),"wb");
				if (fid==NULL) {
					std::string fileError = " Make sure results/ exists.";
					UserErrors::assertTrue(false, UserErrors::FILE_CANNOT_OPEN, funcName, fileNameDefault, fileError);
				}
			} else {
				fid = fopen(fileName.c_str(),"wb");
				if (fid==NULL) {
					std::string fileError = " Double-check file permissions and make sure directory exists.";
					UserErrors::assertTrue(false, UserErrors::FILE_CANNOT_OPEN, funcName, fileName, fileError);
				}
			}
		}

		// return SpikeMonitor object
		return snn_->setSpikeMonitor(grpId, fid);
	}

	// set neuron monitor for group and write neuron state values (voltage, recovery, and total current values) to file
	NeuronMonitor* setNeuronMonitor(int grpId, const std::string& fileName) {
		std::string funcName = "setNeuronMonitor(\"" + getGroupName(grpId) + "\",\"" + fileName + "\")";
		UserErrors::assertTrue(grpId != ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "grpId");		// grpId can't be ALL
		UserErrors::assertTrue(grpId >= 0, UserErrors::CANNOT_BE_NEGATIVE, funcName, "grpId"); // grpId can't be negative
		UserErrors::assertTrue(carlsimState_ == CONFIG_STATE,
			UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG or SETUP.");

		FILE* fid;
		std::string fileNameLower = fileName;
		std::transform(fileNameLower.begin(), fileNameLower.end(), fileNameLower.begin(), ::tolower);
		if (fileNameLower == "null") {
			// user does not want a binary file created
			fid = NULL;
		}
		else {
			// try to open spike file
			if (fileNameLower == "default") {
				std::string fileNameDefault = "results/nrnstate_" + snn_->getGroupName(grpId) + ".dat";
				fid = fopen(fileNameDefault.c_str(), "wb");
				if (fid == NULL) {
					std::string fileError = " Make sure results/ exists.";
					UserErrors::assertTrue(false, UserErrors::FILE_CANNOT_OPEN, funcName, fileNameDefault, fileError);
				}
			}
			else {
				fid = fopen(fileName.c_str(), "wb");
				if (fid == NULL) {
					std::string fileError = " Double-check file permissions and make sure directory exists.";
					UserErrors::assertTrue(false, UserErrors::FILE_CANNOT_OPEN, funcName, fileName, fileError);
				}
			}
		}

		// return NeuronMonitor object
		return snn_->setNeuronMonitor(grpId, fid);
	}

	// assign spike rate to poisson group
	void setSpikeRate(int grpId, PoissonRate* spikeRate, int refPeriod) {
		std::string funcName = "setSpikeRate()";
		UserErrors::assertTrue(carlsimState_==SETUP_STATE || carlsimState_==RUN_STATE,
			UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "SETUP or RUN.");
		UserErrors::assertTrue(isPoissonGroup(grpId), UserErrors::WRONG_NEURON_TYPE, funcName, funcName);
		UserErrors::assertTrue(spikeRate->getNumNeurons()==getGroupNumNeurons(grpId), UserErrors::MUST_BE_IDENTICAL,
			funcName, "PoissonRate length and the number of neurons in the group");
		// FIXME: make sure spikeRate->isOnGPU() consistent with simulation mode
		//UserErrors::assertTrue(!spikeRate->isOnGPU() || spikeRate->isOnGPU()&&getSimMode()==GPU_MODE,
		//	UserErrors::CAN_ONLY_BE_CALLED_IN_MODE, funcName, "PoissonRate on GPU", "GPU_MODE.");

		snn_->setSpikeRate(grpId, spikeRate, refPeriod);
	}

	void setWeight(short int connId, int neurIdPre, int neurIdPost, float weight, bool updateWeightRange) {
		std::stringstream funcName;	funcName << "setWeight(" << connId << "," << neurIdPre << "," << neurIdPost << ","
			<< updateWeightRange << ")";
		UserErrors::assertTrue(carlsimState_==SETUP_STATE || carlsimState_==RUN_STATE,
			UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName.str(), funcName.str(), "SETUP or RUN.");
		UserErrors::assertTrue(connId>=0 && connId<getNumConnections(), UserErrors::MUST_BE_IN_RANGE,
			funcName.str(), "connectionId", "[0,getNumConnections()]");
		UserErrors::assertTrue(weight>=0.0f, UserErrors::CANNOT_BE_NEGATIVE, funcName.str(), "Weight value");

		snn_->setWeight(connId, neurIdPre, neurIdPost, weight, updateWeightRange);
	}


	// +++++++++ PUBLIC METHODS: SETTERS / GETTERS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	CARLsimState getCARLsimState() { return carlsimState_; }

	std::vector<float> getConductanceAMPA(int grpId) {
		std::string funcName = "getConductanceAMPA()";
		UserErrors::assertTrue(carlsimState_ == RUN_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE,
			funcName, funcName, "RUN.");
		UserErrors::assertTrue(grpId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "grpId");
		UserErrors::assertTrue(grpId>=0 && grpId<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName, "grpId",
			"[0,getNumGroups()]");

		return snn_->getConductanceAMPA(grpId);
	}

	std::vector<float> getConductanceNMDA(int grpId) {
		std::string funcName = "getConductanceNMDA()";
		UserErrors::assertTrue(carlsimState_ == RUN_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE,
			funcName, funcName, "RUN.");
		UserErrors::assertTrue(grpId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "grpId");
		UserErrors::assertTrue(grpId>=0 && grpId<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName, "grpId",
			"[0,getNumGroups()]");

		return snn_->getConductanceNMDA(grpId);
	}

	std::vector<float> getConductanceGABAa(int grpId) {
		std::string funcName = "getConductanceGABAa()";
		UserErrors::assertTrue(carlsimState_ == RUN_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE,
			funcName, funcName, "RUN.");
		UserErrors::assertTrue(grpId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "grpId");
		UserErrors::assertTrue(grpId>=0 && grpId<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName, "grpId",
			"[0,getNumGroups()]");

		return snn_->getConductanceGABAa(grpId);
	}

	std::vector<float> getConductanceGABAb(int grpId) {
		std::string funcName = "getConductanceGABAb()";
		UserErrors::assertTrue(carlsimState_ == RUN_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE,
			funcName, funcName, "RUN.");
		UserErrors::assertTrue(grpId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "grpId");
		UserErrors::assertTrue(grpId>=0 && grpId<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName, "grpId",
			"[0,getNumGroups()]");

		return snn_->getConductanceGABAb(grpId);
	}

	RangeDelay getDelayRange(short int connId) {
		std::stringstream funcName;	funcName << "getDelayRange(" << connId << ")";
		UserErrors::assertTrue(connId>=0 && connId<getNumConnections(), UserErrors::MUST_BE_IN_RANGE, funcName.str(),
			"connId", "[0,getNumConnections()]");

		return snn_->getDelayRange(connId);
	}

	// \TODO bad API design (return allocated memory space)
	uint8_t* getDelays(int gIDpre, int gIDpost, int& Npre, int& Npost) {
		std::string funcName = "getDelays()";
		UserErrors::assertTrue(carlsimState_ == SETUP_STATE || carlsimState_ == RUN_STATE,
			UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "SETUP or RUN.");
		UserErrors::assertTrue(gIDpre>=0 && gIDpre<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName, "gIDpre",
			"[0,getNumGroups()]");
		UserErrors::assertTrue(gIDpost>=0 && gIDpost<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName, "gIDpre",
			"[0,getNumGroups()]");

		return snn_->getDelays(gIDpre,gIDpost,Npre,Npost);
	}

	Grid3D getGroupGrid3D(int grpId) {
		std::stringstream funcName;	funcName << "getGroupGrid3D(" << grpId << ")";
		UserErrors::assertTrue(grpId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName.str(), "grpId");
		UserErrors::assertTrue(grpId>=0 && grpId<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName.str(),
			"grpId", "[0,getNumGroups()]");

		return snn_->getGroupGrid3D(grpId);
	}

	int getGroupId(std::string grpName) {
		return snn_->getGroupId(grpName);
	}

	std::string getGroupName(int grpId) {
		std::stringstream funcName; funcName << "getGroupName(" << grpId << ")";
		UserErrors::assertTrue(grpId==ALL || grpId>=0 && grpId<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName.str(),
			"grpId", "[0,getNumGroups()] or must be ALL.");

		return snn_->getGroupName(grpId);
	}

	int getGroupStartNeuronId(int grpId) {
		std::stringstream funcName; funcName << "getGroupStartNeuronId(" << grpId << ")";
		UserErrors::assertTrue(carlsimState_ == SETUP_STATE || carlsimState_ == RUN_STATE,
			UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName.str(), funcName.str(), "SETUP or RUN.");
		UserErrors::assertTrue(grpId>=0 && grpId<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName.str(), "grpId",
			"[0,getNumGroups()]");

		return snn_->getGroupStartNeuronId(grpId);
	}

	int getGroupEndNeuronId(int grpId) {
		std::stringstream funcName; funcName << "getGroupEndNeuronId(" << grpId << ")";
		UserErrors::assertTrue(carlsimState_ == SETUP_STATE || carlsimState_ == RUN_STATE,
			UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName.str(), funcName.str(), "SETUP or RUN.");
		UserErrors::assertTrue(grpId>=0 && grpId<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName.str(), "grpId",
			"[0,getNumGroups()]");

		return snn_->getGroupEndNeuronId(grpId);
	}

	int getGroupNumNeurons(int grpId) {
		std::stringstream funcName; funcName << "getGroupNumNeurons(" << grpId << ")";
		UserErrors::assertTrue(grpId>=0 && grpId<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName.str(), "grpId",
			"[0,getNumGroups()]");

		return snn_->getGroupNumNeurons(grpId);
	}

	Point3D getNeuronLocation3D(int neurId) {
		std::stringstream funcName;	funcName << "getNeuronLocation3D(" << neurId << ")";
		UserErrors::assertTrue(carlsimState_ == SETUP_STATE || carlsimState_ == RUN_STATE,
			UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName.str(), funcName.str(), "SETUP or RUN.");
		UserErrors::assertTrue(neurId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName.str(), "neurId");
		UserErrors::assertTrue(neurId>=0 && neurId<getNumNeurons(), UserErrors::MUST_BE_IN_RANGE, funcName.str(),
			"neurId", "[0,getNumNeurons()]");

		return snn_->getNeuronLocation3D(neurId);
	}

	Point3D getNeuronLocation3D(int grpId, int relNeurId) {
		std::stringstream funcName;	funcName << "getNeuronLocation3D(" << grpId << "," << relNeurId << ")";
		UserErrors::assertTrue(relNeurId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName.str(), "neurId");
		UserErrors::assertTrue(grpId>=0 && grpId<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName.str(),
			"grpId", "[0,getNumGroups()]");
		UserErrors::assertTrue(relNeurId>=0 && relNeurId<getGroupNumNeurons(grpId), UserErrors::MUST_BE_IN_RANGE,
			funcName.str(), "relNeurId", "[0,getGroupNumNeurons()]");

		return snn_->getNeuronLocation3D(grpId, relNeurId);
	}

	int getNumConnections() { return snn_->getNumConnections(); }
	int getMaxNumCompConnections() { return (int)MAX_NUM_COMP_CONN; }

	int getNumGroups() { return snn_->getNumGroups(); }
	int getNumNeurons() { return snn_->getNumNeurons(); }
	int getNumNeuronsReg() { return snn_->getNumNeuronsReg(); }
	int getNumNeuronsRegExc() { return snn_->getNumNeuronsRegExc(); }
	int getNumNeuronsRegInh() { return snn_->getNumNeuronsRegInh(); }
	int getNumNeuronsGen() { return snn_->getNumNeuronsGen(); }
	int getNumNeuronsGenExc() { return snn_->getNumNeuronsGenExc(); }
	int getNumNeuronsGenInh() { return snn_->getNumNeuronsGenInh(); }

	int getNumSynapticConnections(short int connectionId) {
		std::stringstream funcName;	funcName << "getNumConnections(" << connectionId << ")";
		UserErrors::assertTrue(carlsimState_ == SETUP_STATE || carlsimState_ == RUN_STATE,
			UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName.str(), funcName.str(), "SETUP or RUN.");
		UserErrors::assertTrue(connectionId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName.str(), "connectionId");
		UserErrors::assertTrue(connectionId>=0 && connectionId<getNumConnections(), UserErrors::MUST_BE_IN_RANGE,
			funcName.str(), "connectionId", "[0,getNumSynapticConnections()]");
		return snn_->getNumSynapticConnections(connectionId);
	}

	int getNumSynapses() {
		std::string funcName = "getNumSynapses()";
		UserErrors::assertTrue(carlsimState_ == SETUP_STATE || carlsimState_ == RUN_STATE,
			UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "SETUP or RUN.");

		return snn_->getNumSynapses();
	}

	GroupSTDPInfo getGroupSTDPInfo(int grpId) {
		std::stringstream funcName; funcName << "getGroupSTDPInfo(" << grpId << ")";
		UserErrors::assertTrue(grpId >= 0 && grpId<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName.str(),
			"grpId", "[0,getNumGroups()]");

		return snn_->getGroupSTDPInfo(grpId);
	}

	GroupNeuromodulatorInfo getGroupNeuromodulatorInfo(int grpId) {
		std::stringstream funcName; funcName << "getGroupNeuromodulatorInfo(" << grpId << ")";
		UserErrors::assertTrue(grpId >= 0 && grpId<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName.str(),
			"grpId", "[0,getNumGroups()]");
		return snn_->getGroupNeuromodulatorInfo(grpId);
	}

	int getSimTime() { return snn_->getSimTime(); }
	int getSimTimeSec() { return snn_->getSimTimeSec(); }
	int getSimTimeMsec() { return snn_->getSimTimeMs(); }

	// returns pointer to existing SpikeMonitor object, NULL else
	SpikeMonitor* getSpikeMonitor(int grpId) {
		std::stringstream funcName; funcName << "getSpikeMonitor(" << grpId << ")";
		UserErrors::assertTrue(grpId>=0 && grpId<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName.str(),
			"grpId", "[0,getNumGroups()]");

		return snn_->getSpikeMonitor(grpId);
	}

	RangeWeight getWeightRange(short int connId) {
		std::stringstream funcName;	funcName << "getWeightRange(" << connId << ")";
		UserErrors::assertTrue(connId>=0 && connId<getNumConnections(), UserErrors::MUST_BE_IN_RANGE, funcName.str(),
			"connId", "[0,getNumConnections()]");

		return snn_->getWeightRange(connId);
	}

	bool isConnectionPlastic(short int connId) {
		std::stringstream funcName; funcName << "isConnectionPlastic(" << connId << ")";
		UserErrors::assertTrue(connId>=0 && connId<getNumConnections(), UserErrors::MUST_BE_IN_RANGE, funcName.str(),
			"connId", "[0,getNumConnections()]");

		return snn_->isConnectionPlastic(connId);
	}

	bool isGroupWithHomeostasis(int grpId) {
		std::stringstream funcName; funcName << "isGroupWithHomeostasis(" << grpId << ")";
		UserErrors::assertTrue(grpId>=0 && grpId<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName.str(),
			"connId", "[0,getNumGroups()]");

		return snn_->isGroupWithHomeostasis(grpId);	
	}

	bool isExcitatoryGroup(int grpId) {
		std::stringstream funcName; funcName << "isExcitatoryGroup(" << grpId << ")";
		UserErrors::assertTrue(grpId>=0 && grpId<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName.str(),
			"connId", "[0,getNumGroups()]");

		return snn_->isExcitatoryGroup(grpId);
	}

	bool isInhibitoryGroup(int grpId) {
		std::stringstream funcName; funcName << "isInhibitoryGroup(" << grpId << ")";
		UserErrors::assertTrue(grpId>=0 && grpId<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName.str(),
			"connId", "[0,getNumGroups()]");

		return snn_->isInhibitoryGroup(grpId);
	}

	bool isPoissonGroup(int grpId) {
		std::stringstream funcName; funcName << "isPoissonGroup(" << grpId << ")";
		UserErrors::assertTrue(grpId>=0 && grpId<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName.str(),
			"connId", "[0,getNumGroups()]");

		return snn_->isPoissonGroup(grpId);
	}


	// +++++++++ PUBLIC METHODS: SET DEFAULTS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	// set default values for conductance decay times
	void setDefaultConductanceTimeConstants(int tdAMPA, int trNMDA, int tdNMDA, int tdGABAa, int trGABAb,
		int tdGABAb)
	{
		std::stringstream funcName;	funcName << "setDefaultConductanceTimeConstants(" << tdAMPA << "," << trNMDA <<
			"," << tdNMDA << "," << tdGABAa << "," << trGABAb << "," << tdGABAb << ")";
		UserErrors::assertTrue(tdAMPA>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "tdAMPA");
		UserErrors::assertTrue(trNMDA>=0, UserErrors::CANNOT_BE_NEGATIVE, funcName.str(), "trNMDA");
		UserErrors::assertTrue(tdNMDA>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "tdNMDA");
		UserErrors::assertTrue(tdGABAa>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "tdGABAa");
		UserErrors::assertTrue(trGABAb>=0, UserErrors::CANNOT_BE_NEGATIVE, funcName.str(), "trGABAb");
		UserErrors::assertTrue(tdGABAb>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "tdGABAb");
		UserErrors::assertTrue(trNMDA!=tdNMDA, UserErrors::CANNOT_BE_IDENTICAL, funcName.str(), "trNMDA and tdNMDA");
		UserErrors::assertTrue(trGABAb!=tdGABAb, UserErrors::CANNOT_BE_IDENTICAL, funcName.str(), 
			"trGABAb and tdGABAb");
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName.str(),
			"CONFIG.");

		def_tdAMPA_  = tdAMPA;
		def_trNMDA_  = trNMDA;
		def_tdNMDA_  = tdNMDA;
		def_tdGABAa_ = tdGABAa;
		def_trGABAb_ = trGABAb;
		def_tdGABAb_ = tdGABAb;
	}

	void setDefaultHomeostasisParams(float homeoScale, float avgTimeScale) {
		std::string funcName = "setDefaultHomeostasisparams()";
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, 
			funcName, "CONFIG.");
		assert(avgTimeScale>0); // TODO make nice

		def_homeo_scale_ = homeoScale;
		def_homeo_avgTimeScale_ = avgTimeScale;
	}

	void setDefaultSaveOptions(std::string fileName, bool saveSynapseInfo) {
		std::string funcName = "setDefaultSaveOptions()";
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, 
			funcName, "CONFIG.");

		def_save_fileName_ = fileName;
		def_save_synapseInfo_ = saveSynapseInfo;

		// try to open save file to make sure we have permission (so the user immediately knows about the error and 
		// doesn't have to wait until their simulation run has ended)
		FILE* fpTry = fopen(def_save_fileName_.c_str(),"wb");
		UserErrors::assertTrue(fpTry!=NULL,UserErrors::FILE_CANNOT_OPEN,"Default save file",def_save_fileName_);
		fclose(fpTry);
	}

	// wrapper function, set default values for E-STDP params
	void setDefaultSTDPparams(float alphaPlus, float tauPlus, float alphaMinus, float tauMinus, STDPType stdpType) {
		setDefaultESTDPparams(alphaPlus, tauPlus, alphaMinus, tauMinus, stdpType);
	}

	// set default values for E-STDP params
	void setDefaultESTDPparams(float alphaPlus, float tauPlus, float alphaMinus, float tauMinus, STDPType stdpType) {
		std::string funcName = "setDefaultESTDPparams()";
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");
		UserErrors::assertTrue(tauPlus > 0, UserErrors::MUST_BE_POSITIVE, funcName, "tauPlus");
		UserErrors::assertTrue(tauMinus > 0, UserErrors::MUST_BE_POSITIVE, funcName, "tauMinus");
		switch(stdpType) {
			case STANDARD:
			def_STDP_type_ = STANDARD;
			break;
			case DA_MOD:
			def_STDP_type_ = DA_MOD;
			break;
			default:
			stdpType=UNKNOWN_STDP;
			UserErrors::assertTrue(stdpType != UNKNOWN_STDP,UserErrors::CANNOT_BE_UNKNOWN,funcName);
			break;
		}
		def_STDP_alphaLTP_ = alphaPlus;
		def_STDP_tauLTP_ = tauPlus;
		def_STDP_alphaLTD_ = alphaMinus;
		def_STDP_tauLTD_ = tauMinus;
	}

// set default values for I-STDP params
	void setDefaultISTDPparams(float betaLTP, float betaLTD, float lambda, float delta, STDPType stdpType) {
		std::string funcName = "setDefaultISTDPparams()";
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");
		UserErrors::assertTrue(betaLTP > 0, UserErrors::MUST_BE_POSITIVE, funcName);
		UserErrors::assertTrue(betaLTD > 0, UserErrors::MUST_BE_POSITIVE, funcName);
		UserErrors::assertTrue(lambda > 0, UserErrors::MUST_BE_POSITIVE, funcName);
		UserErrors::assertTrue(delta > 0, UserErrors::MUST_BE_POSITIVE, funcName);
		switch(stdpType) {
			case STANDARD:
			def_STDP_type_ = STANDARD;
			break;
			case DA_MOD:
			def_STDP_type_ = DA_MOD;
			break;
			default:
			stdpType=UNKNOWN_STDP;
			UserErrors::assertTrue(stdpType != UNKNOWN_STDP,UserErrors::CANNOT_BE_UNKNOWN,funcName);
			break;
		}
		def_STDP_betaLTP_ = betaLTP;
		def_STDP_betaLTD_ = betaLTD;
		def_STDP_lambda_ = lambda;
		def_STDP_delta_ = delta;
	}

// set default STP values for an EXCITATORY_NEURON or INHIBITORY_NEURON
	void setDefaultSTPparams(int neurType, float STP_U, float STP_tau_u, float STP_tau_x) {
		std::string funcName = "setDefaultSTPparams()";
		UserErrors::assertTrue(neurType==EXCITATORY_NEURON || neurType==INHIBITORY_NEURON, UserErrors::WRONG_NEURON_TYPE,
			funcName);
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");

		assert(STP_tau_u>0.0f);
		assert(STP_tau_x>0.0f);

		switch (neurType) {
			case EXCITATORY_NEURON:
			def_STP_U_exc_ = STP_U;
			def_STP_tau_u_exc_ = STP_tau_u;
			def_STP_tau_x_exc_ = STP_tau_x;
			break;
		case INHIBITORY_NEURON:
			def_STP_U_inh_ = STP_U;
			def_STP_tau_u_inh_ = STP_tau_u;
			def_STP_tau_x_inh_ = STP_tau_x;
			break;
		default:
			// some error message instead of assert
			UserErrors::assertTrue((neurType == EXCITATORY_NEURON || neurType == INHIBITORY_NEURON),UserErrors::CANNOT_BE_UNKNOWN,funcName);
			break;
		}
	}


private:
	// +++++ PRIVATE METHODS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	// unsafe computations that would otherwise go in constructor
	void CARLsimInit() {
		bool gpuAllocationResult = false;
		std::string funcName = "CARLsimInit()";

		UserErrors::assertTrue(loggerMode_!=UNKNOWN_LOGGER,UserErrors::CANNOT_BE_UNKNOWN,"CARLsim()","Logger mode");

		// init SNN object
		snn_ = new SNN(netName_, preferredSimMode_, loggerMode_, randSeed_);

		// set default time constants for synaptic current decay
		// TODO: add ref
		setDefaultConductanceTimeConstants(5, 0, 150, 6, 0, 150);

		// set default values for STDP params
		// \deprecated
		// \TODO: replace with STDP structs
		setDefaultESTDPparams(0.001f, 20.0f, -0.0012f, 20.0f, STANDARD);
		setDefaultISTDPparams(0.001f, 0.0012f, 12.0f, 40.0f, STANDARD);

		// set default values for STP params
		// Misha Tsodyks and Si Wu (2013) Short-term synaptic plasticity. Scholarpedia, 8(10):3153., revision #136920
		setDefaultSTPparams(EXCITATORY_NEURON, 0.45f, 50.0f, 750.0f);
		setDefaultSTPparams(INHIBITORY_NEURON, 0.15f, 750.0f, 50.0f);

		// set default homeostasis params
		// Ref: Carlson, et al. (2013). Proc. of IJCNN 2013.
		setDefaultHomeostasisParams(0.1f, 10.0f);

		// set default save sim params
		// TODO: when we run executable from local dir, put save file in results/
		setDefaultSaveOptions("results/sim_"+netName_+".dat",false);

		connSyn_.clear();
		connComp_.clear();
	}


	// check whether grpId exists in grpIds_
	bool existsGrpId(int grpId) {
		return std::find(grpIds_.begin(), grpIds_.end(), grpId)!=grpIds_.end();
	}

	// print all user warnings, continue only after user input
	void handleUserWarnings() {
		if (userWarnings_.size()) {
			for (int i=0; i<userWarnings_.size(); i++) {
				CARLSIM_WARN("runNetwork()",userWarnings_[i].c_str());
			}

			fprintf(stdout,"Ignore warnings and continue? Y/n ");
			char ignoreWarn = std::cin.get();
			if (std::cin.fail() || ignoreWarn!='y' && ignoreWarn!='Y') {
				fprintf(stdout,"Exiting...\n");
				exit(1);
			}
		}
	}


	// +++++ PRIVATE STATIC PROPERTIES ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	static bool gpuAllocation[MAX_NUM_CUDA_DEVICES];
	static std::string gpuOccupiedBy[MAX_NUM_CUDA_DEVICES];
#if defined(WIN32) || defined(WIN64)
	static HANDLE gpuAllocationLock;
#else
	static pthread_mutex_t gpuAllocationLock;
#endif


	// +++++ PRIVATE PROPERTIES +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	CARLsim* sim_;
	SNN* snn_;                  //!< an instance of CARLsim core class
	std::string netName_;       //!< network name
	int randSeed_;              //!< RNG seed
	LoggerMode loggerMode_;     //!< logger mode (USER, DEVELOPER, SILENT, CUSTOM)
	SimMode preferredSimMode_;  //!< preferred simulation mode (CPU_MODE, GPU_MODE, HYBRID_MODE)
	bool enablePrint_;
	bool copyState_;

	//! a 2D matrix storing for each groupId (first dim) to which other groups it is connected to (second dim)
	std::vector<std::vector<int> > connSyn_;
	//! a 2D matrix storing for each groupId (first dim) to which other groups it is compartmentally connected
	//! to (second dim)
	std::vector<std::vector<int> > connComp_;

	std::map<int, int> groupPrefNetIds_; //!< a list of all created groups' preferred net ids

	unsigned int numConnections_;	//!< keep track of number of allocated connections
	std::vector<std::string> userWarnings_; // !< an accumulated list of user warnings

	std::vector<int> grpIds_;		//!< a list of all created group IDs
	std::vector<SpikeGeneratorCore*> spkGen_; //!< a list of all created spike generators
	std::vector<ConnectionGeneratorCore*> connGen_; //!< a list of all created connection generators

	bool hasSetHomeoALL_;			//!< informs that homeostasis have been set for ALL groups (can't add more groups)
	bool hasSetHomeoBaseFiringALL_;	//!< informs that base firing has been set for ALL groups (can't add more groups)
	bool hasSetSTDPALL_; 			//!< informs that STDP have been set for ALL groups (can't add more groups)
	bool hasSetSTPALL_; 			//!< informs that STP have been set for ALL groups (can't add more groups)
	bool hasSetConductances_;		//!< informs that setConductances has been called
	CARLsimState carlsimState_;	//!< the current state of carlsim

	int def_tdAMPA_;				//!< default value for AMPA decay (ms)
	int def_trNMDA_;				//!< default value for NMDA rise (ms)
	int def_tdNMDA_;				//!< default value for NMDA decay (ms)
	int def_tdGABAa_;				//!< default value for GABAa decay (ms)
	int def_trGABAb_;				//!< default value for GABAb rise (ms)
	int def_tdGABAb_;				//!< default value for GABAb decay (ms)

	// all default values for STDP
	STDPType def_STDP_type_;		//!< default mode for STDP
	float def_STDP_alphaLTP_;		//!< default value for LTP amplitude
	float def_STDP_tauLTP_;			//!< default value for LTP decay (ms)
	float def_STDP_alphaLTD_;		//!< default value for LTD amplitude
	float def_STDP_tauLTD_;			//!< default value for LTD decay (ms)
	float def_STDP_betaLTP_;		//!< default value for LTP amplitude
	float def_STDP_betaLTD_;		//!< default value for LTD amplitude
	float def_STDP_lambda_;			//!< default value for interval of LTP
	float def_STDP_delta_;			//!< default value for interval of LTD

	// all default values for STP
	float def_STP_U_exc_;			//!< default value for STP U excitatory
	float def_STP_tau_u_exc_;		//!< default value for STP u decay (\tau_F) excitatory (ms)
	float def_STP_tau_x_exc_;		//!< default value for STP x decay (\tau_D) excitatory (ms)
	float def_STP_U_inh_;			//!< default value for STP U inhibitory
	float def_STP_tau_u_inh_;		//!< default value for STP u decay (\tau_F) inhibitory (ms)
	float def_STP_tau_x_inh_;		//!< default value for STP x decay (\tau_D) inhibitory (ms)

	// all default values for homeostasis
	float def_homeo_scale_;			//!< default homeoScale
	float def_homeo_avgTimeScale_;	//!< default avgTimeScale

	// all default values for save file
	std::string def_save_fileName_;	//!< file name for saving network info
	bool def_save_synapseInfo_;		//!< flag to inform whether to include synapse info in fpSave_
};




// ****************************************************************************************************************** //
// CARLSIM API IMPLEMENTATION
// ****************************************************************************************************************** //

// initialize static properties
bool CARLsim::Impl::gpuAllocation[MAX_NUM_CUDA_DEVICES] = {false};
std::string CARLsim::Impl::gpuOccupiedBy[MAX_NUM_CUDA_DEVICES];

#if defined(WIN32) || defined(WIN64)
HANDLE CARLsim::Impl::gpuAllocationLock = CreateMutex(NULL, FALSE, NULL);
#else
pthread_mutex_t CARLsim::Impl::gpuAllocationLock = PTHREAD_MUTEX_INITIALIZER;
#endif

// constructor / destructor
CARLsim::CARLsim(const std::string& netName, SimMode preferredSimMode, LoggerMode loggerMode, int ithGPUs, int randSeed) : 
_impl( new Impl(this, netName, preferredSimMode, loggerMode, randSeed) ) {}
CARLsim::~CARLsim() { delete _impl; }

// connect with primitive type
short int CARLsim::connect(int grpId1, int grpId2, const std::string& connType, const RangeWeight& wt, float connProb,
		const RangeDelay& delay, const RadiusRF& radRF, bool synWtType, float mulSynFast, float mulSynSlow) {
	return _impl->connect(grpId1, grpId2, connType, wt, connProb, delay, radRF, synWtType, mulSynFast, mulSynSlow);
}

// connect with custom ConnectionGenerator (short)
// TODO: don't need two versions of this... make it (grpId1, grpId2, conn, synWtType, mulSynFast, mulSynSlow)
short int CARLsim::connect(int grpId1, int grpId2, ConnectionGenerator* conn, bool synWtType) {
	return _impl->connect(grpId1, grpId2, conn, synWtType);
}
short int CARLsim::connect(int grpId1, int grpId2, ConnectionGenerator* conn, float mulSynFast, float mulSynSlow, 
	bool synWtType)
{
	return _impl->connect(grpId1, grpId2, conn, mulSynFast, mulSynSlow, synWtType);
}

short int CARLsim::connectCompartments(int grpIdLower, int grpIdUpper) {
	return _impl->connectCompartments(grpIdLower, grpIdUpper);
}

// create group with / without grid
int CARLsim::createGroup(const std::string& grpName, const Grid3D& grid, int neurType, int preferredPartition, ComputingBackend preferredBackend) {
	return _impl->createGroup(grpName, grid, neurType, preferredPartition, preferredBackend);
}
int CARLsim::createGroup(const std::string& grpName, int nNeur, int neurType, int preferredPartition, ComputingBackend preferredBackend) {
	return _impl->createGroup(grpName, nNeur, neurType, preferredPartition, preferredBackend);
}

// create LIF group with / without grid	
int CARLsim::createGroupLIF(const std::string& grpName, const Grid3D& grid, int neurType, int preferredPartition, ComputingBackend preferredBackend) {
	return _impl->createGroupLIF(grpName, grid, neurType, preferredPartition, preferredBackend);
}
int CARLsim::createGroupLIF(const std::string& grpName, int nNeur, int neurType, int preferredPartition, ComputingBackend preferredBackend) {
	return _impl->createGroupLIF(grpName, nNeur, neurType, preferredPartition, preferredBackend);
}

// create spike gen group with / without grid
int CARLsim::createSpikeGeneratorGroup(const std::string& grpName, const Grid3D& grid, int neurType, int preferredPartition, ComputingBackend preferredBackend) {
	return _impl->createSpikeGeneratorGroup(grpName, grid, neurType, preferredPartition, preferredBackend);
}
int CARLsim::createSpikeGeneratorGroup(const std::string& grpName, int nNeur, int neurType, int preferredPartition, ComputingBackend preferredBackend) {
	return _impl->createSpikeGeneratorGroup(grpName, nNeur, neurType, preferredPartition, preferredBackend);
}

void CARLsim::setCompartmentParameters(int grpId, float couplingUp, float couplingDown) {
	_impl->setCompartmentParameters(grpId, couplingUp, couplingDown);
}

// set conductances
void CARLsim::setConductances(bool isSet) {
	_impl->setConductances(isSet);
}
void CARLsim::setConductances(bool isSet, int tdAMPA, int tdNMDA, int tdGABAa, int tdGABAb) {
	_impl->setConductances(isSet, tdAMPA, tdNMDA, tdGABAa, tdGABAb);
}
void CARLsim::setConductances(bool isSet, int tdAMPA, int trNMDA, int tdNMDA, int tdGABAa, int trGABAb, int tdGABAb) {
	_impl->setConductances(isSet, tdAMPA, trNMDA, tdNMDA, tdGABAa, trGABAb, tdGABAb);
}

// set homeostasis params
void CARLsim::setHomeostasis(int grpId, bool isSet, float homeoScale, float avgTimeScale) {
	_impl->setHomeostasis(grpId, isSet, homeoScale, avgTimeScale);
}
void CARLsim::setHomeostasis(int grpId, bool isSet) {
	_impl->setHomeostasis(grpId, isSet);
}
void CARLsim::setHomeoBaseFiringRate(int grpId, float baseFiring, float baseFiringSD) {
	_impl->setHomeoBaseFiringRate(grpId, baseFiring, baseFiringSD);
}

void CARLsim::setIntegrationMethod(integrationMethod_t method, int numStepsPerMs)
{
	_impl->setIntegrationMethod(method, numStepsPerMs);
}

// set neuron params
void CARLsim::setNeuronParameters(int grpId, float izh_a, float izh_a_sd, float izh_b, float izh_b_sd, float izh_c, 
	float izh_c_sd, float izh_d, float izh_d_sd)
{
	_impl->setNeuronParameters(grpId, izh_a, izh_a_sd, izh_b, izh_b_sd, izh_c, izh_c_sd, izh_d, izh_d_sd);
}
void CARLsim::setNeuronParameters(int grpId, float izh_a, float izh_b, float izh_c, float izh_d) {
	_impl->setNeuronParameters(grpId, izh_a, izh_b, izh_c, izh_d);
}

void CARLsim::setNeuronParameters(int grpId, float izh_C, float izh_k, float izh_vr, float izh_vt,
	float izh_a, float izh_b, float izh_vpeak, float izh_c, float izh_d)
{
	_impl->setNeuronParameters(grpId, izh_C, izh_k, izh_vr, izh_vt, izh_a, izh_b, izh_vpeak, izh_c, izh_d);
}

void CARLsim::setNeuronParameters(int grpId, float izh_C, float izh_C_sd, float izh_k, float izh_k_sd,
	float izh_vr, float izh_vr_sd, float izh_vt, float izh_vt_sd,
	float izh_a, float izh_a_sd, float izh_b, float izh_b_sd,
	float izh_vpeak, float izh_vpeak_sd, float izh_c, float izh_c_sd,
	float izh_d, float izh_d_sd)
{
	_impl->setNeuronParameters(grpId, izh_C, izh_C_sd, izh_k, izh_k_sd, izh_vr, izh_vr_sd, izh_vt, izh_vt_sd,
		izh_a, izh_a_sd, izh_b, izh_b_sd, izh_vpeak, izh_vpeak_sd, izh_c, izh_c_sd, izh_d, izh_d_sd);
}

void CARLsim::setNeuronParametersLIF(int grpId, int tau_m, int tau_ref, float vTh, float vReset, const RangeRmem& rMem)
{
	_impl->setNeuronParametersLIF(grpId, tau_m, tau_ref, vTh, vReset, rMem);
}

void CARLsim::setNeuromodulator(int grpId, float baseDP, float tauDP, float base5HT, float tau5HT, float baseACh, 
	float tauACh, float baseNE, float tauNE)
{
	_impl->setNeuromodulator(grpId, baseDP, tauDP, base5HT, tau5HT, baseACh, tauACh, baseNE, tauNE);
}

// sets default neuromodulators
void CARLsim::setNeuromodulator(int grpId, float tauDP, float tau5HT, float tauACh, float tauNE) {
	_impl->setNeuromodulator(grpId, tauDP, tau5HT, tauACh, tauNE);
}

// Sets default STDP mode and params
void CARLsim::setSTDP(int grpId, bool isSet) { _impl->setSTDP(grpId, isSet); }

// Sets STDP params for a group, custom
void CARLsim::setSTDP(int grpId, bool isSet, STDPType type, float alphaPlus, float tauPlus, float alphaMinus, 
	float tauMinus)
{
	_impl->setSTDP(grpId, isSet, type, alphaPlus, tauPlus, alphaMinus, tauMinus);
}

// Sets default E-STDP mode and parameters
void CARLsim::setESTDP(int grpId, bool isSet) { _impl->setESTDP(grpId, isSet); }

// Sets E-STDP with the exponential curve
void CARLsim::setESTDP(int grpId, bool isSet, STDPType type, ExpCurve curve) {
	_impl->setESTDP(grpId, isSet, type, curve);
}

// Sets E-STDP with the timing-based curve
void CARLsim::setESTDP(int grpId, bool isSet, STDPType type, TimingBasedCurve curve) {
	_impl->setESTDP(grpId, isSet, type, curve);
}

// Sets default I-STDP mode and parameters
void CARLsim::setISTDP(int grpId, bool isSet) { _impl->setESTDP(grpId, isSet); }

// Sets I-STDP with the exponential curve
void CARLsim::setISTDP(int grpId, bool isSet, STDPType type, ExpCurve curve) {
	_impl->setISTDP(grpId, isSet, type, curve);
}

// Sets I-STDP with the pulse curve
void CARLsim::setISTDP(int grpId, bool isSet, STDPType type, PulseCurve curve) {
	_impl->setISTDP(grpId, isSet, type, curve);
}

// Sets STP params U, tau_u, and tau_x of a neuron group (pre-synaptically)
void CARLsim::setSTP(int grpId, bool isSet, float STP_U, float STP_tau_u, float STP_tau_x) {
	_impl->setSTP(grpId, isSet, STP_U, STP_tau_u, STP_tau_x);
}

// Sets STP params U, tau_u, and tau_x of a neuron group (pre-synaptically) using default values
void CARLsim::setSTP(int grpId, bool isSet) { _impl->setSTP(grpId, isSet); }

// Sets the weight and weight change update parameters
void CARLsim::setWeightAndWeightChangeUpdate(UpdateInterval wtANDwtChangeUpdateInterval, bool enableWtChangeDecay, 
	float wtChangeDecay)
{
	_impl->setWeightAndWeightChangeUpdate(wtANDwtChangeUpdateInterval, enableWtChangeDecay, wtChangeDecay);
}


// run the simulation for time=(nSec*seconds + nMsec*milliseconds)
int CARLsim::runNetwork(int nSec, int nMsec, bool printRunSummary) {
	return _impl->runNetwork(nSec, nMsec, printRunSummary);
}

// build the network
void CARLsim::setupNetwork() { _impl->setupNetwork(); }

const FILE* CARLsim::getLogFpInf() { return _impl->getLogFpInf(); }
const FILE* CARLsim::getLogFpErr() { return _impl->getLogFpErr(); }
const FILE* CARLsim::getLogFpDeb() { return _impl->getLogFpDeb(); }
const FILE* CARLsim::getLogFpLog() { return _impl->getLogFpLog(); }

// Saves important simulation and network infos to file.
void CARLsim::saveSimulation(const std::string& fileName, bool saveSynapseInfo) {
	_impl->saveSimulation(fileName, saveSynapseInfo);
}

// Sets the name of the log file
void CARLsim::setLogFile(const std::string& fileName) { _impl->setLogFile(fileName); }

// Sets the file pointers for all log files in CUSTOM mode
void CARLsim::setLogsFpCustom(FILE* fpInf, FILE* fpErr, FILE* fpDeb, FILE* fpLog) {
	_impl->setLogsFpCustom(fpInf, fpErr, fpDeb, fpLog);
}


// Adds a constant bias to the weight of every synapse in the connection
void CARLsim::biasWeights(short int connId, float bias, bool updateWeightRange) {
	_impl->biasWeights(connId, bias, updateWeightRange);
}

// Loads a simulation (and network state) from file. The file pointer fid must point to a
void CARLsim::loadSimulation(FILE* fid) { _impl->loadSimulation(fid); }

// Multiplies the weight of every synapse in the connection with a scaling factor
void CARLsim::scaleWeights(short int connId, float scale, bool updateWeightRange) {
	_impl->scaleWeights(connId, scale, updateWeightRange);
}

// Sets a connection monitor for a group, custom ConnectionMonitor class
ConnectionMonitor* CARLsim::setConnectionMonitor(int grpIdPre, int grpIdPost, const std::string& fname) {
	return _impl->setConnectionMonitor(grpIdPre, grpIdPost, fname);
}

// Sets the amount of current (mA) to inject into a group
void CARLsim::setExternalCurrent(int grpId, const std::vector<float>& current) {
	_impl->setExternalCurrent(grpId, current);
}

// Sets the amount of current (mA) to inject to each neuron in a group
void CARLsim::setExternalCurrent(int grpId, float current) { _impl->setExternalCurrent(grpId, current); }

// Sets a group monitor for a group, custom GroupMonitor class
GroupMonitor* CARLsim::setGroupMonitor(int grpId, const std::string& fname) {
	return _impl->setGroupMonitor(grpId, fname);
}

// Associates a SpikeGenerator object with a group
void CARLsim::setSpikeGenerator(int grpId, SpikeGenerator* spikeGenFunc) {
	_impl->setSpikeGenerator(grpId, spikeGenFunc);
}

// Sets a Spike Monitor for a groups, prints spikes to binary file
SpikeMonitor* CARLsim::setSpikeMonitor(int grpId, const std::string& fileName) {
	return _impl->setSpikeMonitor(grpId, fileName);
}

// Sets a Neuron Monitor for a groups, prints neuron state values (voltage, recovery, and total current values) to binary file
NeuronMonitor* CARLsim::setNeuronMonitor(int grpId, const std::string& fileName) {
	return _impl->setNeuronMonitor(grpId, fileName);
}

// Sets a spike rate
void CARLsim::setSpikeRate(int grpId, PoissonRate* spikeRate, int refPeriod) {
	_impl->setSpikeRate(grpId, spikeRate, refPeriod);
}

// Sets the weight value of a specific synapse
void CARLsim::setWeight(short int connId, int neurIdPre, int neurIdPost, float weight, bool updateWeightRange) {
	_impl->setWeight(connId, neurIdPre, neurIdPost, weight, updateWeightRange);
}

// Enters a testing phase in which all weight changes are disabled
void CARLsim::startTesting(bool updateWeights) { _impl->startTesting(updateWeights); }

// Exits a testing phase, making weight changes possible again
void CARLsim::stopTesting() { _impl->stopTesting(); }

// Returns the current CARLsim state
CARLsimState CARLsim::getCARLsimState() { return _impl->getCARLsimState(); }

// gets AMPA vector of a group
std::vector<float> CARLsim::getConductanceAMPA(int grpId) { return _impl->getConductanceAMPA(grpId); }

// gets NMDA vector of a group
std::vector<float> CARLsim::getConductanceNMDA(int grpId) { return _impl->getConductanceNMDA(grpId); }

// gets GABAa vector of a group
std::vector<float> CARLsim::getConductanceGABAa(int grpId) { return _impl->getConductanceGABAa(grpId); }

// gets GABAb vector of a group
std::vector<float> CARLsim::getConductanceGABAb(int grpId) { return _impl->getConductanceGABAb(grpId); }

// returns the RangeDelay struct for a specific connection ID
RangeDelay CARLsim::getDelayRange(short int connId) { return _impl->getDelayRange(connId); }

// gets delays
uint8_t* CARLsim::getDelays(int gIDpre, int gIDpost, int& Npre, int& Npost) {
	return _impl->getDelays(gIDpre, gIDpost, Npre, Npost);
}

// returns the 3D grid struct of a group
Grid3D CARLsim::getGroupGrid3D(int grpId) { return _impl->getGroupGrid3D(grpId); }

int CARLsim::getGroupId(std::string grpName) { return _impl->getGroupId(grpName); }

// gets group name
std::string CARLsim::getGroupName(int grpId) { return _impl->getGroupName(grpId); }

// returns the 3D location a neuron codes for
Point3D CARLsim::getNeuronLocation3D(int neurId) { return _impl->getNeuronLocation3D(neurId); }

// returns the 3D location a neuron codes for
Point3D CARLsim::getNeuronLocation3D(int grpId, int relNeurId) { return _impl->getNeuronLocation3D(grpId, relNeurId); }

// Returns the number of connections (pairs of pre-post groups) in the network
int CARLsim::getNumConnections() { return _impl->getNumConnections(); }

int CARLsim::getMaxNumCompConnections() { return _impl->getMaxNumCompConnections(); }

// returns the number of connections associated with a connection ID
int CARLsim::getNumSynapticConnections(short int connectionId) { return _impl->getNumSynapticConnections(connectionId); }

// returns the number of groups in the network
int CARLsim::getNumGroups() { return _impl->getNumGroups(); }

// returns the total number of allocated neurons in the network
int CARLsim::getNumNeurons() { return _impl->getNumNeurons(); }

// returns the total number of regular (Izhikevich) neurons
int CARLsim::getNumNeuronsReg() { return _impl->getNumNeuronsReg(); }

// returns the total number of regular (Izhikevich) excitatory neurons
int CARLsim::getNumNeuronsRegExc() { return _impl->getNumNeuronsRegExc(); }

// returns the total number of regular (Izhikevich) inhibitory neurons
int CARLsim::getNumNeuronsRegInh() { return _impl->getNumNeuronsRegInh(); }

// returns the total number of spike generator neurons
int CARLsim::getNumNeuronsGen() { return _impl->getNumNeuronsGen(); }

// returns the total number of excitatory spike generator neurons
int CARLsim::getNumNeuronsGenExc() { return _impl->getNumNeuronsGenExc(); }

// returns the total number of inhibitory spike generator neurons
int CARLsim::getNumNeuronsGenInh() { return _impl->getNumNeuronsGenInh(); }

// returns the total number of allocated post-synaptic connections in the network
int CARLsim::getNumSynapses() { return _impl->getNumSynapses(); }

// returns the first neuron id of a groupd specified by grpId
int CARLsim::getGroupStartNeuronId(int grpId) { return _impl->getGroupStartNeuronId(grpId); }

// returns the last neuron id of a groupd specified by grpId
int CARLsim::getGroupEndNeuronId(int grpId) { return _impl->getGroupEndNeuronId(grpId); }

// returns the number of neurons of a group specified by grpId
int CARLsim::getGroupNumNeurons(int grpId) { return _impl->getGroupNumNeurons(grpId); }

// returns the stdp information of a group specified by grpId
GroupSTDPInfo CARLsim::getGroupSTDPInfo(int grpId) { return _impl->getGroupSTDPInfo(grpId); }

// returns the neuromodulator information of a group specified by grpId
GroupNeuromodulatorInfo CARLsim::getGroupNeuromodulatorInfo(int grpId) {
	return _impl->getGroupNeuromodulatorInfo(grpId);
}

int CARLsim::getSimTime() { return _impl->getSimTime(); }

int CARLsim::getSimTimeSec() { return _impl->getSimTimeSec(); }

int CARLsim::getSimTimeMsec() { return _impl->getSimTimeMsec(); }

// returns pointer to previously allocated SpikeMonitor object, NULL else
SpikeMonitor* CARLsim::getSpikeMonitor(int grpId) { return _impl->getSpikeMonitor(grpId); }

// returns the RangeWeight struct for a specific connection ID
RangeWeight CARLsim::getWeightRange(short int connId) { return _impl->getWeightRange(connId); }

// Returns whether a connection is fixed or plastic
bool CARLsim::isConnectionPlastic(short int connId) { return _impl->isConnectionPlastic(connId); }

// Returns whether a group has homeostasis enabled
bool CARLsim::isGroupWithHomeostasis(int grpId) { return _impl->isGroupWithHomeostasis(grpId); }

bool CARLsim::isExcitatoryGroup(int grpId) { return _impl->isExcitatoryGroup(grpId); }

bool CARLsim::isInhibitoryGroup(int grpId) { return _impl->isInhibitoryGroup(grpId); }

bool CARLsim::isPoissonGroup(int grpId) { return _impl->isPoissonGroup(grpId); }

void CARLsim::setDefaultConductanceTimeConstants(int tdAMPA, int trNMDA, int tdNMDA, int tdGABAa, int trGABAb, 
	int tdGABAb)
{
	_impl->setDefaultConductanceTimeConstants(tdAMPA, trNMDA, tdNMDA, tdGABAa, trGABAb, tdGABAb);
}

// Sets default homeostasis params
void CARLsim::setDefaultHomeostasisParams(float homeoScale, float avgTimeScale) {
	_impl->setDefaultHomeostasisParams(homeoScale, avgTimeScale);
}

// Sets default options for save file
void CARLsim::setDefaultSaveOptions(std::string fileName, bool saveSynapseInfo) {
	_impl->setDefaultSaveOptions(fileName, saveSynapseInfo);
}

// sets default STDP params
void CARLsim::setDefaultSTDPparams(float alphaPlus, float tauPlus, float alphaMinus, float tauMinus, STDPType stdpType) {
	_impl->setDefaultSTDPparams(alphaPlus, tauPlus, alphaMinus, tauMinus, stdpType);
}

// sets default values for E-STDP params
void CARLsim::setDefaultESTDPparams(float alphaPlus, float tauPlus, float alphaMinus, float tauMinus, STDPType stdpType) {
	_impl->setDefaultESTDPparams(alphaPlus, tauPlus, alphaMinus, tauMinus, stdpType);
}

// sets default values for I-STDP params
void CARLsim::setDefaultISTDPparams(float betaLTP, float betaLTD, float lambda, float delta, STDPType stdpType) {
	_impl->setDefaultISTDPparams(betaLTP, betaLTD, lambda, delta, stdpType);
}

// Sets default values for STP params U, tau_u, and tau_x of a neuron group (pre-synaptically)
void CARLsim::setDefaultSTPparams(int neurType, float STP_U, float STP_tau_u, float STP_tau_x) {
	_impl->setDefaultSTPparams(neurType, STP_U, STP_tau_u, STP_tau_x);
}
