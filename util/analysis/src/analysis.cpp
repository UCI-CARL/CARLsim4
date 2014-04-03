#include <analysis.h>



// we aren't using namespace std so pay attention!

analysis::analysis(carlsim* sim){
	sim_=sim;
	return;
}

analysis::~analysis(){
	return;
}

void analysis::setAvgGrpFiringRate(int grpId){
	// record the time we begin counting spikes
	beginTimeSec_=sim->getSimTimeSec();
	beginTimeMs_ =sim->getSimTimeMs();
	// need to see if this has already been set
	// begin counting spikes for this group
	sim->setSpikeCounter(grpId,-1);
	
	return;
}

int analysis::getAvgGrpFiringRate(int grpId){
	// record the time we begin counting spikes
	endTimeSec_=sim->getSimTimeSec();
	endTimeMs_ =sim->getSimTimeMs();
	// need to see if this has already been set
	int* spikeCount;
	int firingRate;
	// grab spikes	
	spikeCount=sim->getSpikeCounter(grpId);
	grpInfo_=getGroupInfo(grpId,0);
	for(int i=0;i<grpInfo_
	
	return;
}

