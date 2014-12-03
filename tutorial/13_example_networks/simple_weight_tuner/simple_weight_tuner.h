#ifndef _SIMPLE_WEIGHT_TUNER_H_
#define _SIMPLE_WEIGHT_TUNER_H_

class CARLsim;
class SpikeMonitor;

class SimpleWeightTuner {
public:
	SimpleWeightTuner(CARLsim *sim, short int connId, float errorMargin=1e-5f, float initWt=0.0f);
	~SimpleWeightTuner();

	void setTargetFiringRate(int grpId, float targetRate);

	void iterate(int runDurationMs=1000);

	bool done();

	float getWeight() { return wtTrack_; }


private:
	CARLsim *sim_;
	SpikeMonitor *SM_;

	short int connId_;

	float errorMargin_;
	int grpId_;
	float targetRate_;
	bool isDone_;
	float initWt_;
	double wtStepSize_;
	bool wtShouldIncrease_;

	float currentError_;

	float wtTrack_;

};

#endif