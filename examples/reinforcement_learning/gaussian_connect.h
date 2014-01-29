// Include CARLsim core so we can inherit from connectionGenerator
#include "snn.h"

extern MTRand getRand;

// ---------------------------------------------------------------------------------------------------
// BEGIN projection callback functions
// ---------------------------------------------------------------------------------------------------
// Callback function determines the E-E & E-I projection within each map
class GaussianConnect: public ConnectionGenerator 
{
private:
	int src_x; // source group dimension
	int src_y; // source group dimension
	int dest_x; // destination group dimension
	int dest_y; // destination group dimension
	float std;
	float maxWeight;
	float weightScale;
	int min_delay;
	int max_delay;
	int delay_range;
	int radius;
	float gaussian_scale;

public:        
  GaussianConnect(int _src_x, int _src_y, int _dest_x, int _dest_y, float _std, float _maxWeight, float _weightScale, int _min_delay, int _max_delay, int _radius){
    src_x = _src_x;
    src_y = _src_y;
    dest_x = _dest_x;
    dest_y = _dest_y;
    std = _std;
    weightScale = _weightScale;
    maxWeight = _maxWeight;
	min_delay = _min_delay;
	max_delay = _max_delay;
	delay_range = _max_delay - _min_delay + 1;
	radius = _radius;
	gaussian_scale = 0.39894f / std;
  }


  void connect(CpuSNN* net, int srcGrp, int src_idx, int destGrp, int dest_idx, float &weight, float &maxWt, float &delay, bool &connected){
    // extract x and y positions for both source neuron and dest neuron...
    int src_i_x = src_idx % src_x;
    int src_i_y = src_idx / src_x;
    int dest_i_x  = dest_idx % dest_x;
    int dest_i_y  = dest_idx / dest_x;

    // note, the unit of distance is "how many neuron", not um.
	float distance_squar_y = ((dest_i_y - src_i_y) * (dest_i_y - src_i_y));
    float distance_squar_x = ((dest_i_x - src_i_x) * (dest_i_x - src_i_x));
	float gaus = 0.39894f * expf(-(distance_squar_x + distance_squar_y)/(std * std * 2.0)) / std;
		
	connected = (gaussian_scale * getRand()) < gaus;
    //delay = 1;
	delay = sqrtf(distance_squar_x + distance_squar_y) / radius * delay_range;
	if (delay > max_delay) delay = max_delay;
	if (delay < min_delay) delay = min_delay;

    maxWt = maxWeight;
    weight = gaus * weightScale;

	//if (connected)
	//	printf("(%d,%d)-(%d,%d) w = %1.5f, d = %f\n", src_i_x, src_i_y, dest_i_x, dest_i_y, weight, delay);
  }
};
// ---------------------------------------------------------------------------------------------------
// END projection callback functions
// ---------------------------------------------------------------------------------------------------
