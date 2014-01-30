/*
 * ISSUE 23: REAL-TIME SPIKE MONITOR
 * see list of issues on GitHub
 *
 * Feature description:
 *   If we are to integrate CARLsim with real-time systems, we need to be able to get spikes out per frame, not only per
 *   second of simulation time.
 *
 * How bug was fixed:
 *   
 *
 * How the following script tests the bug:
 *
 * Ver 01/24/14 mb
 */

#include "snn.h"
#include <sys/time.h>
#include <ctime>

extern MTRand getRand;


/* Returns the amount of milliseconds elapsed since the UNIX epoch. Works on both
 * windows and linux. */

unsigned long int get_time_ms64()
{
#ifdef WIN32
 /* Windows */
 FILETIME ft;
 LARGE_INTEGER li;

 /* Get the amount of 100 nano seconds intervals elapsed since January 1, 1601 (UTC) and copy it
  * to a LARGE_INTEGER structure. */
 GetSystemTimeAsFileTime(&ft);
 li.LowPart = ft.dwLowDateTime;
 li.HighPart = ft.dwHighDateTime;

 unsigned long int ret = li.QuadPart;
 ret -= 116444736000000000LL; /* Convert from file time to UNIX epoch time. */
 ret /= 10000; /* From 100 nano seconds (10^-7) to 1 millisecond (10^-3) intervals */

 return ret;
#else
 /* Linux */
 struct timeval tv;

 gettimeofday(&tv, NULL);

 unsigned long int ret = tv.tv_usec;
 /* Convert from micro seconds (10^-6) to milliseconds (10^-3) */
 ret /= 1000;

 /* Adds the seconds (10^0) after converting them to milliseconds (10^-3) */
 ret += (tv.tv_sec * 1000);

 return ret;
#endif
}


int main() {
	CpuSNN* snn;

	int frameDur = 50; 		// frame duration
	int frameNum = 20;		// number of frames per simulation

	int numNeurIn = 1000;		// number of neurons in input group
//	int numNeurOut = 1000;	// number of neurons in output group

	float rateIn = 10.0f; 	// mean spike rate of input group
	PoissonRate rIn(numNeurIn);

	unsigned int* spkOut;

	FILE* fp = fopen("issue23_log.txt", "w");

	for (int numNeurOut=10; numNeurOut<=10; numNeurOut*=10) {

	for (int recordFact=1; recordFact<=1; recordFact++) {

		for (int sim_mode=1; sim_mode<=1; sim_mode++) {
			// run in CPU and GPU mode

			for (int useSpkMonRT=1; useSpkMonRT<=1; useSpkMonRT++) {
				// run with SpikeMonRT off and on
				printf("%s: spkMonRT=%s,\tnOut=%d,\ttSim=%d,\trecDur=%d\n",
					sim_mode?"GPU_MODE":"CPU_MODE",
					useSpkMonRT?"y":"n",
					numNeurOut,
					frameDur*frameNum/1000,
					recordFact*frameDur);

				long unsigned int time_start = get_time_ms64();

				int recordDur = recordFact*frameDur;

				snn = new CpuSNN("issue23");
				int gIn = snn->createSpikeGeneratorGroup("input", numNeurIn, EXCITATORY_NEURON);
				int gOut = snn->createGroup("output", numNeurOut, EXCITATORY_NEURON);

				snn->setNeuronParameters(gOut, 0.02f, 0.2f, -65.0f, 8.0f); // RS
				snn->setConductances(ALL, true, 5.0, 150.0, 6.0, 150.0); // COBA
				snn->connect(gIn,gOut,"random", 0.025, 0.025, 0.1, 1, 1, SYN_FIXED);

				snn->setLogCycle(0, 0, stdout);
		
				// regular spike monitors, updated every 1000ms
				snn->setSpikeMonitor(gIn);
				snn->setSpikeMonitor(gOut);

				// "real-time" spike monitor, keeps track of the number of spikes per neuron in a group
				// works for excitatory/inhibitory neurons (currently not for spike generators)
				// the recording time can be set to any x number of ms, so that after x ms the spike counts will be reset
				// to zero. if x==-1, then the spike counts will never be reset (should only overflow after 97 days of sim)
				// also, spike counts can be manually reset at any time by calling snn->resetSpikeMonitorRealTime(group);
				// you can have only one real-time spike monitor per group. however, a group can have both a regular and a
				// real-time spike monitor
				if (useSpkMonRT) {
					snn->setSpikeMonitorRealTime(gOut,recordDur);
				}

				// initialize
				snn->runNetwork(0,0, sim_mode);


				for (int j=0; j<numNeurIn; j++)
					rIn.rates[j] = rateIn;
				snn->setSpikeRate(gIn, &rIn);

				// main loop
				for (int i=0; i<frameNum; i++) {
					snn->runNetwork(0,frameDur, sim_mode);
					spkOut = snn->getSpikesRealTime(gOut);	// gets out all the spikes
				}

				float secsItTook= (float) (get_time_ms64()-time_start)/1000;

				fprintf(fp,"%s: spkMonRT=%s,\tnOut=%d,\ttSim=%d,\trecDur=%d,\ttime=%3.2f\n",
					sim_mode?"GPU_MODE":"CPU_MODE",
					useSpkMonRT?"y":"n",
					numNeurOut,
					frameDur*frameNum/1000,
					recordDur,
					secsItTook);

/*
				if (useSpkMonRT) {
					spkOut = snn->getSpikesRealTime(gOut);	// gets out all the spikes

					if (spkOut==NULL)
						printf("Group %d: couldn't get spikes\n",gOut);
					else {
						printf("spikes of group %d:\n",gOut);
						for (int n=0; n<snn->grpNumNeurons(gOut); n++) {
							spkOutTotal += spkOut[n];
							printf("%u\t",spkOut[n]);
						}
					}
					printf("\nTotal: %u spikes\n",spkOutTotal);
				}
*/
printf("a");
				// deallocate
				delete snn;
printf("b\n");
			}
		}
	}
	}

	fclose(fp);

	printf("PASSED\n");
	return 0;	
}
