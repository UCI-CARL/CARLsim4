/*! \brief
 *
 */
#include "PTI.h"
#include <carlsim.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cassert>

using namespace std;
using namespace CARLsim_PTI;

namespace CARLsim_PTI {
    class SimpleCA3Experiment : public Experiment {
    public:
        // Decay constants
        static const float COND_tAMPA=5.0, COND_tNMDA=150.0, COND_tGABAa=6.0, COND_tGABAb=150.0;

        // Neurons
        static const int NUM_NEURONS = 1000;
        static const double RND_FRACTION = 0.1;
        static const double PYR_FRACTION = 0.9;
        static const double PTI_FRACTION = 0.05;
        static const double DTI_FRACTION = 0.05;

        // Izhikevich parameters
        static const float PYR_PARAMS[];
        static const float PTI_PARAMS[];
        static const float DTI_PARAMS[];

        // Simulation time (each must be at least 1s due to bug in SpikeMonitor)
        static const int TRANSIENT_MS = 2000;
        static const int EVAL_MS = 20000;

        // Firing rate of input Poisson neurons
        static const float IN_HZ = 10.0f;

        // Target rates for the objective function
        static const float PYR_TARGET_HZ_MIN = 2.0f;
        static const float PYR_TARGET_HZ_MAX = 4.0f;
        static const float PTI_TARGET_HZ_MIN = 15.0f;
        static const float PTI_TARGET_HZ_MAX = 21.0f;
        static const float DTI_TARGET_HZ_MIN = 6.0f;
        static const float DTI_TARGET_HZ_MAX = 11.0f;

        const simMode_t simMode;
        const loggerMode_t verbosity;
        const int deviceID;

        SimpleCA3Experiment(const simMode_t simMode, const loggerMode_t verbosity, const int deviceID) : simMode(simMode), verbosity(verbosity), deviceID(deviceID) {}

        void run(const ParameterInstances &parameters, std::ostream &outputStream) const {
            for(unsigned int i = 0; i < parameters.getNumInstances(); i++) {
	        SimpleCA3Network * const network = SimpleCA3Network::createNetwork(parameters.getInstance(i), simMode, verbosity, deviceID);
	        network->network.get()->runNetwork(0, TRANSIENT_MS);

                network->pyramidalMonitor->startRecording();
                network->ptiMonitor->startRecording();
                network->dtiMonitor->startRecording();

                network->network.get()->runNetwork(0, EVAL_MS);

                network->pyramidalMonitor->stopRecording();
                network->ptiMonitor->stopRecording();
                network->dtiMonitor->stopRecording();

                const float pyrHz = network->pyramidalMonitor->getPopMeanFiringRate();
                const float ptiHz = network->ptiMonitor->getPopMeanFiringRate();
                const float dtiHz = network->dtiMonitor->getPopMeanFiringRate();

                const float pyrError = withinRange(pyrHz, PYR_TARGET_HZ_MIN, PYR_TARGET_HZ_MAX) ? 0 :
                  (pyrHz < PYR_TARGET_HZ_MIN) ? PYR_TARGET_HZ_MIN - pyrHz : pyrHz - PYR_TARGET_HZ_MAX;
                const float ptiError = withinRange(ptiHz, PTI_TARGET_HZ_MIN, PTI_TARGET_HZ_MAX) ? 0 :
                  (ptiHz < PTI_TARGET_HZ_MIN) ? PTI_TARGET_HZ_MIN - ptiHz : ptiHz - PTI_TARGET_HZ_MAX;
                const float dtiError = withinRange(dtiHz, DTI_TARGET_HZ_MIN, DTI_TARGET_HZ_MAX) ? 0 :
                  (dtiHz < DTI_TARGET_HZ_MIN) ? DTI_TARGET_HZ_MIN - dtiHz : dtiHz - DTI_TARGET_HZ_MAX;

                const float fitness = -(pyrError + ptiError + dtiError);
                outputStream << fitness << endl;
            }
        }

    private:
        typedef struct SimpleCA3Network {
            const auto_ptr<CARLsim> network;
	  SpikeMonitor * const pyramidalMonitor;
	  SpikeMonitor * const ptiMonitor;
	  SpikeMonitor * const dtiMonitor;

	private:
	  const auto_ptr<PoissonRate> in;
	public:

	  SimpleCA3Network(CARLsim * const network, SpikeMonitor * const pyramidalMonitor, SpikeMonitor * const ptiMonitor, SpikeMonitor * const dtiMonitor, PoissonRate * const in)
	    : network(network), pyramidalMonitor(pyramidalMonitor), ptiMonitor(ptiMonitor), dtiMonitor(dtiMonitor), in(in) { };

	  /** Decode a genome and construct a CARLsim network on the heap. */
	  static SimpleCA3Network * const createNetwork(const std::vector<double> parameters, const simMode_t simMode, const loggerMode_t verbosity, const int deviceID) {
	      CARLsim * const network = new CARLsim("SimpleCA3", simMode, verbosity, deviceID);

	      network->setConductances(true,COND_tAMPA,COND_tNMDA,COND_tGABAa,COND_tGABAb);
	      const int poissonGroup = network->createSpikeGeneratorGroup("poisson", RND_FRACTION*NUM_NEURONS, EXCITATORY_NEURON);
	      const int pyramidalGroup = network->createGroup("pyramidal", PYR_FRACTION*NUM_NEURONS, EXCITATORY_NEURON);
	      network->setNeuronParameters(pyramidalGroup, PYR_PARAMS[0], PYR_PARAMS[1], PYR_PARAMS[2], PYR_PARAMS[3]);
	      const int ptiGroup = network->createGroup("PTI", PTI_FRACTION*NUM_NEURONS, INHIBITORY_NEURON);
	      network->setNeuronParameters(ptiGroup, PTI_PARAMS[0], PTI_PARAMS[1], PTI_PARAMS[2], PTI_PARAMS[3]);
	      const int dtiGroup = network->createGroup("DTI", DTI_FRACTION*NUM_NEURONS, INHIBITORY_NEURON);
	      network->setNeuronParameters(dtiGroup, DTI_PARAMS[0], DTI_PARAMS[1], DTI_PARAMS[2], DTI_PARAMS[3]);

	      network->connect(poissonGroup, pyramidalGroup, "random", RangeWeight(parameters[0]), parameters[1], RangeDelay(1));
	      network->connect(pyramidalGroup, pyramidalGroup, "random", RangeWeight(parameters[2]), parameters[3], RangeDelay(1));
	      network->connect(pyramidalGroup, ptiGroup, "random", RangeWeight(parameters[4]), parameters[5], RangeDelay(1));
	      network->connect(pyramidalGroup, dtiGroup, "random", RangeWeight(parameters[6]), parameters[7], RangeDelay(1));
	      network->connect(ptiGroup, pyramidalGroup, "random", RangeWeight(parameters[8]), parameters[9], RangeDelay(1));
	      network->connect(ptiGroup, dtiGroup, "random", RangeWeight(parameters[10]), parameters[11], RangeDelay(1));
	      network->connect(dtiGroup, pyramidalGroup, "random", RangeWeight(parameters[12]), parameters[13], RangeDelay(1));
	      network->connect(dtiGroup, ptiGroup, "random", RangeWeight(parameters[14]), parameters[15], RangeDelay(1));
	      network->setupNetwork();

	      PoissonRate * const in = new PoissonRate(RND_FRACTION*NUM_NEURONS);
				/*for (int i=0;i<RND_FRACTION*NUM_NEURONS;i++)*/
                /*in->rates[i] = IN_HZ;*/
				in->setRates(IN_HZ);
	      network->setSpikeRate(poissonGroup,in);

	      SpikeMonitor * const pyramidalMonitor = network->setSpikeMonitor(pyramidalGroup, "/dev/null");
	      SpikeMonitor * const ptiMonitor = network->setSpikeMonitor(ptiGroup, "/dev/null");
	      SpikeMonitor * const dtiMonitor = network->setSpikeMonitor(dtiGroup, "/dev/null");

	      return new SimpleCA3Network(network, pyramidalMonitor, ptiMonitor, dtiMonitor, in);
	    }

        } SimpleCA3Network;

        static bool withinRange(const float x, const float min, const float max) {
            return (x >= min && x <= max);
        }
    };

    const float SimpleCA3Experiment::PYR_PARAMS[] = { 0.02f, 0.25f, -65.0f, 0.05f };
    const float SimpleCA3Experiment::PTI_PARAMS[] = { 0.1f, 0.2f, -65.0f, 2.0f };
    const float SimpleCA3Experiment::DTI_PARAMS[] = { 0.02f, 0.2f, -65.0f, 8.0f };
}

/** If the command-line arguments contain the option "-parameter", returns the value of the
 * argument that immediately follows the option.  If the arguments do not contain "-parameter",
 * returns NULL. */
const char * const getOpt(int argc, const char * const argv[], const char * const parameter) {
  assert(argc >= 0);
  assert(argv != NULL);
  assert(parameter != NULL);

  for (int i = 1; i < argc - 1; i++) {
    char dashParam[strlen(parameter) + 1];
    strcpy(dashParam, "-");
    strcat(dashParam, parameter);
    if (0 == strcmp(dashParam, argv[i]))
      return argv[i + 1];
  }
  return NULL;
}

/** Returns true iff the command-line arguments contain "-parameter". */
const bool hasOpt(int argc, const char * const argv[], const char * const parameter) {
  assert(argc >= 0);
  assert(argv != NULL);
  assert(parameter != NULL);

  for (int i = 1; i < argc; i++) {
    char dashParam[strlen(parameter) + 1];
    strcpy(dashParam, "-");
    strcat(dashParam, parameter);
    if (0 == strcmp(dashParam, argv[i]))
         return true;
  }
  return false;
}

int main(const int argc, const char * const argv[]) {
  const simMode_t simMode = hasOpt(argc, argv, "cpu") ? CPU_MODE : GPU_MODE;
  const loggerMode_t verbosity = hasOpt(argc, argv, "v") ? USER : SILENT;
  const int deviceID = hasOpt(argc, argv, "device") ? atoi(getOpt(argc, argv, "device")) : 0;
  const SimpleCA3Experiment experiment(simMode, verbosity, deviceID);
  const PTI pti(argc, argv, std::cout, std::cin);

  pti.runExperiment(experiment);

  return 0;
}
