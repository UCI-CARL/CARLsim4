/*! \brief Using PTI to teach a small SNN how to compute XOR.
 *  
 */
#include "PTI.h"
#include "../../izk/network/Network.h"
#include "../../izk/network/ConnectionScheme.h"
#include "../../izk/network/ConnectionGroup.h"
#include "../../izk/network/NeuronGroup.h"
#include "../../izk/neuron/Izhikevich4Neuron.h"
#include "../../izk/neuron/PoissonNeuron.h"

#include <cassert>
#include <iostream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

using namespace CARLsim_PTI;

namespace CARLsim_PTI {
    /**
     * An experiment that builds a network with Izk, parameterizes its
     * synaptic weights with the ParameterInstances it receives, and evaluates
     * the resultin network's ability to solve an exclusive-or problem.
     * 
     * The network has three groups of neurons (i.e. two layers of synapses).
     * In the first group, there is one Poisson-spiking neuron for each input
     * bit.  In the second group, there is one hidden Izhikevich neuron for each
     * Poisson-spiking neuron.  All the hidden neurons then feed into a
     * single Izhikevich neuron that functions as the output.  The frequency
     * of the output neuron's spike train is used to determine whether the
     * network yields 'true' or 'false' on a given input bit string.
     * 
     * @param numInputs The number of bits in the input string.  This defines
     *  the size of the network -- it has 2*numInputs + 1 neurons and
     *  numinputs^2 + numInputs connections.
     * 
     * @param printTimeSeries If true, the voltage time series of the output
     *  neuron will be printed out in CSV format.  This assumes that numInputs
     *  = 2.
     */
    class IzkExperiment : public Experiment {
    public:
        IzkExperiment(const unsigned int numInputs, const bool printTimeSeries): numInputs(numInputs), printTimeSeries(printTimeSeries) {}
        
        /** Run the experiment, evaluating the fitness of every instance in the
         * ParmatereInstances.  Here we just evaluate each instance sequentially
         * in a for loop.  The reason parameters come in sets of instances,
         * however, is so that we could parallelize the execution if we want. */
        void run(const ParameterInstances &parameters, std::ostream &outputStream) const {
            if (printTimeSeries) // Only works in the two-input case.
                printf("input, ms, p1_spike, p2_spike, h1_voltage, h2_voltage, h1_spike, h2_spike, o_voltage, o_spike\n");
            
            // Generate a set of inputs to test the network with.
            const std::vector< std::vector<bool> > inputInstances = getInputPermutations(numInputs);
            // Build a network and execute it on each input individually.
            for (int i = 0; i < parameters.getNumInstances(); i++) {
                const std::vector<double> weights = parameters.getInstance(i);
                double numCorrect = 0;
                for (int j = 0; j < inputInstances.size(); j++) {
                    // Run the network
                    const double result = runSingleSimulation(inputInstances[j], weights);
                    // Compare the result to an n-arity exclusive OR function.
                    const bool expected = nXor(inputInstances[j]);
                    numCorrect += 1.0 - fabs(expected - result);
                }
                const double fitness = static_cast<double>(numCorrect)/inputInstances.size();
                outputStream << fitness << std::endl;
            }
        }
        
    private:
        const unsigned int numInputs;
        const bool printTimeSeries;
        static const unsigned int numOutputs = 1;
        static const double stepSize = 0.1; // mS
        static const double poissonSpikeRate = 0.1; // kHz
        static const double poissonSpikeThreshold = 30; // mV
        static const double conductanceDecayRate = 50; // mS
        static const unsigned int simulationTime = 5000; // mS
        static const double decisionThreshold = 0.1; // kHz
        
        /** Run a two-layer SNN on an input string of bools and return a value
         * between 0 and 1.0 indicating how close the spike frequency of the
         * output neuron is to the decisionThreshold. */
        double runSingleSimulation(const std::vector<bool> inputs, const std::vector<double> &weights) const {
            assert(weights.size() > 0);
            const int numHiddenNeurons = numInputs;
            assert((numInputs * numHiddenNeurons) + (numHiddenNeurons * numOutputs) == weights.size());
            
            // Convert bool inputs to currents.
            double inputCurrents[numInputs];
            for (int i = 0; i < numInputs; i++)
                inputCurrents[i] = inputs[i] ? poissonSpikeThreshold : 0.0;
            
            /* Build a network on the heap with three layers: an input layer of
             * PoissonNeurons, a hidden layer of Izhikevich4Neurons, and an
             * output layer of Izhikevich4Neurons. */
            
            // Set up neurons
            const Neuron* poissonPrototype = izk_makePoissonNeuron(poissonSpikeRate, poissonSpikeThreshold);
            const Neuron * const izhikevichPrototype = izk_makeIzhikevich4Neuron(0.1, 0.2, -65.0, 2.0); // Fast spiking
            const NeuronGroup* const inputGroup = izk_makeNeuronGroup(poissonPrototype, numInputs);
            const NeuronGroup* const hiddenGroup = izk_makeNeuronGroup(izhikevichPrototype, numHiddenNeurons);
            const NeuronGroup* const outputGroup = izk_makeNeuronGroup(izhikevichPrototype, numOutputs);
            
            // Set up connections
            const ConnectionScheme * const inputHiddenScheme = izk_makeCustomConnectionScheme(&weights[0], numInputs, numHiddenNeurons);
            const ConnectionScheme * const hiddenOutputScheme = izk_makeCustomConnectionScheme(&weights[numInputs * numHiddenNeurons], numHiddenNeurons, numOutputs);
            ConnectionGroup* const connections[] = {
                izk_makeConnectionGroup(inputHiddenScheme, conductanceDecayRate, inputGroup, hiddenGroup),
                izk_makeConnectionGroup(hiddenOutputScheme, conductanceDecayRate, hiddenGroup, outputGroup)
            };
            const Network* const network = izk_makeNetwork(connections, 2);
            
            // Run the network and save the spike frequency of the last 1000 steps.
            std::vector<bool> spikeTrain;
            for (int i = 0; i < simulationTime; i++) {
                izk_Network_step(network, inputCurrents, stepSize);
                if (i >= simulationTime - 1000)
                    spikeTrain.push_back(outputGroup->neuronArray[0]->isSpiking);
                if (printTimeSeries)
                    printf("%s, %f, %s, %s, %s, %s, %s\n", binaryToString(inputs), i*stepSize, izk_NeuronGroup_printIsSpiking(inputGroup), izk_NeuronGroup_printVoltage(hiddenGroup), izk_NeuronGroup_printIsSpiking(hiddenGroup), izk_NeuronGroup_printVoltage(outputGroup), izk_NeuronGroup_printIsSpiking(outputGroup));
            }
            // Free everything that was created with a "make" function.
            izk_Network_free(network);
            izk_ConnectionGroup_free(connections[0]);
            izk_ConnectionGroup_free(connections[1]);
            inputHiddenScheme->free(inputHiddenScheme);
            hiddenOutputScheme->free(hiddenOutputScheme);
            izk_NeuronGroup_free(inputGroup);
            izk_NeuronGroup_free(hiddenGroup);
            izk_NeuronGroup_free(outputGroup);
            poissonPrototype->funcTable->free(poissonPrototype);
            izhikevichPrototype->funcTable->free(izhikevichPrototype);
            
            // Compute the output value from the spike train.
            const double outputkHz = spikeTrainTokHz(spikeTrain);
            return (outputkHz >= decisionThreshold) ? 1.0 : outputkHz/decisionThreshold;
        }
        
        // Computes the spike frequency of a train in kHz
        static double spikeTrainTokHz(std::vector<bool> &spikeTrain) {
            assert(spikeTrain.size() > 0);
            unsigned int sum = 0;
            for(std::vector<bool>::iterator it = spikeTrain.begin(); it != spikeTrain.end(); ++it) {
                if (*it)
                    sum++;
            }
            return static_cast<double>(sum)/(static_cast<double>(spikeTrain.size())*stepSize);
        }
        
        // Convert an integer to a string of bits.
        static std::vector<bool> intToBinary(const unsigned int value, const unsigned int numBits) {
            assert(numBits <= 16);
            std::vector<bool> binaryString;
            for (int i = numBits - 1; i >= 0; i--) {
                const int shiftedInt = value >> i;
                binaryString.push_back(shiftedInt & 1); // Select the least significant bit
            }
            return binaryString;
        }
        
        // Convert a bitString to a c-string for display.
        static const char * binaryToString(const std::vector<bool> binaryString) {
            std::stringstream str;
            for (int i = 0; i < binaryString.size(); i++) {
                str << binaryString[i];
            }
            return str.str().c_str();
        }
        
        // Compute all bit strings of length numBits.
        static std::vector< std::vector<bool> > getInputPermutations(const unsigned int numBits) {
            const int numPermutations = (1 << numBits);
            std::vector< std::vector<bool> > bitStrings;
            for (int i = 0; i < numPermutations; i++) {
                bitStrings.push_back(intToBinary(i, numBits));
            }
            return bitStrings;
        }
        
        // n-arity exclusive or.  True iff exactly one bit is true.
        static bool nXor(const std::vector<bool> bitString) {
            int sum = 0;
            for(int i = 0; i < bitString.size(); i++) {
                sum += bitString[i];
            }
            return 1 == sum;
        }
    };
}

int main(int argc, char* argv[]) {
    srand ( time(NULL) );
    // The first argument is the number of inputs the network receives, the second is whether to print the time series.
    const IzkExperiment experiment(atoi(argv[1]), atoi(argv[2]));
    
    // Loads the ParameterInstances from std::cin and use them to execute the experiment.
    const PTI pti(argc, argv, std::cout, std::cin);
    pti.runExperiment(experiment);

    return 0;
}
