/** Izk.c
 *
 * Main method for a very simple spiking neural network simulator.
 *
 * @author Eric 'Siggy' Scott
 */

#include "neuron/Neuron.h"
#include "neuron/Izhikevich4Neuron.h"
#include "neuron/PoissonNeuron.h"
#include "network/Network.h"
#include "network/NeuronGroup.h"
#include "network/ConnectionGroup.h"
#include "network/ConnectionScheme.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static const double INPUT_CURRENT = -99.0;
static const double STEP_SIZE = 0.1;
static const double MAX_WEIGHT = 1.0;

/** Run two Izhikevich neurons to see if they work okay. */
void izk_runTwoNeurons() {
    const Neuron * const prototype = izk_makeIzhikevich4Neuron(0.2, 2.0, -56.0, -16.0);
    const NeuronGroup * const neuronGroup = izk_makeNeuronGroup(prototype, 2);

    const double inputCurrents[2] = {INPUT_CURRENT, INPUT_CURRENT };
    printf("v1, u1, v2, u2\n");
    int i;
    for (i = 0; i < 5000; i++) {
        izk_NeuronGroup_stepAll(neuronGroup, inputCurrents, STEP_SIZE);
        const double v1 = neuronGroup->neuronArray[0]->voltage;
        const double u1 = ((Izhikevich4NeuronImplementationData*)neuronGroup->neuronArray[0]->implementationData)->recovery;
        const double v2 = neuronGroup->neuronArray[1]->voltage;
        const double u2 = ((Izhikevich4NeuronImplementationData*)neuronGroup->neuronArray[1]->implementationData)->recovery;
        printf("%f, %f, %f, %f\n", v1, u1, v2, u2);
    }
}

/** Run two PoissonNeurons, one off and one on, to see if they're working. */
void izk_runTwoPoissonNeurons() {
    srand(10000);
    printf("Building network. . . ");
    const double poissonSpikeRate = 1.0; // 1 kHz -- far higher than biologically plausible, but easy to check in the output.
    const double poissonSpikeThreshold = 30; // 30 mV
    const double inputCurrents[] = { 0.0, 40.0 };
    
    const Neuron * const prototype = izk_makePoissonNeuron(poissonSpikeRate, poissonSpikeThreshold);
    const int numNeurons = 2;
    const NeuronGroup * const neuronGroup = izk_makeNeuronGroup(prototype, numNeurons);
    
    printf("s1, t1, s2, t2\n");
    int i;
    for (i = 0; i < 5000; i++) {
        izk_NeuronGroup_stepAll(neuronGroup, inputCurrents, STEP_SIZE);
        const bool s1 = neuronGroup->neuronArray[0]->isSpiking;
        const PoissonNeuronImplementationData * const n1Imp = ((PoissonNeuronImplementationData*)neuronGroup->neuronArray[0]->implementationData);
        const double t1 = n1Imp->nextSpikeTime - n1Imp->timeSinceLastSpike;
        
        const bool s2 = neuronGroup->neuronArray[1]->isSpiking;
        const PoissonNeuronImplementationData * const n2Imp = ((PoissonNeuronImplementationData*)neuronGroup->neuronArray[1]->implementationData);
        const double t2 = n2Imp->nextSpikeTime - n2Imp->timeSinceLastSpike;
        printf("%d, %f, %d, %f\n", s1, t1, s2, t2);
    }
}

/** Run a pair of layers and manually propagate signals between them.
 * This is to show that NeuronGroup and ConnectionGroup are roughly working. */
void izk_runSingleLayerNetwork() {
    srand(10000);
    printf("Building network. . . ");
    const Neuron * const prototype = izk_makeIzhikevich4Neuron(0.2, 2.0, -56.0, -16.0); // Chaotic
    const int numNeuronsLayer1 = 100;
    const int numNeuronsLayer2 = 4;
    const NeuronGroup * const sourceGroup = izk_makeNeuronGroup(prototype, numNeuronsLayer1);
    const NeuronGroup * const destGroup = izk_makeNeuronGroup(prototype, numNeuronsLayer2);
    const ConnectionScheme * const scheme = izk_makeRandomConnectionScheme(0.5, 0.0, MAX_WEIGHT);
    ConnectionGroup * const connections = izk_makeConnectionGroup(scheme, 0.0, sourceGroup, destGroup);
    
    izk_ConnectionGroup_printConnections(connections);
    izk_ConnectionGroup_printWeights(connections);
    
    double inputCurrents[numNeuronsLayer1];
    int i;
    for (i = 0; i < numNeuronsLayer1; i++) {
        inputCurrents[i] = INPUT_CURRENT;
    }
    
    printf("v1, u1, v2, u2, v3, u3, v4, u4\n");
    for (i = 0; i < 5000; i++) {
        izk_NeuronGroup_stepAll(sourceGroup, inputCurrents, STEP_SIZE);
        double layerTwoInputs[4] = { 0.0, 0.0, 0.0, 0.0 };
        int j, k;
        for (j = 0; j < sourceGroup->length; j++) {
            for (k = 0; k < destGroup->length; k++) {
                if (izk_ConnectionGroup_isConnected(connections, j, k))
                    layerTwoInputs[k] += ((sourceGroup->neuronArray[j]->isSpiking) ? 1 : 0);
            }
        }
        izk_NeuronGroup_stepAll(destGroup, layerTwoInputs, STEP_SIZE);
        const double v1 = destGroup->neuronArray[0]->voltage;
        const double u1 = ((Izhikevich4NeuronImplementationData*)destGroup->neuronArray[0]->implementationData)->recovery;
        const double v2 = destGroup->neuronArray[1]->voltage;
        const double u2 = ((Izhikevich4NeuronImplementationData*)destGroup->neuronArray[1]->implementationData)->recovery;
        const double v3 = destGroup->neuronArray[2]->voltage;
        const double u3 = ((Izhikevich4NeuronImplementationData*)destGroup->neuronArray[2]->implementationData)->recovery;
        const double v4 = destGroup->neuronArray[3]->voltage;
        const double u4 = ((Izhikevich4NeuronImplementationData*)destGroup->neuronArray[3]->implementationData)->recovery;
        printf("%f, %f, %f, %f, %f, %f, %f, %f\n", v1, u1, v2, u2, v3, u3, v4, u4);
    }
    
    prototype->funcTable->free(prototype);
    izk_NeuronGroup_free(sourceGroup);
    izk_NeuronGroup_free(destGroup);
    izk_ConnectionGroup_free(connections);
}

void izk_runSimpleNetwork(const bool input[], const unsigned int numInputs, const unsigned int numHiddenNeurons, const unsigned int numOutputs) {
    assert(input != NULL);
    const double poissonSpikeRate = 0.010; // kHz
    const double poissonSpikeThreshold = 30; // mV
    const double conductanceDecayRate = 50; // mS
    
    double inputCurrents[numInputs];
    int i;
    for (i = 0; i < numInputs; i++)
        inputCurrents[i] = input[i] ? poissonSpikeThreshold : 0.0;
    // Set up neurons
    const Neuron* poissonPrototype = izk_makePoissonNeuron(poissonSpikeRate, poissonSpikeThreshold);
    const Neuron * const izhikevichPrototype = izk_makeIzhikevich4Neuron(0.1, 0.2, -65.0, 2.0); // Fast spiking
    const NeuronGroup* const inputGroup = izk_makeNeuronGroup(poissonPrototype, numInputs);
    const NeuronGroup* const hiddenGroup = izk_makeNeuronGroup(izhikevichPrototype, numHiddenNeurons);
    const NeuronGroup* const outputGroup = izk_makeNeuronGroup(izhikevichPrototype, numOutputs);
    // Set up connections
    const ConnectionScheme * const scheme = izk_makeFullConnectionScheme(MAX_WEIGHT, MAX_WEIGHT);
    ConnectionGroup* const connections[] = {
        izk_makeConnectionGroup(scheme, conductanceDecayRate, inputGroup, hiddenGroup),
        izk_makeConnectionGroup(scheme, conductanceDecayRate, hiddenGroup, outputGroup)
    };
    const Network* const network = izk_makeNetwork(connections, 2);
    
    printf("ms, p1_spike, p2_spike, h1_voltage, h2_voltage, h1_spike, h2_spike, o_voltage, o_spike\n");
    for (i = 0; i < 5000; i++) {
        izk_Network_step(network, inputCurrents, STEP_SIZE);
        printf("%f, %s, ", i*STEP_SIZE, izk_NeuronGroup_printIsSpiking(inputGroup));
        printf("%s, ", izk_NeuronGroup_printVoltage(hiddenGroup));
        printf("%s, ", izk_NeuronGroup_printIsSpiking(hiddenGroup));
        printf("%s, ", izk_NeuronGroup_printVoltage(outputGroup));
        printf("%s\n", izk_NeuronGroup_printIsSpiking(outputGroup));
    }
    
    // Free everything that was created with a "make" function.
    izk_Network_free(network);
    izk_ConnectionGroup_free(connections[0]);
    izk_ConnectionGroup_free(connections[1]);
    scheme->free(scheme);
    izk_NeuronGroup_free(inputGroup);
    izk_NeuronGroup_free(hiddenGroup);
    izk_NeuronGroup_free(outputGroup);
    poissonPrototype->funcTable->free(poissonPrototype);
    izhikevichPrototype->funcTable->free(izhikevichPrototype);
}

void parseArguments(int argc, char * argv[], bool ** const inputs, unsigned int * const numInputs, unsigned int * const numHiddenNeurons, unsigned int * const numOutputs) {
    assert(argc > 0);
    assert(argv != NULL);
    assert(numInputs != NULL);
    assert(numHiddenNeurons != NULL);
    assert(numOutputs != NULL);
    
    *numInputs = atoi(argv[1]);
    assert(*numInputs + 4 == argc);
    *numHiddenNeurons = atoi(argv[2]);
    *numOutputs = atoi(argv[3]);
    *inputs = malloc(*numInputs * sizeof(bool));
    int i;
    for (i = 0; i < *numInputs; i++) {
        (*inputs)[i] = atoi(argv[i+4]);
    }
}


int main(int argc, char * argv[]) {
    int numInputs, numHiddenNeurons, numOutputs;
    bool * inputs;
    parseArguments(argc, argv, &inputs, &numInputs, &numHiddenNeurons, &numOutputs);
    //printf("numInputs: %d\nnumHiddenNeurons: %d\nnumOutputs: %d\n", numInputs, numHiddenNeurons, numOutputs);
    //izk_runTwoNeurons();
    //izk_runSingleLayerNetwork();
    //izk_runTwoPoissonNeurons();
    izk_runSimpleNetwork(inputs, numInputs, numHiddenNeurons, numOutputs);
    
    free(inputs);
    return 0;
}
