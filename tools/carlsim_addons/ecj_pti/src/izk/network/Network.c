#include "Network.h"
#include "ConnectionGroup.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

static const double VOLTAGE_OFFSET = 70.0; // Used in the neurotransmitter model.

Network * izk_makeNetwork(ConnectionGroup * const * const connectionGroups, const unsigned int numConnectionGroups) {
    assert(connectionGroups != NULL);
    
    const Network stackNetwork = { connectionGroups, numConnectionGroups };
    Network * const network = malloc(sizeof(*network));
    memcpy(network, &stackNetwork, sizeof(*network));
    return network;
}

/** Compute the current of a single neuron as a function of its voltage and the
 * conductances at each of its synapses. */
double getCurrent(const ConnectionGroup * const cGroup, unsigned int neuron) {
    assert(cGroup != NULL);
    assert(neuron < cGroup->destGroup->length);
    
    const double postSynapticVoltage = cGroup->destGroup->neuronArray[neuron]->voltage;
    
    int i;
    double current = 0.0;
    for (i = 0; i < cGroup->sourceGroup->length; i++) {
        if (izk_ConnectionGroup_isConnected(cGroup, i, neuron))
            current += izk_ConnectionGroup_getConductance(cGroup, i, neuron) * (postSynapticVoltage + VOLTAGE_OFFSET);
    }
    return current;
}

/** Determing the transmembrane current in a group of postsynaptic neurons, based
 * on its current voltage and the neurotransmitter-modulated conductance of each synapse. */
void getCurrents(const ConnectionGroup * const cGroup, double* currents) {
    assert(cGroup != NULL);
    assert(currents != NULL);
    
    int i;
    for (i = 0; i < cGroup->destGroup->length; i++) {
        currents[i] = getCurrent(cGroup, i);
    }
}

/** Steps all neurons and connections. */
void izk_Network_step(const Network* network, const double* const inputCurrents, const double stepSize) {
    assert(network != NULL);
    assert(stepSize > 0.0);
    
    izk_NeuronGroup_stepAll(network->connectionGroups[0]->sourceGroup, inputCurrents, stepSize);
    int i;    
    for (i = 0; i < network->numConnectionGroups; i++) {
        ConnectionGroup* const cGroup = network->connectionGroups[i];
        
        // Compute the input currents to the destination layer.
        double currents[cGroup->destGroup->length];
        getCurrents(cGroup, currents);
        
        // Produce output voltages from each neuron.
        izk_NeuronGroup_stepAll(cGroup->destGroup, currents, stepSize);
        // Update the conductances (which determine the input currents in the next step).
        izk_ConnectionGroup_step(cGroup, stepSize);
    }
}

void izk_Network_free(const Network* network) {
    // Casting away constness because I don't interpret freeing memory as modifying it.
    free((void*) network);
}
