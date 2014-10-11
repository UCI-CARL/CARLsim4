#include "ConnectionGroup.h"
#include "NeuronGroup.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/** ConnectionGroup.h
 * 
 * A class which constructs and stores synapses between two NeuronGroups.
 *
 * @author Eric 'Siggy' Scott
 */

/** Generate an adjacency matrix from its functional representation.  Note that
 * the resulting matrix may be very large, even if the connections are sparse. */
void izk_ConnectionGroup_connectionSchemeToAdjacencyMatrix(const ConnectionScheme * const connectionScheme, const int sourceLength, const int destLength, bool** const adjacencyMatrix, double** const weightMatrix) {
    assert(connectionScheme != NULL);
    assert(sourceLength > 0);
    assert(destLength > 0);
    
    bool * const adj = malloc(sourceLength * destLength * sizeof(*adj));
    double * const wgt = malloc(sourceLength * destLength * sizeof(*wgt));
    int i, j;
    for (i = 0; i < sourceLength; i++) {
        for (j = 0; j < destLength; j++) {
            adj[i*destLength + j] = connectionScheme->isConnected(connectionScheme, i, j);
            wgt[i*destLength + j] = connectionScheme->getWeight(connectionScheme, i, j);
        }
    }
    *adjacencyMatrix = adj;
    *weightMatrix = wgt;
    
}

/** Constructor. */
ConnectionGroup * izk_makeConnectionGroup(const ConnectionScheme * const connectionScheme, const double conductanceDecayRate, const NeuronGroup* const sourceGroup, const NeuronGroup* const destGroup) {
    assert(connectionScheme != NULL);
    assert(conductanceDecayRate > 0.0);
    assert(sourceGroup != NULL);
    assert(destGroup != NULL);
    bool* adjacencyMatrix;
    double* weightMatrix;
    izk_ConnectionGroup_connectionSchemeToAdjacencyMatrix(connectionScheme, sourceGroup->length, destGroup->length, &adjacencyMatrix, &weightMatrix);
    
    double * const conductances = malloc(sourceGroup->length * destGroup->length * sizeof(*conductances));
    memset(conductances, 0, sourceGroup->length * destGroup->length * sizeof(*conductances));
    
     // Initialize on the stack and copy to heap, because struct ConnectionGroup has const members
    const ConnectionGroup stackConnectionGroup = { sourceGroup, destGroup, adjacencyMatrix, weightMatrix, conductances, conductanceDecayRate};
    ConnectionGroup* const connectionGroup = malloc(sizeof(*connectionGroup));
    memcpy(connectionGroup, &stackConnectionGroup, sizeof(ConnectionGroup));
    return connectionGroup;
}

void izk_ConnectionGroup_step(const ConnectionGroup * const obj, const double stepSize) {
    assert(obj != NULL);
    assert(stepSize > 0.0);
    int i, j;
    for (i = 0; i < obj->sourceGroup->length; i++) {
        for (j = 0; j < obj->destGroup->length; j++) {
            // Execute the decay equation for conductance
            const double g = obj->conductanceMatrix[i*obj->destGroup->length + j];
            obj->conductanceMatrix[i*obj->destGroup->length + j] -= g/obj->conductanceDecayRate * stepSize; // FIXME This line is somehow writing to obj->adjacencyMatrix[0]
            // If the presynaptic neuron is spiking, the conductance gets a boost
            if (obj->sourceGroup->neuronArray[i]->isSpiking) {
                obj->conductanceMatrix[i*obj->destGroup->length + j] += obj->weightMatrix[i*obj->destGroup->length + j];
            }
        }
    }
}

/** Returns true if neuron i from sourceGroup feeds into neuron j from destGroup. */
bool izk_ConnectionGroup_isConnected(const ConnectionGroup * const obj, unsigned int i, unsigned int j) {
    assert(obj != NULL);
    assert(i < obj->sourceGroup->length);
    assert(j < obj->destGroup->length);
    const bool result = obj->adjacencyMatrix[i*obj->destGroup->length + j];
    return result;
}

/** Returns the weight of the connection between i and j.  This value is not meaningful if i and j are not connected. */
double izk_ConnectionGroup_getWeight(const ConnectionGroup * const obj, unsigned int i, unsigned int j) {
    assert(obj != NULL);
    assert(i < obj->sourceGroup->length);
    assert(j < obj->destGroup->length);
    return obj->weightMatrix[i*obj->destGroup->length + j];
}


double izk_ConnectionGroup_getConductance(const ConnectionGroup * const obj, unsigned int i, unsigned int j) {
    assert(obj != NULL);
    assert(i < obj->sourceGroup->length);
    assert(j < obj->destGroup->length);
    return obj->conductanceMatrix[i*obj->destGroup->length + j];
    
}

void izk_ConnectionGroup_printConnections(const ConnectionGroup * const obj) {
    assert(obj != NULL);
    int i, j;
    for (i = 0; i < obj->sourceGroup->length; i++) {
        for (j = 0; j < obj->destGroup->length; j++) {
            printf("%d\t", (obj->adjacencyMatrix[i*obj->destGroup->length + j] ? 1 : 0));
        }
        printf("\n");
    }
}

void izk_ConnectionGroup_printWeights(const ConnectionGroup * const obj) {
    assert(obj != NULL);
    int i, j;
    for (i = 0; i < obj->sourceGroup->length; i++) {
        for (j = 0; j < obj->destGroup->length; j++) {
            printf("%f\t", obj->weightMatrix[i*obj->destGroup->length + j]);
        }
        printf("\n");
    }
}

/** Destructor. */
void izk_ConnectionGroup_free(const ConnectionGroup * const obj) {
    assert(obj != NULL);
    free(obj->conductanceMatrix);
    // Casting away constness here, because I don't interpret deallocating memory as modifying it.
    free((void*) obj->adjacencyMatrix);
    free((void*) obj->weightMatrix);
    free((void*) obj);
}
