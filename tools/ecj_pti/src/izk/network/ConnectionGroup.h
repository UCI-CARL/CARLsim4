#ifndef IZK_CONNECTION_GROUP_H
#define IZK_CONNECTION_GROUP_H

#ifdef __cplusplus
extern "C" {
#endif
    
#include "ConnectionScheme.h"
#include "NeuronGroup.h"
#include <stdbool.h>

/** ConnectionGroup.h
 * 
 * A class which constructs and stores synapses between two NeuronGroups.
 *
 * @author Eric 'Siggy' Scott
 */

typedef struct ConnectionGroup {
    const NeuronGroup* const sourceGroup;
    const NeuronGroup* const destGroup;
    const bool * const adjacencyMatrix; // adjacencyMatrix[x*destGroup->length+y] = true iff x -> y
    const double * const weightMatrix; // weightMatrix[x*destGroup->length+y] is the weight of x -> y, or contains an arbitrary value if x and y are not connected.
    double * const conductanceMatrix;
    double conductanceDecayRate;
} ConnectionGroup;

/** Constructor. Creates a ConnectionGroup on the heap. 
 * @param connectionFunction A callback function defining which neurons are connected and the strength of their weights. 
 * @param sourceGroup The group of presynaptic neurons.
 * @param destGroup the group of postsynaptic neurons. */
ConnectionGroup* izk_makeConnectionGroup(const ConnectionScheme * const connectionFunction, double conductanceDecayRate, const NeuronGroup* const sourceGroup, const NeuronGroup* const destGroup);

void izk_ConnectionGroup_step(const ConnectionGroup * const obj, double stepSize);

/** Returns true if neuron i from sourceGroup feeds into neuron j from destGroup. */
bool izk_ConnectionGroup_isConnected(const ConnectionGroup * const obj, unsigned int i, unsigned int j);
/** Returns the weight of the connection between i and j.  This value is not meaningful if i and j are not connected. */
double izk_ConnectionGroup_getWeight(const ConnectionGroup * const obj, unsigned int i, unsigned int j);

double izk_ConnectionGroup_getConductance(const ConnectionGroup * const obj, unsigned int i, unsigned int j);

void izk_ConnectionGroup_printConnections(const ConnectionGroup * const obj);

void izk_ConnectionGroup_printWeights(const ConnectionGroup * const obj);

/** Destructor. */
void izk_ConnectionGroup_free(const ConnectionGroup * const obj);

#ifdef __cplusplus
}
#endif

#endif /* IZK_CONNECTION_GROUP_H */
