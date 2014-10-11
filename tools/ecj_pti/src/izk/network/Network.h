#ifndef NETWORK_H
#define NETWORK_H

#ifdef __cplusplus
extern "C" {
#endif
    
#include "ConnectionGroup.h"

typedef struct Network {
    ConnectionGroup* const * const connectionGroups;
    const unsigned int numConnectionGroups;
} Network;

/** Constructor.  Creates a Network on the heap.
 * @param connectionGroups An array of pointers to the ConnectionGroups that make up the network. 
 * @param numConnectionGroups The length of the connectionGroups array. */
Network * izk_makeNetwork(ConnectionGroup * const * const connectionGroups, const unsigned int numConnectionGroups);

void izk_Network_step(const Network* network, const double* inputCurrents, double stepSize);

// Destructor
void izk_Network_free(const Network* network);

#ifdef __cplusplus
}
#endif

#endif //NETWORK_H
