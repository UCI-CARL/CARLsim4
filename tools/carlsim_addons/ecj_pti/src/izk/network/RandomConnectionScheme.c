#include "ConnectionScheme.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

/** This file defines a concrete implementation of the abstract class
 * ConnectionScheme.  See the documentation for the constructor in
 * ConnectionScheme.h. */

typedef struct RandomConnectionSchemeImplementationData {
    const double p;
    const double minWeight;
    const double maxWeight;
} RandomConnectionSchemeImplementationData;

bool izk_RandomConnectionScheme_isConnected(const ConnectionScheme * const obj, const unsigned int i, const unsigned int j) {
    assert(obj != NULL);
    const RandomConnectionSchemeImplementationData * const impl = (RandomConnectionSchemeImplementationData*) obj->implementationData;
    assert(impl != NULL);
    const double coinToss = (double)rand() / (double)RAND_MAX;
    return (coinToss <= impl->p);
}

double izk_RandomConnectionScheme_getWeight(const ConnectionScheme * const obj, const unsigned int i, const unsigned int j) {
    assert(obj != NULL);
    const RandomConnectionSchemeImplementationData* impl = (RandomConnectionSchemeImplementationData*) obj->implementationData;
    assert(impl != NULL);
    const double range = impl->maxWeight - impl->minWeight;
    assert(range >= 0);
    const double coinToss = (double)rand() / (double)RAND_MAX;
    return impl->minWeight + coinToss*range;
}

void izk_RandomConnectionScheme_free(const ConnectionScheme* obj) {
    assert(obj != NULL);
    free((void*) obj->implementationData);
    free((void*) obj);
}

ConnectionScheme* izk_makeRandomConnectionScheme(const double p, const double minWeight, const double maxWeight) {
    assert(p >= 0.0);
    assert(p <= 1.0);
    assert(minWeight <= maxWeight);
    // A two-step construction is required to initialize const members of a struct on the heap.
    const RandomConnectionSchemeImplementationData stackImpl = { p, minWeight, maxWeight };
    RandomConnectionSchemeImplementationData * const heapImpl = malloc(sizeof(*heapImpl));
    memcpy(heapImpl, &stackImpl, sizeof(*heapImpl));
    
    const ConnectionScheme stackResult = {
        &izk_RandomConnectionScheme_isConnected,
        &izk_RandomConnectionScheme_getWeight,
        &izk_RandomConnectionScheme_free,
        heapImpl
    };
    ConnectionScheme* const heapResult = malloc(sizeof(*heapResult));
    memcpy(heapResult, &stackResult, sizeof(*heapResult));
    return heapResult;
}
