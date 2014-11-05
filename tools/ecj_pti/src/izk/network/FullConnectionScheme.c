#include "ConnectionScheme.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

/** This file defines a concrete implementation of the abstract class
 * ConnectionScheme.  See the documentation for the constructor in
 * ConnectionScheme.h. */

typedef struct FullConnectionSchemeImplementationData {
    const double minWeight;
    const double maxWeight;
} FullConnectionSchemeImplementationData;

bool izk_FullConnectionScheme_isConnected(const ConnectionScheme * const obj, const unsigned int i, const unsigned int j) {
    assert(obj != NULL);
    return true;
}

double izk_FullConnectionScheme_getWeight(const ConnectionScheme * const obj, const unsigned int i, const unsigned int j) {
    assert(obj != NULL);
    const FullConnectionSchemeImplementationData * const impl = (FullConnectionSchemeImplementationData*) obj->implementationData;
    assert(impl != NULL);
    const double range = impl->maxWeight - impl->minWeight;
    assert(range >= 0);
    const double coinToss = (double)rand() / (double)RAND_MAX;
    return impl->minWeight + coinToss*range;
}

void izk_FullConnectionScheme_free(const ConnectionScheme* obj) {
    assert(obj != NULL);
    free((void*) obj->implementationData);
    free((void*) obj);
}

ConnectionScheme* izk_makeFullConnectionScheme(const double minWeight, const double maxWeight) {
    assert(minWeight <= maxWeight);
    // A two-step construction is required to initialize const members of a struct on the heap.
    const FullConnectionSchemeImplementationData stackImpl = { minWeight, maxWeight };
    FullConnectionSchemeImplementationData * const heapImpl = malloc(sizeof(*heapImpl));
    memcpy(heapImpl, &stackImpl, sizeof(*heapImpl));
    
    const ConnectionScheme stackResult = {
        &izk_FullConnectionScheme_isConnected,
        &izk_FullConnectionScheme_getWeight,
        &izk_FullConnectionScheme_free,
        heapImpl
    };
    ConnectionScheme* const heapResult = malloc(sizeof(*heapResult));
    memcpy(heapResult, &stackResult, sizeof(*heapResult));
    return heapResult;
}
