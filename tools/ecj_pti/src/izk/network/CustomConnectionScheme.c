#include "ConnectionScheme.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

/** This file defines a concrete implementation of the abstract class
 * ConnectionScheme.  See the documentation for the constructor in
 * ConnectionScheme.h. */

typedef struct CustomConnectionSchemeImplementationData {
    const double * const weightMatrix;
    const unsigned int numRows;
    const unsigned int numColumns;
} CustomConnectionSchemeImplementationData;

bool izk_CustomConnectionScheme_isConnected(const ConnectionScheme * const obj, const unsigned int i, const unsigned int j) {
    assert(obj != NULL);
    return true;
}

double izk_CustomConnectionScheme_getWeight(const ConnectionScheme * const obj, const unsigned int i, const unsigned int j) {
    assert(obj != NULL);
    const CustomConnectionSchemeImplementationData * const impl = (CustomConnectionSchemeImplementationData*) obj->implementationData;
    assert(impl != NULL);
    const int index = i * impl->numColumns + j;
    assert(index < impl->numRows * impl->numColumns);
    return impl->weightMatrix[index];
}

void izk_CustomConnectionScheme_free(const ConnectionScheme* obj) {
    assert(obj != NULL);
    free((void*) obj->implementationData);
    free((void*) obj);
}

ConnectionScheme* izk_makeCustomConnectionScheme(const double * weightMatrix, const unsigned int numRows, const unsigned int numColumns) {
    assert(weightMatrix != NULL);
    // A two-step construction is required to initialize const members of a struct on the heap.
    const CustomConnectionSchemeImplementationData stackImpl = {weightMatrix, numRows, numColumns};
    CustomConnectionSchemeImplementationData * const heapImpl = malloc(sizeof (*heapImpl));
    memcpy(heapImpl, &stackImpl, sizeof (*heapImpl));

    const ConnectionScheme stackResult = {
        &izk_CustomConnectionScheme_isConnected,
        &izk_CustomConnectionScheme_getWeight,
        &izk_CustomConnectionScheme_free,
        heapImpl
    };
    ConnectionScheme * const heapResult = malloc(sizeof (*heapResult));
    memcpy(heapResult, &stackResult, sizeof (*heapResult));
    return heapResult;
}
