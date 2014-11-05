#ifndef CONNECTION_SCHEME_H
#define CONNECTION_SCHEME_H

#ifdef __cplusplus
extern "C" {
#endif
    
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>


/** Instead of using a callback function to define connectivity (like CARLsim),
* Izk uses a POD class called ConnectionScheme.  Subclasses of ConnectionScheme
* may generate connections and weights randomly, or may store additional data
* obtained by, say, loading a weight matrix from a file.
*  We do this by giving implementations
* of ConnectionScheme's virtual methods and writing a constructor that
* populates the function table of a ConnectionScheme.
*/
typedef struct ConnectionScheme {
    /** True iff the ith neuron of the source group is connected to the jth
     * neuron of the destination group. */
    bool (* const isConnected)(const struct ConnectionScheme * obj, unsigned int i, unsigned int j);
    /** The weight of the connection between the ith neuron of the source group
     * and the jth neuron of the destination group.  This value is meaningless
     * if isConnected(i, j) is false. */
    double (* const getWeight)(const struct ConnectionScheme * obj, unsigned int i, unsigned int j);
    /** Destructor. */
    void (* const free)(const struct ConnectionScheme * obj);
    // Subtype-specific data goes here.
    void * const implementationData;
} ConnectionScheme;



/** Construct a ConnectionScheme on the heap that connects every neuron i to
 * every neuron j, and assigns it a weight between minWeight and maxWeight
 * (inclusive). Just set minWeight and maxWeight to the same value if you want
 * uniform weights. */
ConnectionScheme* izk_makeFullConnectionScheme(double minWeight, double maxWeight);

/** Construct a ConnectionScheme on the heap that connects a neuron i to neuron
 * j with probability p, and assigns it a weight between minWeight and maxWeight
 * (inclusive). */
ConnectionScheme* izk_makeRandomConnectionScheme(double p, double minWeight, double maxWeight);

/** Construct a ConnectionScheme on the heap that connects every neuron i to
 * every neuron j, and assigns it the weight found in weightMatrix[i*numColumns + j]. */
ConnectionScheme* izk_makeCustomConnectionScheme(const double * weightMatrix, const unsigned int numRows, const unsigned int numColumns);


#ifdef __cplusplus
}
#endif

#endif // CONNECTION_SCHEME_H
