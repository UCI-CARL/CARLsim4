#ifndef POISSON_NEURON_H
#define POISSON_NEURON_H

#ifdef __cplusplus
extern "C" {
#endif
    
#include "Neuron.h"
#include <stdbool.h>

typedef struct PoissonNeuronImplementationData {
    const double spikeRate; // In spikes per millisecond, i.e. kHz.
    const double threshold; // We start spiking if input current passes this value.
    double timeSinceLastSpike;
    double nextSpikeTime;
    bool on;
} PoissonNeuronImplementationData;

/** Constructor.  Allocates a new neuron on the heap.
 * @param spikeRate The Poisson spike rate in kHz. 
 * @param threshold We don't spike at all if the input current is below this value. */
Neuron * izk_makePoissonNeuron (const double spikeRate, const double threshold);
/** Copy constructor.  ref must be a PoissonNeuron. */
Neuron * izk_copyPoissonNeuron (const Neuron * ref);

// Function table for public methods.  Implementation-specific public methods can be added here.
typedef struct PoissonNeuronFunctionTable {
    void (*step)(Neuron * obj, double inputCurrent, double stepSize);
    bool (*repOK)(const Neuron * obj);
    Neuron* (* const copy)(const Neuron * obj); // Copy constructor
    void (* const free)(const Neuron * obj); // Destructor
} PoissonNeuronFunctionTable;

#ifdef __cplusplus
}
#endif

#endif // POISSON_NEURON_H
