#include "Neuron.h"
#include "Izhikevich4Neuron.h"

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/** Izhikevich4.c
 *
 * A 4-parameter Izhikevich neuron.
 *
 * @author Eric 'Siggy' Scott
 */

static const double SPIKE_THRESHOLD = 30.0;

// Forward-declare members of the public function table.
void izk_izhikevich4_step(Neuron * const neuron, const double inputCurrent, const double stepSize);
bool izk_izhikevich4_repOK(const Neuron * const neuron);
void izk_izhikevich4_free(const Neuron * const neuron);

static const struct Izhikevich4NeuronFunctionTable izhikevich4NeuronFunctionTable = {
    &izk_izhikevich4_step,
    &izk_izhikevich4_repOK,
    &izk_copyIzhikevich4Neuron,
    &izk_izhikevich4_free
};

/** Constructor.  Allocates a new neuron on the heap. */
Neuron * izk_makeIzhikevich4Neuron(const double a, const double b, const double c, const double d) {
    // Initialize implementation-specific data
    const Izhikevich4NeuronImplementationData const stackRdata = {a, b, c, d, 0.0}; // Build on the stack first to set up const members
    Izhikevich4NeuronImplementationData * const rdata = malloc(sizeof *rdata);
    memcpy(rdata, &stackRdata, sizeof (*rdata));

    // Initialize public data
    const Neuron const stackNeuron = {0.0, false, (const struct NeuronFunctionTable*) &izhikevich4NeuronFunctionTable, rdata};
    Neuron * const neuron = malloc(sizeof *neuron);
    memcpy(neuron, &stackNeuron, sizeof (Neuron));

    return neuron;
}

/** Copy constructor.  ref must be an Izhikevich4Neuron. */
Neuron * izk_copyIzhikevich4Neuron(const Neuron * const ref) {
    assert(ref != NULL);
    const Izhikevich4NeuronImplementationData * const pNeuron = (Izhikevich4NeuronImplementationData*) ref->implementationData;

    Neuron * const neuron = izk_makeIzhikevich4Neuron(pNeuron->a, pNeuron->b, pNeuron->c, pNeuron->d);
    ((Izhikevich4NeuronImplementationData * const) neuron->implementationData)->recovery = pNeuron->recovery;
    neuron->voltage = ref->voltage;
    neuron->isSpiking = ref->isSpiking;
    return neuron;
}

void izk_izhikevich4_step(Neuron * const neuron, const double inputCurrent, const double stepSize) {
    assert(neuron != NULL);
    assert(stepSize > 0);
    Izhikevich4NeuronImplementationData * const ikzData = (Izhikevich4NeuronImplementationData * const) neuron->implementationData;

    // The Izhikevich equations
    neuron->voltage += stepSize * (0.04 * pow(neuron->voltage, 2) + 5 * neuron->voltage + 140 - ikzData->recovery + inputCurrent);
    ikzData->recovery += stepSize * (ikzData->a * (ikzData->b * neuron->voltage - ikzData->recovery));

    // The nonlinear reset condition  
    if (neuron->voltage >= 30.0) {
        neuron->isSpiking = true;
        neuron->voltage = ikzData->c;
        ikzData->recovery = ikzData->recovery + ikzData->d;
    }
    else {
        neuron->isSpiking = false;
    }

    assert(izk_izhikevich4_repOK(neuron));
}

bool izk_izhikevich4_repOK(const Neuron * const neuron) {
    assert(neuron != NULL);
    return neuron->voltage < 30.0;
}

/** Destructor. */
void izk_izhikevich4_free(const Neuron * const neuron) {
    assert(neuron != NULL);
    free(neuron->implementationData); // Free the subclass data
    izk_Neuron_free(neuron); // Free the superclass data
}
