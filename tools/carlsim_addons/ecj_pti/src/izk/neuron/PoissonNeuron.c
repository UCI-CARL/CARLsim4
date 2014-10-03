#include "PoissonNeuron.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Forward-declare members of the public function table.
void izk_PoissonNeuron_step(Neuron* const neuron, const double inputCurrent, const double stepSize);
bool izk_PoissonNeuron_repOK(const Neuron* const neuron);
void izk_PoissonNeuron_free(const Neuron* const neuron);

static const struct PoissonNeuronFunctionTable poissonNeuronFunctionTable = {
    &izk_PoissonNeuron_step,
    &izk_PoissonNeuron_repOK,
    &izk_copyPoissonNeuron,
    &izk_PoissonNeuron_free
};

double izk_PoissonNeuron_getNewSpikeTime(const double spikeRate) {
    assert(spikeRate >= 0.0);
    const double uRand = (double)rand() / (double)RAND_MAX * 1.0;
    return -log(1.0 - uRand)/spikeRate;
}

/** Constructor.  Allocates a new neuron on the heap. */
Neuron * izk_makePoissonNeuron (const double spikeRate, const double threshold) {
    // Initialize implementation-specific data
    // Build on the stack first to set up const members
    const PoissonNeuronImplementationData const stackRdata = { spikeRate, threshold, 0.0, izk_PoissonNeuron_getNewSpikeTime(spikeRate), false};
    PoissonNeuronImplementationData * const rdata = malloc (sizeof *rdata);
    memcpy(rdata, &stackRdata, sizeof(*rdata));
    
    // Initialize supertype
    const Neuron const stackNeuron = { 0.0, false, (const struct NeuronFunctionTable*) &poissonNeuronFunctionTable, rdata };
    Neuron * const neuron = malloc (sizeof *neuron);
    memcpy(neuron, &stackNeuron, sizeof(Neuron));

    assert(izk_PoissonNeuron_repOK(neuron));
    return neuron;
}

/** Copy constructor.  ref must be a PoissonNeuron. */
Neuron * izk_copyPoissonNeuron (const Neuron * const ref) {
    const PoissonNeuronImplementationData * const pNeuron = (PoissonNeuronImplementationData*) ref->implementationData;
    
    Neuron * const neuron = izk_makePoissonNeuron(pNeuron->spikeRate, pNeuron->threshold);
    PoissonNeuronImplementationData* const nImpl = ((PoissonNeuronImplementationData* const) neuron->implementationData);
    nImpl->timeSinceLastSpike = pNeuron->timeSinceLastSpike;
    nImpl->nextSpikeTime = pNeuron->nextSpikeTime;
    nImpl->on = pNeuron->on;
    neuron->voltage = ref->voltage;
    neuron->isSpiking = ref->isSpiking;
    return neuron;
}

void izk_PoissonNeuron_step(Neuron* const neuron, const double inputCurrent, const double stepSize) {
  assert(stepSize > 0);
  PoissonNeuronImplementationData* const rData = (PoissonNeuronImplementationData* const) neuron->implementationData;
  
  PoissonNeuronImplementationData* const nImpl = ((PoissonNeuronImplementationData* const) neuron->implementationData);
  
  // Spike and generate the next Poisson interval 
  if (inputCurrent >= rData->threshold && rData->timeSinceLastSpike >= rData->nextSpikeTime) {
    neuron->isSpiking = true;
    nImpl->timeSinceLastSpike = 0.0;
    nImpl->nextSpikeTime = izk_PoissonNeuron_getNewSpikeTime(rData->spikeRate);
  }
  else {
      neuron->isSpiking = false;
  }
  nImpl->timeSinceLastSpike += stepSize;
    
  assert(izk_PoissonNeuron_repOK(neuron));
}

bool izk_PoissonNeuron_repOK(const Neuron* const neuron) {
  return neuron->voltage == 0.0
          && ((PoissonNeuronImplementationData* const) neuron->implementationData)->nextSpikeTime >= 0.0
          && ((PoissonNeuronImplementationData* const) neuron->implementationData)->timeSinceLastSpike >= 0.0;
}

/** Destructor. */
void izk_PoissonNeuron_free(const Neuron* const neuron) {
    assert(neuron != NULL);
    free(neuron->implementationData); // Free the subclass data
    izk_Neuron_free(neuron); // Free the superclass data
}
