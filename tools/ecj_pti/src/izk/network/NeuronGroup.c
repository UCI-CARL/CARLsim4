#include "NeuronGroup.h"
#include "../neuron/Neuron.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/** NeuronGroup.c
 *
 *
 * @author Eric 'Siggy' Scott
 */

/** Constructor. */
const NeuronGroup* izk_makeNeuronGroup(const Neuron * const neuronPrototype, const int length) {
    assert(neuronPrototype != NULL);
    assert(length > 0);
    
    Neuron ** const newNeuronArray = malloc(length*sizeof(*newNeuronArray));
    int i;
    for(i = 0; i < length; i++) {
        newNeuronArray[i] = neuronPrototype->funcTable->copy(neuronPrototype);
    }
    
    const NeuronGroup stackNeuronGroup = { newNeuronArray, length };
    NeuronGroup* const neuronGroup = malloc(sizeof( *neuronGroup));
    memcpy(neuronGroup, &stackNeuronGroup, sizeof(NeuronGroup));
    return neuronGroup;
}

/** Execute a simulation step on every neuron in the order the appear in the array. */
void izk_NeuronGroup_stepAll(const NeuronGroup* const obj, const double* const inputCurrents, const double stepSize) {
    int i;
    for (i = 0; i < obj->length; i++) {
        Neuron* const neuron = obj->neuronArray[i];
        neuron->funcTable->step(neuron, inputCurrents[i], stepSize);
    }
}

char * izk_NeuronGroup_printVoltage(const NeuronGroup* const obj) {
    int i;
    const unsigned int numDigits = 4;
    char * const str = malloc((obj->length + 1) * (numDigits + 4) * sizeof(*str));
    sprintf(str, "%.*f", numDigits, obj->neuronArray[0]->voltage);
    for (i = 1; i < obj->length; i++) {
        const Neuron * const neuron = obj->neuronArray[i];
        char nextStr[numDigits + 2];
        sprintf(nextStr, ", %.*f", numDigits, neuron->voltage);
        strcat(str, nextStr);
    }
    return str;
}

char * izk_NeuronGroup_printIsSpiking(const NeuronGroup* const obj) {
    int i;
    char * const str = malloc((obj->length + 1) * (2) * sizeof(*str));
    sprintf(str, "%d", obj->neuronArray[0]->isSpiking);
    for (i = 1; i < obj->length; i++) {
        const Neuron * const neuron = obj->neuronArray[i];
        char nextStr[4];
        sprintf(nextStr, ", %d", neuron->isSpiking);
        strcat(str, nextStr);
    }
    return str;
}

/** Destructor. */
void izk_NeuronGroup_free(const NeuronGroup* const obj) {
    int i;
    for (i = 0; i < obj->length; i++) {
        obj->neuronArray[i]->funcTable->free(obj->neuronArray[i]);
    }
    // Casting away constness here, because I don't interpret deallocating memory as modifying it.
    free((void *) obj->neuronArray);
    free((void *) obj);
}
