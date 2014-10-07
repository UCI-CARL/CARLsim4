#ifndef IZK_NEURON_GROUP_H
#define IZK_NEURON_GROUP_H

#ifdef __cplusplus
extern "C" {
#endif
    
#include "../neuron/Neuron.h"

/** NeuronGroup.h
 *
 * Data structure holding an array of Neurons.
 *
 * @author Eric 'Siggy' Scott
 */
typedef struct NeuronGroup {
    /** Since Neuron has const members, there would be no clean way to
     * initialize an array of Neurons. So we use an array of pointers to Neurons.  */
    Neuron* const * const neuronArray; 
    const int length;
} NeuronGroup;

/** Constructor. Create a NeuronGroup on the heap.
 * @param neuronPrototype Copies of this neuron will be used to create the group. 
 * @param length Number of neurons in the group. */
const NeuronGroup * izk_makeNeuronGroup(const Neuron * const neuronPrototype, const int length);

void izk_NeuronGroup_stepAll(const NeuronGroup* const obj, const double* const inputCurrents, const double stepSize);

char * izk_NeuronGroup_printIsSpiking(const NeuronGroup* const obj);

char* izk_NeuronGroup_printVoltage(const NeuronGroup* const obj);

void izk_NeuronGroup_free(const NeuronGroup* const obj);

#ifdef __cplusplus
}
#endif

#endif /* IZK_NEURON_GROUP_H */
