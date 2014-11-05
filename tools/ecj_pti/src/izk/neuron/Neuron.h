#ifndef IZK_NEURON_H
#define IZK_NEURON_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

/** Neuron.h
 *
 * An abstract interface for defining neurons.  This uses function pointers
 * to define virtual functions -- defining them is equivalent to implementing
 * a subclass.
 * 
 * @author Eric 'Siggy' Scott
 */

// Data for parent type
typedef struct Neuron {
    double voltage;
    bool isSpiking;
    const struct NeuronFunctionTable * const funcTable;
    void * const implementationData; // Subclass-specific data
} Neuron;

// Dispatch vector providing pointers to virtual methods
typedef struct NeuronFunctionTable {
    void (* const step)(Neuron * obj, double inputCurrent, double stepSize); // Execute an Euler step of the simulation
    bool (* const repOK)(const Neuron * obj); // Representation invariant
    Neuron* (* const copy)(const Neuron * obj); // Copy constructor
    void (* const free)(const Neuron * obj); // Destructor
} NeuronFunctionTable;

// Default implementation of virtual method -- should be used like a call to "super.free()" in Java
void izk_Neuron_free(const Neuron* obj);

#ifdef __cplusplus
}
#endif

#endif /* IZK_NEURON_H */
