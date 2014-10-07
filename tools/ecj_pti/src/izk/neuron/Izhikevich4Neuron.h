#ifndef IZK_IZHIKEVICH4_NEURON_H
#define IZK_IZHIKEVICH4_NEURON_H

#ifdef __cplusplus
extern "C" {
#endif
    
#include "Neuron.h"
#include <stdbool.h>

/** Izhikevich4.h
 *
 * A 4-parameter Izhikevich neuron.
 *
 * This file is a class that implements the Neuron interface (see the body of
 * izk_makeIzhikevich4Neuron -- the constructor -- to understand how).
 * 
 * @author Eric 'Siggy' Scott
 */

// Private class members
typedef struct Izhikevich4NeuronImplementationData {
  const double a, b, c, d;
  float recovery;
} Izhikevich4NeuronImplementationData;

/** Constructor.  Allocates a new neuron on the heap. */
Neuron * izk_makeIzhikevich4Neuron (const double a, const double b, const double c, const double d);
/** Copy constructor.  ref must be an Izhikevich4Neuron. */
Neuron * izk_copyIzhikevich4Neuron (const Neuron * ref);

// Function table for public methods
// Right now this is identical to NeuronFunctionTable, but we can add
// subclass-specific methods here.
typedef struct Izhikevich4NeuronFunctionTable {
    void (*step)(Neuron * obj, double inputCurrent, double stepSize);
    bool (*repOK)(const Neuron * obj);
    Neuron* (* const copy)(const Neuron * obj); // Copy constructor
    void (* const free)(const Neuron * obj); // Destructor
} Izhikevich4NeuronFunctionTable;

#ifdef __cplusplus
}
#endif

#endif /* IZK_IZHIKEVICH4_NEURON_H */
