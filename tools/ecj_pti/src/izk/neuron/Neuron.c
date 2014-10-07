#include "Neuron.h"

/** Default implementation of virtual destructor.  This should be called
 * by the subclass's implementation of free.
 *
 * This method should *not* be called until the subclass has freed
 * neuron->privateData.  */
void izk_Neuron_free(const Neuron* const neuron) {
    // We do not free the function table, because function tables should not be
    // defined in dynamic memory.
    free(neuron);
}
