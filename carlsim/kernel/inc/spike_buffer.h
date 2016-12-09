/* * Copyright (c) 2016 Regents of the University of California. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*
* 1. Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*
* 3. The names of its contributors may not be used to endorse or promote
*    products derived from this software without specific prior written
*    permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
* A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
* LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
* NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* *********************************************************************************************** *
* CARLsim
* created by: (MDR) Micah Richert, (JN) Jayram M. Nageswaran
* maintained by:
* (MA) Mike Avery <averym@uci.edu>
* (MB) Michael Beyeler <mbeyeler@uci.edu>,
* (KDC) Kristofor Carlson <kdcarlso@uci.edu>
* (TSC) Ting-Shuo Chou <tingshuc@uci.edu>
* (HK) Hirak J Kashyap <kashyaph@uci.edu>
*
* CARLsim v1.0: JM, MDR
* CARLsim v2.0/v2.1/v2.2: JM, MDR, MA, MB, KDC
* CARLsim3: MB, KDC, TSC
* CARLsim4: TSC, HK
*
* CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
* Ver 12/31/2016
*/

#ifndef _SPIKE_BUFFER_H_
#define _SPIKE_BUFFER_H_


#include <stdlib.h> // size_t


/*!
 * \brief Circular buffer for delivering spikes
 *
 * This class implements a ring buffer for spike delivery.
 * Spikes are scheduled to be delivered at a time t + delay using SpikeBuffer::schedule. All scheduled spikes can
 * then be retrieved by iterating over the list, from first element SpikeIterator::front until SpikeIterator::back.
 *
 * \since v4.0
 */
class SpikeBuffer {
public:
    /*!
     * \brief SpikeBuffer Constructor
     *
     * A SpikeBuffer is used to schedule and deliver spikes after a certain delay t + delay.
     * Spikes are scheduled to be delivered at a time t + delay using SpikeBuffer::schedule. All scheduled spikes can
     * then be retrieved by iterating over the list, from first element SpikeIterator::front until SpikeIterator::back.
     * \param[in] minDelay Minimum delay (in number of time steps) the buffer can handle
     * \param[in] maxDelay Maximum delay (in number of time steps) the buffer can handle
    */
    SpikeBuffer(int minDelay, int maxDelay);

    /*!
     * \brief SpikeBuffer Destructor
     *
     * The destructor deallocates all scheduled spikes. 
     */
    ~SpikeBuffer();


    // +++++ PUBLIC DATA STRUCTURES +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

    //! linked list to hold the corresponding neuron Id and delivery delay for each spike
    struct SpikeNode {
        int neurId; //!< corresponding global neuron Id
		int grpId; //!< corresponding global group Id
        unsigned short int delay; //!< scheduling delay (in number of time steps)
        SpikeNode* next; //!< pointer to the next element in the list
    };

    //! Iterator to loop over the scheduled spikes at a certain delay
    class SpikeIterator {
    public:
        SpikeIterator() : _node(NULL) {}
        SpikeIterator(SpikeNode* n) : _node(n) {}

        SpikeNode* operator->() {
            return _node;
        }

        int operator*() {
            return _node->neurId;
        }

        bool operator==(const SpikeIterator& other) {
            return (_node == other._node);
        }

        bool operator!=(const SpikeIterator& other) {
            return (_node != other._node);
        }

        inline SpikeIterator* operator++() {
            _node = _node->next;
            return this;
        }

    private:
        SpikeNode* _node;
    };


    /*!
     * \brief Schedule a spike
     *
     * This method schedules a spike to be delivered to neuron with ID neurID, after a delay of t + delay time steps.
     * \param[in] neurId corresponding neuron ID
     * \param[in] delay scheduling delay (in number of time steps)
     */
    void schedule(int neurId, int grpId, unsigned short int delay);

    //! advance to next time step
    void step();

    /*!
     * \brief Reset buffer data
     *
     * This method resets all allocated data. Must be called at the beginning of a simulation.
     * \param[in] minDelay Minimum delay (in number of time steps) the buffer can handle
     * \param[in] maxDelay Maximum delay (in number of time steps) the buffer can handle
     */
    void reset(int minDelay, int maxDelay);

    //! retrieve actual length of the buffer
    size_t length();

    //! pointer to the front of the spike buffer
    SpikeIterator front(int stepOffset=0);

    //! pointer to the back of the spike buffer
    SpikeIterator back();


private:
    // This class provides a pImpl for the CARLsim User API.
    // \see https://marcmutz.wordpress.com/translated-articles/pimp-my-pimpl/
    class Impl;
    Impl* _impl;
};


#endif
