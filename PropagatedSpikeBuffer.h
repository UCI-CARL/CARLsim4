#ifndef PROPAGATEDSPIKEBUFFER_H_
#define PROPAGATEDSPIKEBUFFER_H_

///////////////////////////////////
// IMPORTED FROM PCSIM SOURCE CODE  http://www.lsm.tugraz.at/pcsim/
/////////////////////////////////

//#include "globaldefinitions.h"
//#include "SimParameter.h"

#include <iostream>
using std::cout;
using std::endl;

#include <list>
using std::list;

#include <vector>
using std::vector;

#include <assert.h>

typedef int spikegroupid_t ;

//! Type for specifying delays in time steps
typedef unsigned short int delay_t;

//! Type for specifying the delay in time steps
typedef unsigned short int delaystep_t;

//! The size of one allocation chunk size in PropagatedSpikeBuffer
#define PROPAGATED_SPIKE_BUFFER_CHUNK_SIZE 1024

//! Schedule/Store spikes to be delivered at a later point in the simulation
class PropagatedSpikeBuffer
{

public:

    //! New spike buffer
    /*! \param minDelay Minimum delay (in number of time steps) the buffer can handle
    *  \param maxDelay Maximum delay (in number of time steps) the buffer can handle
    *  \param chunkSize Constant used for internal memory management.
    *                   Should increase with increasing number of synapses and
    *                   exptected number of spikes.
    */
    PropagatedSpikeBuffer(int minDelay,
                          int maxDelay,
                          int chunkSize = PROPAGATED_SPIKE_BUFFER_CHUNK_SIZE );

    //! Destructor: Deletes all scheduled spikes
    virtual ~PropagatedSpikeBuffer();

    //! Schedule a group of spike targets to get a spike at time t + delay
    /*! \param stg  The identifier of the spike target group to be used as
     *              identifier in SpikeTargetGroupPool::beginSpikeTarget( stg )
     *  \param delay The number of time steps to delay the deliver of the spike
     */
    void scheduleSpikeTargetGroup(spikegroupid_t stg, delaystep_t delay);

    //! Structure which stores the index of the spike target group and a pointer to the next element in various lists
    struct StgNode
    {
        spikegroupid_t stg;
        delaystep_t delay;
        StgNode *next;
    };

    //! Iterator to loop over the scheduled spikes at a certain delay
    class const_iterator
    {
    public:
        const_iterator(): node(NULL) {};
        const_iterator(StgNode *n): node(n) {};

        StgNode* operator->() { return node; }
        spikegroupid_t operator*() { return node->stg; }

        bool operator==(const const_iterator& other) { return ( this->node == other.node ); }

        bool operator!=(const const_iterator& other) { return ( this->node != other.node ); }

        inline const_iterator& operator++() { node = node->next; return *this; }

    private:
        StgNode *node;
    };

    //! Returns an iterator to loop over all scheduled spike target groups
    /*! \param stepOffset Determines at which position ( current timestep + stepOffset )
     * the spike target groups are read of
     */
    const_iterator beginSpikeTargetGroups(int stepOffset = 0)
    {
#ifdef DEBUG
        // this assertion fails if stepOffset < -length()
        if (const_iterator( ringBufferFront[ (currIdx + stepOffset + length() ) % length()  ] ) != NULL)
            //        cout << "in beginSTG currT=" << currT << " stepOffset=" << stepOffset << endl;
            assert( (currIdx + stepOffset + length() ) % length() >= 0 );
#endif
        return const_iterator( ringBufferFront[ (currIdx + stepOffset + length() ) % length()  ] );
    };

    //! End iterator corresponding to beginSynapseGroups
    const_iterator endSpikeTargetGroups()
    {
        return const_iterator(NULL);
    };

    //! Must be called to tell the buffer that it should move on to the next time step
    void nextTimeStep();

    //! Must be called at the begin (reset) of a simulation
    /*! \param minDelay Minimum delay (in number of time steps) the buffer can handle
     *  \param maxDelay Maximum delay (in number of time steps) the buffer can handle
     */
    void reset(int minDelay, int maxDelay);

    //! Return the actual length of the buffer
    inline size_t length() { return ringBufferFront.size(); };

    // PropagatedSpikeBuffer & operator=(const PropagatedSpikeBuffer &src);

private :

    //! Set up internal memory management
    void init(size_t maxDelaySteps);

    //! Get a new StgNode from the internal memory management
    StgNode *getFreeNode(void);

    //! The index into the ring buffer which corresponds to the current time step
    int currIdx ;

    //! A ring buffer storing a pointer to the first StgNode of the scheduled spike receiving groups
    vector< StgNode* > ringBufferFront;

    //! A ring buffer storing a pointer to the last StgNode of the scheduled spike receiving groups
    vector< StgNode* > ringBufferBack;

    //! Buffer with pointers to chunks of memory allocated
    vector< StgNode* > chunkBuffer;

    //! Pointer to array of SrgNodes to take from (from position nextFreeSrgNodeIdx)
    StgNode *currentFreeChunk;

    //! Index into array currentFreeChunk which specifies the next StgNode to use
    int  nextFreeSrgNodeIdx;

    //! Index into chunkBuffer which specifies the next free chunk of memory to use
    size_t nextFreeChunkIdx;

    //! Head of list of SrgNodes which can be reused
    StgNode* recycledNodes;

    //! Number of SrgNodes per chunk of memory to allocate
    int chunkSize;

    int currT;

    double fillitup[32];
};

#endif /*PROPAGATEDSPIKEBUFFER_H_*/
