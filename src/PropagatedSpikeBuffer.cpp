///////////////////////////////////
// IMPORTED FROM PCSIM SOURCE CODE  
// http://www.lsm.tugraz.at/pcsim/
/////////////////////////////////

#include "PropagatedSpikeBuffer.h"

#include <iostream>

using std::cout;
using std::endl;

PropagatedSpikeBuffer::PropagatedSpikeBuffer(int minDelay, int maxDelay, int chunkSize ):
        currIdx(0),
        ringBufferFront(maxDelay+1),
        ringBufferBack(maxDelay+1),
        chunkBuffer(0),
        currentFreeChunk(NULL),
        nextFreeSrgNodeIdx(0),
        nextFreeChunkIdx(0),
        recycledNodes(NULL),
        chunkSize(chunkSize)
{
    // Check arguments
    //assert( minDelay <= maxDelay );
    //assert( minDelay >= 0 );
    //assert( maxDelay >= 0 );
    
    cout << "Ringbuffer size is: " << ringBufferFront.size() << endl; 

    reset( minDelay, maxDelay );

    currT = 0;
}


/* PropagatedSpikeBuffer & PropagatedSpikeBuffer::operator=(const PropagatedSpikeBuffer &src)
{
    *this = src;
    currentFreeChunk = chunkBuffer[0];
    return *this;
} */

PropagatedSpikeBuffer::~PropagatedSpikeBuffer()
{
    for(size_t i=0; i<chunkBuffer.size(); i++) {
        delete[] chunkBuffer[i];
    }
}

void PropagatedSpikeBuffer::init(size_t maxDelaySteps)
{
    //! Check arguments
    //assert( maxDelaySteps > 0 );

    if( ringBufferFront.size() != maxDelaySteps + 1 ) {
        ringBufferFront.resize( maxDelaySteps + 1 );
        ringBufferBack.resize( maxDelaySteps + 1 );
    }
    if( chunkBuffer.size() < 1 ) {
        chunkBuffer.reserve( 10 );
        chunkBuffer.resize( 0 );
        chunkBuffer.push_back( new StgNode[ chunkSize ] );
    }
}

void PropagatedSpikeBuffer::reset(int minDelay, int maxDelay)
{
    //assert( minDelay <= maxDelay );
    //assert( minDelay >= 0 );
    //assert( maxDelay >= 0 );

    init( maxDelay + minDelay );

    for(size_t i=0; i<ringBufferFront.size(); i++) {
        ringBufferFront[i] = ringBufferBack[i] = NULL;
    }

    currentFreeChunk = chunkBuffer[0];
    nextFreeChunkIdx = 1;

    nextFreeSrgNodeIdx  = 0;

    recycledNodes = NULL;

    currIdx = 0;
}

PropagatedSpikeBuffer::StgNode *PropagatedSpikeBuffer::getFreeNode(void)
{
    StgNode *n;
    if (recycledNodes != NULL) {
        // get a recycled node
        n = recycledNodes;
        recycledNodes = recycledNodes->next;
    } else if ( nextFreeSrgNodeIdx < chunkSize ) {
        // get slot from the current (allocIdx) pre-allocated memory chunk
        n = &(currentFreeChunk[nextFreeSrgNodeIdx++]);
    } else if (nextFreeChunkIdx < chunkBuffer.size() ) {
        // current (currentFreeChunk) pre-allocated memory chunk used up: go to next chunk
        currentFreeChunk = chunkBuffer[nextFreeChunkIdx++];
        n = &(currentFreeChunk[0]);
        nextFreeSrgNodeIdx = 1;
    } else {
        // no more chunks available: alloc a new one
        currentFreeChunk = new StgNode[chunkSize];
        chunkBuffer.push_back( currentFreeChunk );
        nextFreeChunkIdx++;
        n = &(currentFreeChunk[0]);
        nextFreeSrgNodeIdx = 1;
    }
    return n;
}

void PropagatedSpikeBuffer::scheduleSpikeTargetGroup(spikegroupid_t stg, delaystep_t delay)
{
    //cout << "in buffer timestep = " << currT << " delay=" << delay << endl;
    StgNode *n = getFreeNode();

    //assert( n != NULL );

    int writeIdx = ( currIdx + delay ) % ringBufferFront.size();

    n->stg    = stg;
    n->delay  = delay;
    n->next   = NULL;
    if( ringBufferFront[writeIdx] == NULL ) {
        ringBufferBack[writeIdx] = ringBufferFront[writeIdx] = n;
    } else {
       ringBufferBack[writeIdx]->next = n;
       ringBufferBack[writeIdx] = n;
    }    
}

void PropagatedSpikeBuffer::nextTimeStep()
{
    // move the solts of currIdx to recycled slots
    //    cout << ">>> curIDx="  << currIdx << ", bksz=" << ringBufferBack.size() << ", frsz=" << ringBufferFront.size() << endl;
    if( ringBufferFront[ currIdx ] != NULL ) {
        ringBufferBack[ currIdx ]->next = recycledNodes;
        recycledNodes = ringBufferFront[ currIdx ];
    }
    // mark current index as processed
    //  cout << ">>> curIDx="  << currIdx << ", bksz=" << ringBufferBack.size() << ", frsz=" << ringBufferFront.size() << endl;
    ringBufferBack[ currIdx ] = ringBufferFront[ currIdx ] = NULL;
    currIdx = ( currIdx + 1 ) % ringBufferFront.size();
    currT ++;
}


