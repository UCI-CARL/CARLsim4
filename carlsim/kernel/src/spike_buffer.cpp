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
#include <spike_buffer.h>

#include <vector>


// the size of an allocation chunk
#define MAX_CHUNK_SIZE 1024


class SpikeBuffer::Impl {
public:
	// +++++ PUBLIC METHODS: SETUP / TEAR-DOWN ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	Impl(int minDelay, int maxDelay) : _currNodeId(0), _spikeBufFront(maxDelay+1), 
		_spikeBufBack(maxDelay+1), _chunkBuf(0), _currFreeChunkId(NULL), _nextFreeNodeId(0), _nextFreeChunkId(0),
		_recycledNodes(NULL)
	{
		reset(minDelay, maxDelay);
	}
	
	~Impl() {
		for (size_t i=0; i<_chunkBuf.size(); i++) {
			delete[] _chunkBuf[i];
		}
	}

	void reset(int minDelay, int maxDelay) {
		init(maxDelay + minDelay);

		for (size_t i=0; i<_spikeBufFront.size(); i++) {
			_spikeBufFront[i] = NULL;
			_spikeBufBack[i] = NULL;
		}

		_currFreeChunkId = _chunkBuf[0];
		_nextFreeChunkId = 1;

		_currNodeId = 0;
		_nextFreeNodeId = 0;
		_recycledNodes = NULL;
	}



	// +++++ PUBLIC METHODS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	// points to front of buffer
	SpikeIterator front(int stepOffset=0) {
		return SpikeIterator(_spikeBufFront[(_currNodeId + stepOffset + length()) % length()]);
	};

	// End iterator corresponding to beginSynapseGroups
	SpikeIterator back() {
		return SpikeIterator(NULL);
	};

	// retrieve actual length of buffer
	size_t length() {
		return _spikeBufFront.size();
	}
	
	// schedule a spike at t + delay for neuron neurId
	void schedule(int neurId, int grpId, unsigned short int delay) {
		SpikeNode* n = getFreeNode();

		int writeIdx = (_currNodeId+delay) % _spikeBufFront.size();

		n->neurId = neurId;
		n->grpId = grpId;
		n->delay = delay;
		n->next = NULL;
		if (_spikeBufFront[writeIdx] == NULL) {
			_spikeBufFront[writeIdx] = n;
			_spikeBufBack[writeIdx] = n;
		} else {
			_spikeBufBack[writeIdx]->next = n;
			_spikeBufBack[writeIdx] = n;
		}
	}

	void step() {
	// move the solts of _currNodeId to recycled slots
		if (_spikeBufFront[_currNodeId] != NULL) {
			_spikeBufBack[_currNodeId]->next = _recycledNodes;
			_recycledNodes = _spikeBufFront[_currNodeId];
		}
	
		// mark current index as processed
		_spikeBufFront[_currNodeId] = NULL;
		_spikeBufBack[_currNodeId] = NULL;
		_currNodeId = (_currNodeId + 1) % _spikeBufFront.size();
	}


private:
	//! Set up internal memory management
	void init(size_t maxDelaySteps) {
		if (_spikeBufFront.size() != maxDelaySteps + 1) {
			_spikeBufFront.resize(maxDelaySteps + 1);
			_spikeBufBack.resize(maxDelaySteps + 1);
		}
		if (_chunkBuf.size() < 1) {
			_chunkBuf.reserve(10);
			_chunkBuf.resize(0);
			_chunkBuf.push_back(new SpikeNode[MAX_CHUNK_SIZE]);
		}
	}

	//! Get a new SpikeNode from the internal memory management
	SpikeNode* getFreeNode() {
		SpikeNode* n;
		if (_recycledNodes != NULL) {
			// find a recycled node
			n = _recycledNodes;
			_recycledNodes = _recycledNodes->next;
		} else if (_nextFreeNodeId < MAX_CHUNK_SIZE) {
			// as long as there is pre-allocated memory left: get a new slot from the current chunk
			n = &(_currFreeChunkId[_nextFreeNodeId++]);
		} else if (_nextFreeChunkId < _chunkBuf.size()) {
			// pre-allocated memory chunk is used up: go to next chunk
			_currFreeChunkId = _chunkBuf[_nextFreeChunkId++];
			n = &(_currFreeChunkId[0]);
			_nextFreeNodeId = 1;
		} else {
			// all chunks used up: need to allocate new one
			_currFreeChunkId = new SpikeNode[MAX_CHUNK_SIZE];
			_chunkBuf.push_back(_currFreeChunkId);

			_nextFreeChunkId++;
			_nextFreeNodeId = 1;
			n = &(_currFreeChunkId[0]);
		}
		return n;
	}

	//! The index into the ring buffer which corresponds to the current time step
	int _currNodeId;

	//! A ring buffer storing a pointer to the first SpikeNode
	std::vector<SpikeNode*> _spikeBufFront;

	//! A ring buffer storing a pointer to the last SpikeNode
	std::vector<SpikeNode*> _spikeBufBack;

	//! Buffer with pointers to chunks of pre-allocated memory
	std::vector<SpikeNode*> _chunkBuf;

	//! Pointer to array of SpikeNodes to take from (from position _nextFreeNodeId)
	SpikeNode* _currFreeChunkId;

	//! Index into array _currFreeChunkId which specifies the next SpikeNode to use
	int _nextFreeNodeId;

	//! Index into _chunkBuf which specifies the next free chunk of memory to use
	size_t _nextFreeChunkId;

	//! Head of list of SpikeNodes which can be reused
	SpikeNode* _recycledNodes;
};


// ****************************************************************************************************************** //
// SPIKEBUFFER API IMPLEMENTATION
// ****************************************************************************************************************** //

// constructor / destructor
SpikeBuffer::SpikeBuffer(int minDelay, int maxDelay) : 
	_impl( new Impl(minDelay, maxDelay) ) {}
SpikeBuffer::~SpikeBuffer() { delete _impl; }

// public methods
void SpikeBuffer::schedule(int neurId, int grpId, unsigned short int delay) { _impl->schedule(neurId, grpId, delay); }
void SpikeBuffer::step() { _impl->step(); }
void SpikeBuffer::reset(int minDelay, int maxDelay) { _impl->reset(minDelay, maxDelay); }
size_t SpikeBuffer::length() { return _impl->length(); }
SpikeBuffer::SpikeIterator SpikeBuffer::front(int stepOffset) { return _impl->front(stepOffset); }
SpikeBuffer::SpikeIterator SpikeBuffer::back() { return _impl->back(); }
