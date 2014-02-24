/* 
 * Copyright (c) 2013 Regents of the University of California. All rights reserved.
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
 * created by: 		(MDR) Micah Richert, (JN) Jayram M. Nageswaran
 * maintained by:	(MA) Mike Avery <averym@uci.edu>, (MB) Michael Beyeler <mbeyeler@uci.edu>,
 *					(KDC) Kristofor Carlson <kdcarlso@uci.edu>
 *					(TSC) Ting-Shuo Chou <tingshuc@uci.edu>
 *
 * CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
 * Ver 07/13/2013
 */

#ifndef _SPARSE_WEIGHT_DELAY_MATRIX_H_
#define _SPARSE_WEIGHT_DELAY_MATRIX_H_

#include <stdlib.h>
#include <stdint.h>

class SparseWeightDelayMatrix {
public:
	uint32_t	count;
	uint32_t	size;
	uint32_t	maxPreId;
	uint32_t	maxPostId;
	float		*weights;
	float		*maxWeights;
	float		*mulSynFast;	//!< scaling factor for fast synaptic current (AMPA / GABAa)
	float 		*mulSynSlow;	//!< scaling factor for slow synaptic current (NMDA / GABAb)
	uint32_t	*preIds;
	uint32_t	*postIds;
	uint32_t	*delay_opts; //!< first 8 bits are delay, higher are for Fixed/Plastic and any other future options

	SparseWeightDelayMatrix(int Npre, int Npost, int initSize = 0);

	~SparseWeightDelayMatrix();

	void resize(int inc);

	int add(int preId, int postId, float weight, float maxWeight, uint8_t delay, float _mulSynFast, float _mulSynSlow,
				int opts = 0);
};

#endif