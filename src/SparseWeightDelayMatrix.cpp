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

#include "SparseWeightDelayMatrix.h"

SparseWeightDelayMatrix::SparseWeightDelayMatrix(int Npre, int Npost, int initSize) {
	count = 0;
	size = 0;
	weights = NULL;
	maxWeights = NULL;
	preIds = NULL;
	preIds = NULL;
	delay_opts = NULL;

	maxPreId = 0;
	maxPostId = 0;

	resize(initSize);
}

SparseWeightDelayMatrix::~SparseWeightDelayMatrix() {
	free(weights);
	free(maxWeights);
	free(preIds);
	free(postIds);
	free(delay_opts);
}

void SparseWeightDelayMatrix::resize(int inc) {
	size += inc;

	weights = (float*)realloc(weights, size * sizeof(float));
	maxWeights = (float*)realloc(maxWeights, size * sizeof(float));
	preIds = (unsigned int*)realloc(preIds, size * sizeof(int));
	postIds = (unsigned int*)realloc(postIds, size * sizeof(int));
	delay_opts = (unsigned int*)realloc(delay_opts, size * sizeof(int));
}

int SparseWeightDelayMatrix::add(int preId, int postId, float weight, float maxWeight, uint8_t delay, int opts) {
	if (count == size)
		resize(size == 0 ? 1000 : size * 2);

	weights[count] = weight;
	maxWeights[count] = maxWeight;
	preIds[count] = preId;
	postIds[count] = postId;
	delay_opts[count] = delay | (opts << 8);

	if (preId > maxPreId)
		maxPreId = preId;

	if (postId > maxPostId)
		maxPostId = postId;

	return ++count;
}

