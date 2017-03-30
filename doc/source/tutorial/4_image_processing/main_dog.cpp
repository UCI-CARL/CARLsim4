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
 *
 * CARLsim available from http://socsci.uci.edu/~jkrichma/CARL/CARLsim
 */
#include <carlsim.h>
#include <visual_stimulus.h>

#include <algorithm>
#include <vector>
#include <cassert>
#include <cstdio>

using namespace std;

class ConstantISI : public SpikeGenerator {
public:
	ConstantISI(int numNeur) {
		_numNeur = numNeur;
	}
	~ConstantISI() {}

	void updateISI(unsigned char* stimGray, float maxRateHz=50.0f, float minRateHz=0.0f) {
		_isi.clear();

		// calculate inter-spike interval (ISI) from firing rate
		for (int i=0; i<_numNeur; i++) {
			// convert grayscale value to firing rate
			float rateHz = (float)stimGray[i] / 255.0f * (maxRateHz - minRateHz) + minRateHz;

			// invert firing rate to get inter-spike interval (ISI)
			int isi = (rateHz > 0.0f) ? max(1, (int)(1000 / rateHz)) : 1000000;

			// add value to vector
			_isi.push_back(isi);
		}
	}

	int nextSpikeTime(CARLsim* sim, int grpId, int nid, int currentTime,
		int lastScheduledSpikeTime, int endOfTimeSlice)
	{
		// printf("_numNeur=%d, getGroupNumNeurons=%d\n",_numNeur, sim->getGroupNumNeurons(grpId));
		assert(_numNeur == sim->getGroupNumNeurons(grpId));

		// periodic spiking according to ISI
		return (max(currentTime, lastScheduledSpikeTime) + _isi[nid]);
	}

private:
	int _numNeur;
	std::vector<int> _isi;
};



int main(int argc, const char* argv[]) {
	// ---------------- CONFIG STATE -------------------
	CARLsim sim("dog", GPU_MODE, USER);

	// Input stimulus created from an image using the MATLAB script
	// "createStimFromImage.m":
	VisualStimulus stim("input/carl.dat");
	stim.print();


	// Arrange neurons on a 3D grid, such that every neuron corresponds to
	// pixel: <width x height x channels>. For a grayscale image, the number
	// of channels is 1 -- for an RGB image the number is 3.
	Grid3D imgDim(stim.getWidth(), stim.getHeight(), stim.getChannels());

	// The output group should be smaller than the input, depending on the
	// Gaussian kernel. The number of channels here should be 1, since we
	// will be summing over all color channels.
	Grid3D imgSmallDim(imgDim.numX, imgDim.numY, 1);


	// Input group has firing rates at constant inter-spike intervals (ISI),
	// converted directly from pixel grayscale values
	int gIn = sim.createSpikeGeneratorGroup("input", imgDim, EXCITATORY_NEURON);
	ConstantISI constISI(imgDim.N);
	sim.setSpikeGenerator(gIn, &constISI);


	int gSmoothExc = sim.createGroup("smoothExc", imgSmallDim, EXCITATORY_NEURON);
	sim.setNeuronParameters(gSmoothExc, 0.02f, 0.2f, -65.0f, 8.0f);

	int gSmoothInh = sim.createGroup("smoothInh", imgSmallDim, INHIBITORY_NEURON);
	sim.setNeuronParameters(gSmoothInh, 0.02f, 0.2f, -65.0f, 8.0f);

	// To get edges: subtract gSmoothIn activity from gSmoothExc activity
	int gEdges = sim.createGroup("edges", imgSmallDim, EXCITATORY_NEURON);
	sim.setNeuronParameters(gEdges, 0.02f, 0.2f, -65.0f, 8.0f);


	// Exc: one-to-one in x and y, but summing over color channels
	sim.connect(gIn, gSmoothExc, "gaussian", RangeWeight(10.0f), 1.0f,
		RangeDelay(1), RadiusRF(0.5,0.5,-1));

	// Inh: Blurring the image in x and y, summing over color channels
	sim.connect(gIn, gSmoothInh, "gaussian", RangeWeight(5.0f), 1.0f,
		RangeDelay(1), RadiusRF(3,3,-1));

	// Edges: Exc - Inh
	sim.connect(gSmoothExc, gEdges, "one-to-one", RangeWeight(16.5f), 1.0f,
		RangeDelay(1));
	sim.connect(gSmoothInh, gEdges, "one-to-one", RangeWeight(100.0f), 1.0f,
		RangeDelay(1));

	sim.setConductances(false);


	// ---------------- SETUP STATE -------------------
	sim.setupNetwork();

	sim.setSpikeMonitor(gIn, "DEFAULT");
	sim.setSpikeMonitor(gSmoothExc, "DEFAULT");
	sim.setSpikeMonitor(gSmoothInh, "DEFAULT");
	sim.setSpikeMonitor(gEdges, "DEFAULT");


	// ---------------- RUN STATE -------------------
	for (int i=0; i<stim.getLength(); i++) {
		constISI.updateISI(stim.readFrameChar(), 50.0f, 0.0f);
 		sim.runNetwork(1,0); // run the network
 	}


	return 0;
}
