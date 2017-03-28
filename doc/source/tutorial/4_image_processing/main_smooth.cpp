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

#include <vector>
#include <cassert>

#include <cstdio>



int main(int argc, const char* argv[]) {
	// ---------------- CONFIG STATE -------------------
	CARLsim sim("smooth", GPU_MODE, USER);

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
	Grid3D imgSmallDim(imgDim.numX/2, imgDim.numY/2, 1);

	int gIn = sim.createSpikeGeneratorGroup("input", imgDim, EXCITATORY_NEURON);
	int gSmooth = sim.createGroup("smooth", imgSmallDim, EXCITATORY_NEURON);
	sim.setNeuronParameters(gSmooth, 0.02f, 0.2f, -65.0f, 8.0f);

	// The connect call takes care of the Gaussian smoothing: We define a
	// 5x5 Gaussian kernel in x and y. The -1 says that we shall sum over all
	// color channels (3rd dimension of the grid). If we wanted to smooth only
	// within channels, we would use RadiusRF(5,5,0) and make sure we adjust
	// imgSmallDim to have 3 color channels above.
	sim.connect(gIn, gSmooth, "gaussian", RangeWeight(2.0f), 1.0f,
		RangeDelay(1), RadiusRF(5,5,-1));

	// Use CUBA mode
	sim.setConductances(false);


	// ---------------- SETUP STATE -------------------
	sim.setupNetwork();

	sim.setSpikeMonitor(gIn, "DEFAULT");
	sim.setSpikeMonitor(gSmooth, "DEFAULT");


	// ---------------- RUN STATE -------------------
	for (int i=0; i<stim.getLength(); i++) {
		PoissonRate* rates = stim.readFramePoisson(50.0f, 0.0f);
		sim.setSpikeRate(gIn, rates);
 		sim.runNetwork(1,0); // run the network
 	}


	return 0;
}
