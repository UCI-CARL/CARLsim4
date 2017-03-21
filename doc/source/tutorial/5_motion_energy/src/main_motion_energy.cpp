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
 */
#include <carlsim.h>

#include <motion_energy.h>
#include <visual_stimulus.h>

#include <vector>
#include <cassert>

#include <cstdio>



int main(int argc, const char* argv[]) {
	// ---------------- CONFIG STATE -------------------
	bool onGPU = true;
	int frameDurMs = 50;
	int numDir = 8;
	double speed = 1.5;

	CARLsim sim("me", onGPU?GPU_MODE:CPU_MODE, USER);

	// Input stimulus created using the MATLAB script "createGratingVideo.m"
	VisualStimulus stim("input/grating.dat");
	stim.print();

	// The MotionEnergy object has to be initialized with the stimulus size.
	MotionEnergy me(stim.getWidth(), stim.getHeight(), stim.getChannels());
	unsigned char* frame;

	// We want to create neuronal populations responding to `numDir` different
	// directions of motion at every pixel of the stimulus.
	Grid3D gridV1(stim.getWidth(), stim.getHeight(), numDir);
	Grid3D gridMT(stim.getWidth()/2, stim.getHeight()/2, numDir);

	// Primary visual cortex (V1) neurons will receive input from the `MotionEnergy`
	// object. Filter responses of the `MotionEnergy` object will be converted to
	// mean firing rates of a Poisson process (using the `PoissonRate` object).
	int gV1=sim.createSpikeGeneratorGroup("V1", gridV1, EXCITATORY_NEURON);


	// Middle temporal area (MT) neurons will sum over V1 neurons using a Gaussian
	// kernel in x and y (7x7 pixels), but will not sum across directions (the third
	// dimension of the grid). Thus the `RadiusRF` struct should be initialized as
	// `RadiusRF(7,7,0)`.
	int gMT=sim.createGroup("MT", gridMT, EXCITATORY_NEURON);
	sim.setNeuronParameters(gMT, 0.02f, 0.2f, -65.0f, 8.0f);
	sim.connect(gV1, gMT, "gaussian", RangeWeight(0.02), 1.0f, RangeDelay(1), RadiusRF(7,7,0));

	// Use COBA mode
	sim.setConductances(true);


	// ---------------- SETUP STATE -------------------
	sim.setupNetwork();

	PoissonRate rateV1(gridV1.N, onGPU);
	sim.setSpikeRate(gV1, &rateV1);

	sim.setSpikeMonitor(gV1, "DEFAULT");
	sim.setSpikeMonitor(gMT, "DEFAULT");


	// ---------------- RUN STATE -------------------
	for (int i=0; i<stim.getLength(); i++) {
		// Repeated calls to this function will ready one frame at a time from the
		// stimulus file, automatically advancing to the next frame.
		frame = stim.readFrameChar();

		// The `MotionEnergy` object has a bunch of intermediate steps that could be
		// accessed, such as the V1 linear filter response (excluding normalization, etc.).
		// Here we want to convert a `frame` of the stimulus to V1 complex cell responses,
		// and use them to fill up the `rateV1` container. Thus we have to pass the right
		// PoissonRate pointer (could live either on the host or the device).
		// Since the `MotionEnergy` object does all calculations on the GPU, passing the
		// `onGPU` flag will avoid copying all data from GPU to CPU, only to copy it back
		// to a PoissonRate container on the GPU.
		me.calcV1complex(frame, onGPU?rateV1.getRatePtrGPU():rateV1.getRatePtrCPU(),
			speed, onGPU);

		sim.runNetwork(frameDurMs/1000, frameDurMs%1000);
	}

	return 0;
}
