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
 * CARLsim available from http://socsci.uci.edu/~jkrichma/CARL/CARLsim/
 * Ver 12/3/2014
 */

#include <carlsim.h>
#include <stdio.h>

#if (WIN32 || WIN64)
	#define _CRT_SECURE_NO_WARNINGS
#endif

int main() {
	// ---------------- CONFIG STATE -------------------
	CARLsim sim("Hello World", CPU_MODE, USER);
	int gIn = sim.createSpikeGeneratorGroup("input", 1, EXCITATORY_NEURON);
	int gOut = sim.createGroup("output", 1, EXCITATORY_NEURON);
	sim.setNeuronParameters(gOut, 0.1f, 0.2f, -65.0f, 2.0f);
	sim.setConductances(true);
	// connect our groups
	sim.connect(gIn, gOut, "one-to-one", RangeWeight(0.05f), 1.0f);

	// ---------------- SETUP STATE -------------------
	sim.setupNetwork();
	sim.setSpikeMonitor(gOut,"DEFAULT");

	// ---------------- RUN STATE -------------------
	PoissonRate in(1);
	in.setRates(20.0f);
	sim.setSpikeRate(gIn, &in);
	sim.runNetwork(10,0);

	return 0;
}
