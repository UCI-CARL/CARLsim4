/* * Copyright (c) 2015 Regents of the University of California. All rights reserved.
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
*
* CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
* Ver 5/22/2015
*/

// include CARLsim user interface
#include <carlsim.h>

int main() {
	// Declare and initialize a CARLsim object in either GPU or CPU mode.
	CARLsim* sim = new CARLsim("compartments", GPU_MODE, USER, 0, 42);
	// Set the integration method used for updating neurons' voltage & recovery values.
	// RUNGE_KUTTA4 is more accurate than FORWARD_EULER, but is also more computationally intensive.
	// Larger number of timesteps delivers more accuracy, but is also more computationally intensive.
	sim->setIntegrationMethod(RUNGE_KUTTA4, 30);

	// In this tutorial we assign 5 neurons to each compartment group.
	// Note: all compartmentally connected groups must have the same number of neurons.
	int N = 5;

	// Each one of the groups below represents a single layer.
	int grpSoma = sim->createGroup("excit", N, EXCITATORY_NEURON);    // grpSoma 
	int grpD_one = sim->createGroup("excit", N, EXCITATORY_NEURON);   // grpD_one
	int grpD_two = sim->createGroup("excit", N, EXCITATORY_NEURON);   // grpD_two 
	int grpD_three = sim->createGroup("excit", N, EXCITATORY_NEURON); // grpD_three 

	sim->setNeuronParameters(grpSoma, 550.0f, 2.3330991f, -59.101414f, -50.428886f, 0.0021014998f,
		-0.41361538f, 24.98698f, -53.223213f, 109.0f); // 9 parameter setNeuronParametersCall (Regular Spiking NEURON) (soma)
	sim->setNeuronParameters(grpD_one, 367.0f, 1.1705916f, -59.101414f, -44.298294f, 0.2477681f,
		3.3198094f, 20.274296f, -46.076824f, 24.0f);   // 9 parameter setNeuronParametersCall (Regular Spiking NEURON) (dendr)
	sim->setNeuronParameters(grpD_two, 425.0f, 2.2577047f, -59.101414f, -25.137894f, 0.32122386f,
		0.14995363f, 13.203414f, -38.54892f, 69.0f);   // 9 parameter setNeuronParametersCall (Regular Spiking NEURON) (dendr)
	sim->setNeuronParameters(grpD_three, 225.0f, 1.109572f, -59.101414f, -36.55802f, 0.29814243f,
		-4.385603f, 21.473854f, -40.343994f, 21.0f);   // 9 parameter setNeuronParametersCall (Regular Spiking NEURON) (dendr)

	// Set the coupling (up & down) constants for each layer. 
	sim->setCompartmentParameters(grpD_one, 28.396f, 5.526f); // grpD_one 28 and 5
	sim->setCompartmentParameters(grpD_two, 50.474f, 0.0f);   // grpD_two 50 and 0
	sim->setCompartmentParameters(grpD_three, 0.0f, 49.14f);  // grpD_three 0 and 49
	sim->setCompartmentParameters(grpSoma, 116.861f, 4.60f);  // grpSoma (somatic) 116 and 4

	// Create a dummy spike generator & connect it to one of the layers.
	// CARLsim requires such a connection in order to run the network. 
	int gin = sim->createSpikeGeneratorGroup("input", N, EXCITATORY_NEURON);
	sim->connect(gin, grpSoma, "one-to-one", RangeWeight(0.0f), 1.0f, RangeDelay(1), RadiusRF(-1));

	// We will ignore conductances for this tutorial.
	sim->setConductances(0);

	// Connect the 4 groups (layers) compartmentally.
	// The layers are arranged in following manner:
	// D-3
	// ||
	// Soma
	// ||
	// D-1
	// ||
	// D-2
	sim->connectCompartments(grpD_two, grpD_one);
	sim->connectCompartments(grpD_one, grpSoma);
	sim->connectCompartments(grpSoma, grpD_three);

	// Ignore STDP for this tutorial.
	sim->setSTDP(grpD_one, false);
	sim->setSTDP(grpD_two, false);
	sim->setSTDP(grpD_three, false);
	sim->setSTDP(grpSoma, false);

	sim->setupNetwork();

	// Set-up spike monitors so that we can observe the neurons' spike times.

	SpikeMonitor* spkMonSoma = sim->setSpikeMonitor(grpSoma, "DEFAULT"); // put spike times into file
	SpikeMonitor* spkMonDOne = sim->setSpikeMonitor(grpD_one, "DEFAULT"); // put spike times into file
	SpikeMonitor* spkMonDTwo = sim->setSpikeMonitor(grpD_two, "DEFAULT"); // put spike times into file
	SpikeMonitor* spkMonDthree = sim->setSpikeMonitor(grpD_three, "DEFAULT"); // put spike times into file

	spkMonSoma->startRecording();
	spkMonDOne->startRecording();
	spkMonDTwo->startRecording();
	spkMonDthree->startRecording();

	// Continuously inject 600mA of current into soma layer.
	sim->setExternalCurrent(grpSoma, 600);
	sim->runNetwork(1, 0);

	spkMonSoma->stopRecording();
	spkMonDOne->stopRecording();
	spkMonDTwo->stopRecording();
	spkMonDthree->stopRecording();

	// Print out the spike times.
	spkMonSoma->print();
	spkMonDOne->print();
	spkMonDTwo->print();
	spkMonDthree->print();

	// Expected spike times: soma should fire due to the current being injected into it.
	// Each one of the 5 neurons in some group (layer) should fire at the ~following times:
	// 48 86 135 215 383 553 722 891
	// The exact times depend on the integration method & number of timesteps.
	// Each one of the 5 neurons in dendrite-3 group (layer) should fire at the ~following times:
	// 48 85 135 215 383 552 721 890
	// The dendrite-1 and dendrite-2 groups (layers) will be silent.
	// This is due to the way coupling constants are arranged. 

	delete sim;

	return 0;
}