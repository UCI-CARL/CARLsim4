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
#include "gtest/gtest.h"
#include "carlsim_tests.h"

#include <carlsim.h>
#include <vector>
#include <math.h> // sqrt

/// **************************************************************************************************************** ///
/// Connect FUNCTIONALITY
/// **************************************************************************************************************** ///

TEST(Connect, connectFull) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("Connect.connectFull",CPU_MODE,SILENT,1,42);
	Grid3D grid(2,3,4);
	int g0=sim->createGroup("excit0", grid, EXCITATORY_NEURON);
	int g1=sim->createGroup("excit1", grid, EXCITATORY_NEURON);
	int g2=sim->createGroup("excit2", grid, EXCITATORY_NEURON);
	int g3=sim->createGroup("excit3", grid, EXCITATORY_NEURON);
	int g4=sim->createGroup("excit4", grid, EXCITATORY_NEURON);
	int g5=sim->createGroup("excit5", grid, EXCITATORY_NEURON);
	sim->setNeuronParameters(g0, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g3, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g4, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g5, 0.02f, 0.2f, -65.0f, 8.0f);

	int c0=sim->connect(g0,g0,"full",RangeWeight(0.1), 1.0, RangeDelay(1)); // full
	int c1=sim->connect(g1,g1,"full",RangeWeight(0.1), 1.0, RangeDelay(1), RadiusRF(-1.0, 0.0, 0.0)); // all in x
	int c2=sim->connect(g2,g2,"full",RangeWeight(0.1), 1.0, RangeDelay(1), RadiusRF(-1.0, 0.0, -1.0)); // all in x and z
	int c3=sim->connect(g3,g3,"full",RangeWeight(0.1), 1.0, RangeDelay(1), RadiusRF(2.0, 3.0, 0.0)); // 2D ellipse x,y
	int c4=sim->connect(g4,g4,"full",RangeWeight(0.1), 1.0, RangeDelay(1), RadiusRF(0.0, 2.0, 3.0)); // 2D ellipse y,z
	int c5=sim->connect(g5,g5,"full",RangeWeight(0.1), 1.0, RangeDelay(1), RadiusRF(2.0, 2.0, 2.0)); // 3D ellipsoid

	sim->setupNetwork(); // need SETUP state for this function to work

	EXPECT_EQ(sim->getNumSynapticConnections(c0), grid.N * grid.N);
	EXPECT_EQ(sim->getNumSynapticConnections(c1), grid.N * grid.numX);
	EXPECT_EQ(sim->getNumSynapticConnections(c2), grid.N * grid.numX * grid.numZ);
	EXPECT_EQ(sim->getNumSynapticConnections(c3), 144); // these numbers are less than what the grid would
	EXPECT_EQ(sim->getNumSynapticConnections(c4), 224); // say because of edge effects
	EXPECT_EQ(sim->getNumSynapticConnections(c5), 320);

	delete sim;
}

TEST(Connect, connectFullNoDirect) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("CORE.connectRadiusRF",CPU_MODE,SILENT,1,42);
	Grid3D grid(2,3,4);
	int g0=sim->createGroup("excit0", grid, EXCITATORY_NEURON);
	int g1=sim->createGroup("excit1", grid, EXCITATORY_NEURON);
	int g2=sim->createGroup("excit2", grid, EXCITATORY_NEURON);
	int g3=sim->createGroup("excit3", grid, EXCITATORY_NEURON);
	int g4=sim->createGroup("excit4", grid, EXCITATORY_NEURON);
	int g5=sim->createGroup("excit5", grid, EXCITATORY_NEURON);
	sim->setNeuronParameters(g0, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g3, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g4, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g5, 0.02f, 0.2f, -65.0f, 8.0f);

	int c0=sim->connect(g0,g0,"full-no-direct",RangeWeight(0.1), 1.0, RangeDelay(1)); // full
	int c1=sim->connect(g1,g1,"full-no-direct",RangeWeight(0.1), 1.0, RangeDelay(1), RadiusRF(-1.0, 0.0, 0.0)); // all in x
	int c2=sim->connect(g2,g2,"full-no-direct",RangeWeight(0.1), 1.0, RangeDelay(1), RadiusRF(-1.0, 0.0, -1.0)); // all in x,z
	int c3=sim->connect(g3,g3,"full-no-direct",RangeWeight(0.1), 1.0, RangeDelay(1), RadiusRF(2.0, 3.0, 0.0)); // ellipse x,y
	int c4=sim->connect(g4,g4,"full-no-direct",RangeWeight(0.1), 1.0, RangeDelay(1), RadiusRF(0.0, 2.0, 3.0)); // ellipse y,z
	int c5=sim->connect(g5,g5,"full-no-direct",RangeWeight(0.1), 1.0, RangeDelay(1), RadiusRF(2.0, 2.0, 2.0)); // ellipsoid

	sim->setupNetwork(); // need SETUP state for this function to work

	// same as connect full, but no self-connections
	EXPECT_EQ(sim->getNumSynapticConnections(c0), grid.N * (grid.N-1) );
	EXPECT_EQ(sim->getNumSynapticConnections(c1), grid.N * (grid.numX-1) );
	EXPECT_EQ(sim->getNumSynapticConnections(c2), grid.N * (grid.numX*grid.numZ-1) );
	EXPECT_EQ(sim->getNumSynapticConnections(c3), 120); // these numbers are less than what the grid would
	EXPECT_EQ(sim->getNumSynapticConnections(c4), 200); // say because of edge effects
	EXPECT_EQ(sim->getNumSynapticConnections(c5), 296);

	delete sim;
}

TEST(Connect, connectOneToOne) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("Connect.connectOneToOne",CPU_MODE,SILENT,1,42);
	Grid3D grid(2,3,4);
	int g0=sim->createGroup("excit0", grid, EXCITATORY_NEURON);
	int g1=sim->createGroup("excit1", grid, EXCITATORY_NEURON);
	int g2=sim->createGroup("excit2", grid, EXCITATORY_NEURON);
	sim->setNeuronParameters(g0, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);

	int c0=sim->connect(g0,g0,"one-to-one",RangeWeight(0.1), 1.0, RangeDelay(1)); // full
	int c1=sim->connect(g1,g1,"one-to-one",RangeWeight(0.1), 1.0, RangeDelay(1), RadiusRF(-1.0, 0.0, 0.0)); // all in x
	int c2=sim->connect(g2,g2,"one-to-one",RangeWeight(0.1), 1.0, RangeDelay(1), RadiusRF(-1.0, 0.0, -1.0)); // all in x,z

	sim->setupNetwork(); // need SETUP state for this function to work

	// RadiusRF should not matter
	EXPECT_EQ(sim->getNumSynapticConnections(c0), grid.N);
	EXPECT_EQ(sim->getNumSynapticConnections(c1), grid.N);
	EXPECT_EQ(sim->getNumSynapticConnections(c2), grid.N);

	delete sim;
}

TEST(Connect, connectRandom) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("Connect.connectRandom",CPU_MODE,SILENT,1,42);
	Grid3D grid(2,3,4);
	int g0=sim->createGroup("excit0", grid, EXCITATORY_NEURON);
	int g1=sim->createGroup("excit1", grid, EXCITATORY_NEURON);
	int g2=sim->createGroup("excit2", grid, EXCITATORY_NEURON);
	int g3=sim->createGroup("excit3", grid, EXCITATORY_NEURON);
	int g4=sim->createGroup("excit4", grid, EXCITATORY_NEURON);
	int g5=sim->createGroup("excit5", grid, EXCITATORY_NEURON);
	sim->setNeuronParameters(g0, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g3, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g4, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g5, 0.02f, 0.2f, -65.0f, 8.0f);

	double prob = 0.2;
	int c0=sim->connect(g0,g0,"random",RangeWeight(0.1), prob, RangeDelay(1)); // full
	int c1=sim->connect(g1,g1,"random",RangeWeight(0.1), prob, RangeDelay(1), RadiusRF(-1,0,0)); // all in x
	int c2=sim->connect(g2,g2,"random",RangeWeight(0.1), prob, RangeDelay(1), RadiusRF(-1,0,-1)); // all in x and z
	int c3=sim->connect(g3,g3,"random",RangeWeight(0.1), prob, RangeDelay(1), RadiusRF(2,3,0)); // 2D ellipse x,y
	int c4=sim->connect(g4,g4,"random",RangeWeight(0.1), prob, RangeDelay(1), RadiusRF(0,2,3)); // 2D ellipse y,z
	int c5=sim->connect(g5,g5,"random",RangeWeight(0.1), prob, RangeDelay(1), RadiusRF(2,2,2)); // 3D ellipsoid

	sim->setupNetwork(); // need SETUP state for this function to work

	// from SNN::connect: estimate max number of connections needed using binomial distribution
	// at 7.5 standard deviations
	int errorMargin = 7.5*sqrt(prob*(1-prob)*grid.N)+0.5;
	EXPECT_NEAR(sim->getNumSynapticConnections(c0), prob * grid.N * grid.N, errorMargin);
	EXPECT_NEAR(sim->getNumSynapticConnections(c1), prob * grid.N * grid.numX, errorMargin);
	EXPECT_NEAR(sim->getNumSynapticConnections(c2), prob * grid.N * grid.numX * grid.numZ, errorMargin);
	EXPECT_NEAR(sim->getNumSynapticConnections(c3), prob*144, errorMargin); // these numbers are less than what the
	EXPECT_NEAR(sim->getNumSynapticConnections(c4), prob*224, errorMargin); // grid would say because of edge effects
	EXPECT_NEAR(sim->getNumSynapticConnections(c5), prob*320, errorMargin);

	delete sim;
}


TEST(Connect, connectGaussian) {
	CARLsim* sim = NULL;

	// test all possible cases, where any dimension < 0, == 0, or some real-valued RF > 0
	for (int xRad=-1; xRad<=1; xRad++) {
		for (int yRad=-1; yRad<=1; yRad++) {
			for (int zRad=-1; zRad<=1; zRad++) {
				// generate RadiusRF struct from iter vars
				// xRad=-1 and xRad=0 are input as is, if xRad=1 we use an actual real-valued radius
				RadiusRF radius( (xRad>0)?xRad*2.0:xRad, (yRad>0)?yRad*3.0:yRad, (zRad>0)?zRad*4.0:zRad );

				if (radius.radX<0 && radius.radY<0 && radius.radZ<0) {
					// not allowed
					continue;
				}

				sim = new CARLsim("CORE.connectGaussian",CPU_MODE,SILENT,1,42);
				Grid3D grid(11,11,11);
				int g0=sim->createGroup("excit", grid, EXCITATORY_NEURON);
				sim->setNeuronParameters(g0, 0.02f, 0.2f, -65.0f, 8.0f);

				double wt = 0.1;
				int c0=sim->connect(g0, g0, "gaussian", RangeWeight(wt), 1.0, RangeDelay(1), radius);

				sim->setupNetwork();

				ConnectionMonitor* CM0 = sim->setConnectionMonitor(g0,g0,"NULL");
				std::vector< std::vector<float> > wt0 = CM0->takeSnapshot();
				EXPECT_EQ(wt0.size(), grid.N);
				EXPECT_EQ(wt0[0].size(), grid.N);

				int nSyn = 0;
				for (int i=0; i<wt0.size(); i++) {
					Point3D pre = sim->getNeuronLocation3D(g0, i);
					for (int j=0; j<wt0[0].size(); j++) {
						Point3D post = sim->getNeuronLocation3D(g0, j);

						// prepare RF distance
						double rfDist = -1.0;

						// inverse semi-principal axes of the ellipsoid
						// avoid division by zero by working with inverse of semi-principal axes (set to large value)
						double aInv = (radius.radX>0) ? 1.0/radius.radX : 1e+20;
						double bInv = (radius.radY>0) ? 1.0/radius.radY : 1e+20;
						double cInv = (radius.radZ>0) ? 1.0/radius.radZ : 1e+20;

						// there are 27 different cases to consider
						if (radius.radX < 0) {
							// x < 0
							if (radius.radY < 0) {
								// x < 0 && y < 0
								if (radius.radZ < 0) {
									// x < 0 && y < 0 && z < 0
									rfDist = 0.0; // always true
								} else if (radius.radZ == 0) {
									// x < 0 && y < 0 && z == 0
									rfDist = (pre.z == post.z) ? 0.0 : -1.0;
								} else {
									// x < 0 && y < 0 && z > 0
									rfDist = (pre.z-post.z)*(pre.z-post.z)*cInv*cInv;
								}
							} else if (radius.radY == 0) {
								// x < 0 && y == 0
								if (radius.radZ < 0) {
									// x < 0 && y == 0 && z < 0
									rfDist = (pre.y == post.y) ? 0.0 : -1.0;
								} else if (radius.radZ == 0) {
									// x < 0 && y == 0 && z == 0
									rfDist = (pre.y == post.y && pre.z == post.z) ? 0.0 : -1.0;
								} else {
									// x < 0 && y == 0 && z > 0
									rfDist = (pre.y == post.y) ? (pre.z-post.z)*(pre.z-post.z)*cInv*cInv : -1.0;
								}
							} else {
								// x < 0 && y > 0
								if (radius.radZ < 0) {
									// x < 0 && y > 0 && z < 0
									rfDist = (pre.y-post.y)*(pre.y-post.y)*bInv*bInv;
								} else if (radius.radZ == 0) {
									// x < 0 && y > 0 && z == 0
									rfDist = (pre.z == post.z) ? (pre.y-post.y)*(pre.y-post.y)*bInv*bInv : -1.0;
								} else {
									// x < 0 && y > 0 && z > 0
									rfDist = (pre.y-post.y)*(pre.y-post.y)*bInv*bInv + (pre.z-post.z)*(pre.z-post.z)*cInv*cInv;
								}
							}
						} else if (radius.radX == 0) {
							// x == 0
							if (radius.radY < 0) {
								// x == 0 && y < 0
								if (radius.radZ < 0) {
									// x == 0 && y < 0 && z < 0
									rfDist = (pre.x == post.x) ? 0.0 : -1.0;
								} else if (radius.radZ == 0) {
										// x == 0 && y < 0 && z == 0
									rfDist = (pre.x == post.x && pre.z == post.z) ? 0.0 : -1.0;
								} else {
									// x == 0 && y < 0 && z > 0
									rfDist = (pre.x == post.x) ? (pre.z-post.z)*(pre.z-post.z)*cInv*cInv : -1.0;
								}
							} else if (radius.radY == 0) {
								// x == 0 && y == 0
								if (radius.radZ < 0) {
									// x == 0 && y == 0 && z < 0
									rfDist = (pre.x == post.x && pre.y == post.y) ? 0.0 : -1.0;
								} else if (radius.radZ == 0) {
									// x == 0 && y == 0 && z == 0
									rfDist = (pre.x == post.x && pre.y == post.y && pre.z == post.z) ? 0.0 : -1.0;
								} else {
									// x == 0 && y == 0 && z > 0
									rfDist = (pre.x == post.x && pre.y == post.y) ? (pre.z-post.z)*(pre.z-post.z)*cInv*cInv : -1.0;
								}
							} else {
								// x == 0 && y > 0
								if (radius.radZ < 0) {
									// x == 0 && y > 0 && z < 0
									rfDist = (pre.x == post.x) ? (pre.y-post.y)*(pre.y-post.y)*bInv*bInv : -1.0;
								} else if (radius.radZ == 0) {
									// x == 0 && y > 0 && z == 0
									rfDist = (pre.x == post.x && pre.z == post.z) ? (pre.y-post.y)*(pre.y-post.y)*bInv*bInv : -1.0;
								} else {
									// x == 0 && y > 0 && z > 0
									rfDist = (pre.x == post.x) ? (pre.y-post.y)*(pre.y-post.y)*bInv*bInv+(pre.z-post.z)*(pre.z-post.z)*cInv*cInv : -1.0;
								}
							}
						} else {
							// x > 0
							if (radius.radY < 0) {
								// x > 0 && y < 0
								if (radius.radZ < 0) {
									// x > 0 && y < 0 && z < 0
									rfDist = (pre.x-post.x)*(pre.x-post.x)*aInv*aInv;
								} else if (radius.radZ == 0) {
									// x > 0 && y < 0 && z == 0
									rfDist = (pre.z == post.z) ? (pre.x-post.x)*(pre.x-post.x)*aInv*aInv : -1.0;
								} else {
									// x > 0 && y < 0 && z > 0
									rfDist = (pre.x-post.x)*(pre.x-post.x)*aInv*aInv + (pre.z-post.z)*(pre.z-post.z)*cInv*cInv;
								}
							} else if (radius.radY == 0) {
								// x > 0 && y == 0
								if (radius.radZ < 0) {
									// x > 0 && y == 0 && z < 0
									rfDist = (pre.y == post.y) ? (pre.x-post.x)*(pre.x-post.x)*aInv*aInv : -1.0;
								} else if (radius.radZ == 0) {
									// x > 0 && y == 0 && z == 0
									rfDist = (pre.y == post.y && pre.z == post.z) ? (pre.x-post.x)*(pre.x-post.x)*aInv*aInv : -1.0;
								} else {
									// x > 0 && y == 0 && z > 0
									rfDist = (pre.y == post.y) ? (pre.x-post.x)*(pre.x-post.x)*aInv*aInv+(pre.z-post.z)*(pre.z-post.z)*cInv*cInv : -1.0;
								}
							} else {
								// x > 0 && y > 0
								if (radius.radZ < 0) {
									// x > 0 && y > 0 && z < 0
									rfDist = (pre.x-post.x)*(pre.x-post.x)*aInv*aInv + (pre.y-post.y)*(pre.y-post.y)*bInv*bInv;
								} else if (radius.radZ == 0) {
									// x > 0 && y > 0 && z == 0
									rfDist = (pre.z == post.z) ? (pre.x-post.x)*(pre.x-post.x)*aInv*aInv+(pre.y-post.y)*(pre.y-post.y)*bInv*bInv : -1.0;
								} else {
									// x > 0 && y > 0 && z > 0
									rfDist = (pre.x-post.x)*(pre.x-post.x)*aInv*aInv+(pre.y-post.y)*(pre.y-post.y)*bInv*bInv+(pre.z-post.z)*(pre.z-post.z)*cInv*cInv;
								}
							}
						}

						if (rfDist < 0.0 || rfDist > 1.0) {
							// RF distance is not valid or too large
							EXPECT_TRUE(isnan(wt0[i][j]));
						} else {
							// RF distance seems ok, compute gaussian weight
							double gaussWt = exp(-2.3026*rfDist);
							if (gaussWt < 0.1) {
								// gaussian cut-off, weight would be < 10 % of max
								EXPECT_TRUE(isnan(wt0[i][j]));
							} else {
								// check weight and update synapse number
								nSyn++;
								EXPECT_FLOAT_EQ(wt0[i][j], gaussWt*wt);
							}
						}
					}
				}
				EXPECT_EQ(sim->getNumSynapticConnections(c0), nSyn);
				delete sim;
			}
		}
	}
}
